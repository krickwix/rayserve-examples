import random
import string
import time
import json
from typing import List, Optional, Dict

from fastapi import FastAPI, Request, HTTPException
from starlette.responses import StreamingResponse, JSONResponse

from ray import serve

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ErrorResponse,
    ChatCompletionResponse,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_engine import LoRAModulePath
from transformers import AutoTokenizer
import logging
import os
# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("ray.serve")
logger.setLevel(logging.DEBUG)

app = FastAPI()

# Define multiple models
models = {
    "model1": {
        "name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "tp_size": 4
    },
    "model2": {
        "name": "meta-llama/Llama-2-7b-chat-hf",
        "tp_size": 4
    }
}

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.debug(f"Received request: {request.method} {request.url}")
    logger.debug(f"Request headers: {request.headers}")
    
    response = await call_next(request)
    
    logger.debug(f"Response status: {response.status_code}")
    logger.debug(f"Response headers: {response.headers}")
    
    return response

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": {"message": str(exc), "type": "internal_server_error"}}
    )

@serve.deployment(
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 10,
        "target_ongoing_requests": 5,
    },
    max_ongoing_requests=10,
)
@serve.ingress(app)
class MultiModelVLLMDeployment:
    def __init__(
        self,
        model_configs: Dict[str, Dict],
        response_role: str,
        lora_modules: Optional[List[LoRAModulePath]] = None,
    ):
        import huggingface_hub
        self.model_configs = model_configs
        self.response_role = response_role
        self.lora_modules = lora_modules
        self.engines = {}
        self.openai_serving_chats = {}
        self.chat_templates = {}

        for model_id, config in self.model_configs.items():
            engine_args = AsyncEngineArgs(
                model=config['name'],
                tensor_parallel_size=config['tp_size'],
                worker_use_ray=True,
            )
            self.hf_token = os.environ.get("HUGGING_FACE_TOKEN")
            if not self.hf_token:
                raise ValueError("HUGGING_FACE_TOKEN environment variable is not set")
            huggingface_hub.login(token=self.hf_token)

            logger.info(f"Initializing engine for {model_id} with model: {config['name']}")
            self.engines[model_id] = AsyncLLMEngine.from_engine_args(engine_args)

            tokenizer = AutoTokenizer.from_pretrained(config['name'])
            self.chat_templates[model_id] = None
            if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
                try:
                    if isinstance(tokenizer.chat_template, str):
                        self.chat_templates[model_id] = json.loads(tokenizer.chat_template)
                    elif isinstance(tokenizer.chat_template, dict):
                        self.chat_templates[model_id] = tokenizer.chat_template
                    else:
                        logger.warning(f"Unexpected chat_template type for {model_id}: {type(tokenizer.chat_template)}")
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse chat template as JSON for {model_id}. Using default.")
            
            if self.chat_templates[model_id] is None:
                logger.warning(f"No valid chat template found for {model_id}. Using default.")

            logger.info(f"Initialized engine for {model_id} with model: {config['name']}")
            logger.info(f"Tensor parallel size: {config['tp_size']}")

    async def get_openai_serving_chat(self, model_id: str):
        if model_id not in self.openai_serving_chats:
            engine = self.engines[model_id]
            model_config = await engine.get_model_config()
            served_model_name = self.model_configs[model_id]['name']
            self.openai_serving_chats[model_id] = OpenAIServingChat(
                engine,
                model_config,
                [served_model_name],
                self.response_role,
                lora_modules=self.lora_modules,
                chat_template=self.chat_templates[model_id],
                prompt_adapters=None,
                request_logger=None
            )
        return self.openai_serving_chats[model_id]

    @app.post("/v1/chat/completions")
    async def create_chat_completion(self, request: Request):
        try:
            body = await request.json()
            logger.debug(f"Received chat completion request: {body}")
            vllm_request = ChatCompletionRequest(**body)

            # Determine which model to use
            model_id = body.get('model', 'model1')  # Default to model1 if not specified
            if model_id not in self.model_configs:
                raise HTTPException(status_code=400, detail=f"Invalid model: {model_id}")

            openai_serving_chat = await self.get_openai_serving_chat(model_id)

            logger.debug(f"Calling create_chat_completion with request: {vllm_request}")
            generator_or_response = await openai_serving_chat.create_chat_completion(
                vllm_request, request
            )
            if isinstance(generator_or_response, ErrorResponse):
                raise HTTPException(status_code=generator_or_response.code, detail=generator_or_response.message)

            if vllm_request.stream:
                async def openai_stream_generator():
                    async for chunk in generator_or_response:
                        yield f"data: {json.dumps(chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                return StreamingResponse(openai_stream_generator(), media_type="text/event-stream")
            else:
                if isinstance(generator_or_response, ChatCompletionResponse):
                    response = generator_or_response
                else:
                    response = await generator_or_response.__anext__()
                return JSONResponse(content={
                    "id": "chatcmpl-" + ''.join(random.choices(string.ascii_letters + string.digits, k=29)),
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": self.model_configs[model_id]['name'],
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response.choices[0].message.content,
                        },
                        "finish_reason": response.choices[0].finish_reason,
                    }],
                    "usage": response.usage.model_dump() if response.usage else None,
                })
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON")
        except StopIteration:
            return JSONResponse(content={
                "id": "chatcmpl-" + ''.join(random.choices(string.ascii_letters + string.digits, k=29)),
                "object": "chat.completion",
                "created": int(time.time()),
                "model": self.model_configs[model_id]['name'],
                "choices": [],
                "usage": None,
            })
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/v1/models")
    async def list_models(self):
        return JSONResponse(content={
            "data": [
                {
                    "id": config['name'],
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "organization",
                }
                for config in self.model_configs.values()
            ]
        })

    @app.get("/health")
    async def health_check(self):
        return {"status": "ok"}

def build_app(model_configs: Dict[str, Dict]) -> serve.Application:
    pg_resources = []
    for config in model_configs.values():
        tp = config['tp_size']
        for _ in range(tp):
            pg_resources.append({"CPU": 1, "GPU": 1})  # for the vLLM actors
    print(pg_resources)
    return MultiModelVLLMDeployment.options(
        placement_group_bundles=pg_resources, placement_group_strategy="PACK"
    ).bind(
        model_configs=model_configs,
        response_role="assistant",
        lora_modules=None,
    )

deployment = build_app(model_configs=models)