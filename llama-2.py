import random
import string
import time
import json
from typing import List, Optional

from fastapi import FastAPI, Request, HTTPException
from starlette.responses import StreamingResponse, JSONResponse

from ray import serve

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ErrorResponse,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_engine import LoRAModulePath
from transformers import AutoTokenizer
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("ray.serve")

app = FastAPI()

model_name = "NousResearch/Llama-2-7b-chat-hf"
tp_size = 8

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
class VLLMDeployment:
    def __init__(
        self,
        engine_args: AsyncEngineArgs,
        response_role: str,
        lora_modules: Optional[List[LoRAModulePath]] = None,
    ):
        self.openai_serving_chat = None
        self.engine_args = engine_args
        self.response_role = response_role
        self.lora_modules = lora_modules

        tokenizer = AutoTokenizer.from_pretrained(self.engine_args.model)
        self.chat_template = None
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
            try:
                if isinstance(tokenizer.chat_template, str):
                    self.chat_template = json.loads(tokenizer.chat_template)
                elif isinstance(tokenizer.chat_template, dict):
                    self.chat_template = tokenizer.chat_template
                else:
                    logger.warning(f"Unexpected chat_template type: {type(tokenizer.chat_template)}")
            except json.JSONDecodeError:
                logger.warning("Failed to parse chat template as JSON. Using default.")
        
        if self.chat_template is None:
            logger.warning("No valid chat template found in the model. Using default.")

        logger.info(f"Initializing VLLMDeployment with model: {self.engine_args.model}")
        logger.info(f"Tensor parallel size: {self.engine_args.tensor_parallel_size}")
        logger.info(f"Data type: {self.engine_args.dtype}")

        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    @app.post("/v1/chat/completions")
    async def create_chat_completion(self, request: Request):        
        try:
            body = await request.json()
            logger.debug(f"Received chat completion request: {body}")
            vllm_request = ChatCompletionRequest(**body)

            if not self.openai_serving_chat:
                logger.info("Initializing OpenAIServingChat")
                model_config = await self.engine.get_model_config()
                if self.engine_args.served_model_name is not None:
                    served_model_names = self.engine_args.served_model_name
                else:
                    served_model_names = [self.engine_args.model]
                self.openai_serving_chat = OpenAIServingChat(
                    self.engine,
                    model_config,
                    served_model_names,
                    self.response_role,
                    lora_modules=self.lora_modules,
                    chat_template=self.chat_template,
                    prompt_adapters=None,
                    request_logger=None
                )

            logger.debug(f"Calling create_chat_completion with request: {vllm_request}")
            generator = await self.openai_serving_chat.create_chat_completion(
                vllm_request, request
            )
            if isinstance(generator, ErrorResponse):
                raise HTTPException(status_code=generator.code, detail=generator.message)

            if vllm_request.stream:
                async def openai_stream_generator():
                    async for chunk in generator:
                        yield f"data: {json.dumps(chunk.model_dump())}\n\n"
                    yield "data: [DONE]\n\n"
                return StreamingResponse(openai_stream_generator(), media_type="text/event-stream")
            else:
                response = await generator.__anext__()
                return JSONResponse(content={
                    "id": "chatcmpl-" + ''.join(random.choices(string.ascii_letters + string.digits, k=29)),
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": self.engine_args.model,
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
                "model": self.engine_args.model,
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
                    "id": self.engine_args.model,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "organization",
                }
            ]
        })

    @app.get("/health")
    async def health_check(self):
        return {"status": "ok"}

def build_app(model_name, tensor_parallel_size) -> serve.Application:
    tp = tensor_parallel_size
    engine_args = AsyncEngineArgs(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        dtype="float16",
        worker_use_ray=True,
    )
    logger.info(f"Tensor parallelism = {tp}")
    pg_resources = []
    pg_resources.append({"CPU": 1})  # for the deployment replica
    for i in range(tp):
        pg_resources.append({"CPU": 1, "GPU": 1})  # for the vLLM actors

    return VLLMDeployment.options(
        placement_group_bundles=pg_resources, placement_group_strategy="STRICT_PACK"
    ).bind(
        engine_args,
        response_role="assistant",
        lora_modules=None,
    )

deployment = build_app(model_name=model_name, tensor_parallel_size=tp_size)
