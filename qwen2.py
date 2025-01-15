from typing import Dict, Optional, List
import logging
import os
import json
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse
from starlette.middleware.cors import CORSMiddleware

from ray import serve

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_engine import LoRAModulePath
from transformers import AutoTokenizer
import huggingface_hub

logger = logging.getLogger("ray.serve")
logger.setLevel(logging.INFO)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@dataclass
class ModelPath:
    name: str
    path: str

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
        
        # Initialize HuggingFace token
        self.hf_token = os.environ.get("HUGGING_FACE_TOKEN")
        if not self.hf_token:
            raise ValueError("HUGGING_FACE_TOKEN environment variable is not set")
        huggingface_hub.login(token=self.hf_token)

        # Create model paths with proper structure
        if isinstance(self.engine_args.served_model_name, str):
            served_names = [self.engine_args.served_model_name]
        elif isinstance(self.engine_args.served_model_name, (list, tuple)):
            served_names = self.engine_args.served_model_name
        else:
            served_names = None

        if served_names:
            self.base_model_paths = [
                ModelPath(name=name, path=name) 
                for name in served_names
            ]
        else:
            self.base_model_paths = [
                ModelPath(name=self.engine_args.model, path=self.engine_args.model)
            ]

        logger.info(f"Initialized base model paths: {self.base_model_paths}")

        # Initialize tokenizer and chat template
        tokenizer = AutoTokenizer.from_pretrained(self.engine_args.model)
        self.chat_template = None
        self.chat_template_content_format = "text"

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

        # Initialize engine
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
                self.openai_serving_chat = OpenAIServingChat(
                    self.engine,
                    model_config,   
                    self.base_model_paths,
                    self.response_role,
                    lora_modules=self.lora_modules,
                    chat_template=self.chat_template,
                    chat_template_content_format=self.chat_template_content_format,
                    prompt_adapters=None,
                    request_logger=None
                )

            logger.debug(f"Calling create_chat_completion with request: {vllm_request}")
            generator = await self.openai_serving_chat.create_chat_completion(
                vllm_request, request
            )

            if isinstance(generator, ErrorResponse):
                return JSONResponse(
                    content=generator.model_dump(), 
                    status_code=generator.code
                )

            if vllm_request.stream:
                return StreamingResponse(
                    content=generator, 
                    media_type="text/event-stream"
                )
            else:
                if not isinstance(generator, ChatCompletionResponse):
                    generator = await anext(generator)
                return JSONResponse(
                    content=generator.model_dump()
                )

        except Exception as e:
            logger.error(f"Error in create_chat_completion: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

def build_app(model_name: str, tensor_parallel_size: int) -> serve.Application:
    """Builds the Serve app with the specified model configuration."""
    engine_args = AsyncEngineArgs(
        model=model_name,
        served_model_name=model_name,
        tensor_parallel_size=tensor_parallel_size,
        worker_use_ray=True,
        rope_scaling = {
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 32768,
            "beta_fast": 32,
            "beta_slow": 1
        }
    )

    logger.info(f"Tensor parallelism = {tensor_parallel_size}")
    
    # Configure placement group resources
    pg_resources = [{"CPU": 1}]  # for the deployment replica
    for _ in range(tensor_parallel_size):
        pg_resources.append({"CPU": 1, "GPU": 1})  # for the vLLM actors

    return VLLMDeployment.options(
        placement_group_bundles=pg_resources, 
        placement_group_strategy="STRICT_PACK"
    ).bind(
        engine_args,
        response_role="assistant",
        lora_modules=None,
    )

# Initialize the deployment
deployment = build_app(
    model_name="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4", 
    tensor_parallel_size=4
)