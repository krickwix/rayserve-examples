from typing import Dict, Optional, List
import logging
import os
import json
from dataclasses import dataclass
import ast

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
from vllm.entrypoints.openai.serving_models import (BaseModelPath,
                                                    OpenAIServingModels)

from transformers import AutoTokenizer
import huggingface_hub

logger = logging.getLogger("ray.serve")
logger.setLevel(logging.DEBUG)

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
        "max_replicas": 4,
        "target_ongoing_requests": 1,
    },
    max_ongoing_requests=10,
)

@serve.ingress(app)
class VLLMDeployment:
    def __init__(
        self,
        engine_args: AsyncEngineArgs,
        response_role: str,
    ):
        self.openai_serving_chat = None
        self.engine_args = engine_args
        self.response_role = response_role
        self.lora_modules = None
        
        logger.debug(f'initialization: engine_args = {engine_args}')

        # Initialize HuggingFace token
        self.hf_token = os.environ.get("HUGGING_FACE_TOKEN")
        if not self.hf_token:
            raise ValueError("HUGGING_FACE_TOKEN environment variable is not set")
        huggingface_hub.login(token=self.hf_token)

        # Create model paths with proper structure
        # if isinstance(self.engine_args.served_model_name, str):
        #     served_names = [self.engine_args.served_model_name]
        # elif isinstance(self.engine_args.served_model_name, (list, tuple)):
        #     served_names = self.engine_args.served_model_name
        # else:
        #     served_names = None

        # logger.debug(f'initialization: served_names = {served_names}')

        # if served_names:
        # self.base_model_paths = [
        #     ModelPath(name=name, path=name) 
        #     for name in served_names
        # ]
        # else:
        #     self.base_model_paths = [
        #         ModelPath(name=self.engine_args.model, path=self.engine_args.model)
        #     ]

        # logger.debug(f"Initialized base model paths: {self.base_model_paths}")

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
                MODEL_NAME = self.engine_args.model
                BASE_MODEL_PATHS = [BaseModelPath(name=MODEL_NAME, model_path=MODEL_NAME)]
                models = OpenAIServingModels(self.engine, model_config, BASE_MODEL_PATHS)
                self.openai_serving_chat = OpenAIServingChat(
                    self.engine,
                    model_config,   
                    models,
                    self.response_role,
                    # lora_modules=self.lora_modules,
                    chat_template=self.chat_template,
                    chat_template_content_format=self.chat_template_content_format,
                    # prompt_adapters=None,
                    request_logger=None,
                    enable_auto_tool_choice=True,
                    tool_call_parser="mistral"
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

def parse_engine_args(args_str: str) -> dict:
    """
    Parse engine arguments from a string representation into a dictionary.
    Handles nested structures and common Python types.
    """
    if not args_str:
        return {}
    try:
        # Convert string representation to Python literal structure
        parsed_args = ast.literal_eval(args_str)
        if not isinstance(parsed_args, dict):
            raise ValueError("ENGINE_ARGS must evaluate to a dictionary")
        return parsed_args
    except (SyntaxError, ValueError) as e:
        logger.error(f"Failed to parse ENGINE_ARGS: {e}")
        raise ValueError(f"Invalid ENGINE_ARGS format: {e}")

def build_app(model_name: str, tensor_parallel_size: int) -> serve.Application:
    """Builds the Serve app with the specified model configuration."""

    # Get additional engine arguments from environment
    engine_args_str = os.getenv("ENGINE_ARGS", "{}")
    additional_args = parse_engine_args(engine_args_str)

    # Base engine arguments
    base_args = {
        "model": model_name,
        "tensor_parallel_size": tensor_parallel_size,
        "distributed_executor_backend": "ray",
        "trust_remote_code": True,
    }
    # Merge base arguments with additional arguments
    combined_args = {**base_args, **additional_args}
    logger.info(f"Initializing engine with arguments: {combined_args}")
    engine_args = AsyncEngineArgs(**combined_args)

    logger.info(f"Tensor parallelism = {tensor_parallel_size}")
    
    # Configure placement group resources
    pg_resources = [{"CPU": 1}]  # for the deployment replica
    for _ in range(tensor_parallel_size):
        pg_resources.append({"CPU": 4, "GPU": 1})  # for the vLLM actors

    return VLLMDeployment.options(
        placement_group_bundles=pg_resources, 
        placement_group_strategy="STRICT_PACK"
    ).bind(
        engine_args,
        response_role="assistant",
        # lora_modules=None,
    )

# Initialize the deployment
model_name = os.getenv("HF_MODEL_NAME")
tensor_parallel_size = int(os.getenv("TENSOR_PARALLEL_SIZE"))
deployment = build_app(
    model_name=model_name,
    tensor_parallel_size=tensor_parallel_size,
)