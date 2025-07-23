import logging
import os
from typing import List, Optional
from urllib.parse import unquote

import google.generativeai as genai
from adalflow.components.model_client.ollama_client import OllamaClient
from adalflow.core.types import ModelType
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from api.config import get_model_config, configs, OPENROUTER_API_KEY, OPENAI_API_KEY, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
from .config import get_model_config, configs, get_context_window_for_model
from .utils import count_tokens, get_file_content, truncate_prompt_to_fit
from api.openai_client import OpenAIClient
from api.openrouter_client import OpenRouterClient
from api.bedrock_client import BedrockClient
from api.azureai_client import AzureAIClient
from api.rag import RAG

# Configure logging
from api.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


# Initialize FastAPI app
app = FastAPI(
    title="Simple Chat API",
    description="Simplified API for streaming chat completions"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Models for the API
class ChatMessage(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str

class ChatCompletionRequest(BaseModel):
    """
    Model for requesting a chat completion.
    """
    repo_url: str = Field(..., description="URL of the repository to query")
    messages: List[ChatMessage] = Field(..., description="List of chat messages")
    filePath: Optional[str] = Field(None, description="Optional path to a file in the repository to include in the prompt")
    token: Optional[str] = Field(None, description="Personal access token for private repositories")
    type: Optional[str] = Field("github", description="Type of repository (e.g., 'github', 'gitlab', 'bitbucket')")

    # model parameters
    provider: str = Field("google", description="Model provider (google, openai, openrouter, ollama, bedrock, azure)")
    model: Optional[str] = Field(None, description="Model name for the specified provider")

    language: Optional[str] = Field("en", description="Language for content generation (e.g., 'en', 'ja', 'zh', 'es', 'kr', 'vi')")
    excluded_dirs: Optional[str] = Field(None, description="Comma-separated list of directories to exclude from processing")
    excluded_files: Optional[str] = Field(None, description="Comma-separated list of file patterns to exclude from processing")
    included_dirs: Optional[str] = Field(None, description="Comma-separated list of directories to include exclusively")
    included_files: Optional[str] = Field(None, description="Comma-separated list of file patterns to include exclusively")

@app.post("/chat/completions/stream")
async def chat_completions_stream(request: ChatCompletionRequest):
    """Stream a chat completion response."""
    try:
        # Initialize RAG instance
        request_rag = RAG(provider=request.provider, model=request.model)

        # Prepare retriever
        try:
            excluded_dirs = [unquote(d) for d in request.excluded_dirs.split(',') if d] if request.excluded_dirs else None
            excluded_files = [unquote(f) for f in request.excluded_files.split(',') if f] if request.excluded_files else None
            included_dirs = [unquote(d) for d in request.included_dirs.split(',') if d] if request.included_dirs else None
            included_files = [unquote(f) for f in request.included_files.split(',') if f] if request.included_files else None
            
            request_rag.prepare_retriever(
                request.repo_url, request.type, request.token,
                excluded_dirs, excluded_files, included_dirs, included_files
            )
            logger.info(f"Retriever prepared for {request.repo_url}")
        except Exception as e:
            logger.error(f"Error preparing retriever: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error preparing retriever: {e}")

        # Validate messages
        if not request.messages:
            raise HTTPException(status_code=400, detail="No messages provided")
        
        query = request.messages[-1].content

        # Get RAG context
        retrieved_docs, _ = request_rag.call(query, language=request.language)
        context_text = "\n\n".join([doc.content for doc in retrieved_docs]) if retrieved_docs else ""

        # Get file content
        file_content = ""
        if request.filePath:
            file_content = get_file_content(request.repo_url, request.filePath, request.type, request.token)

        # Prepare prompt components
        system_prompt = "You are an expert code analyst. Please provide a direct and concise answer."
        conversation_history = "\n".join([f"{m.role}: {m.content}" for m in request.messages[:-1]])

        # Get model configuration
        model_config = get_model_config(request.provider, request.model)
        model_kwargs = model_config.get("model_kwargs", {})
        context_window = get_context_window_for_model(request.provider, request.model)
        max_completion_tokens = model_kwargs.get("max_tokens", 4096)

        # Truncate context to fit
        file_content, context_text = truncate_prompt_to_fit(
            context_window=context_window,
            max_completion_tokens=max_completion_tokens,
            system_prompt=system_prompt,
            conversation_history=conversation_history,
            file_content=file_content,
            context_text=context_text,
            query=query
        )

        # Construct final prompt
        prompt = f"{system_prompt}\n\n<conversation_history>\n{conversation_history}\n</conversation_history>\n\n<currentFileContent path=\"{request.filePath}\">\n{file_content}\n</currentFileContent>\n\n<retrieved_context>\n{context_text}\n</retrieved_context>\n\n<query>\n{query}\n</query>\n\nAssistant:"

        # Initialize client and stream response
        client = model_config["model_client"](**model_config.get("initialize_kwargs", {}))

        async def response_stream():
            try:
                api_kwargs = client.convert_inputs_to_api_kwargs(
                    input=prompt,
                    model_kwargs=model_kwargs,
                    model_type=ModelType.LLM
                )
                response = await client.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
                
                if model_kwargs.get("stream", False):
                    async for chunk in response:
                        yield chunk
                else:
                    yield str(response)
            except Exception as e:
                logger.error(f"Error during model generation: {e}", exc_info=True)
                yield f"Error: {e}"

        return StreamingResponse(response_stream(), media_type="text/event-stream")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat completion stream: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint to check if the API is running"""
    return {"status": "API is running", "message": "Navigate to /docs for API documentation"}
