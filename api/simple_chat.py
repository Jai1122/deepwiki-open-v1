import logging
import os
from typing import List, Optional
from urllib.parse import unquote

from adalflow.core.types import ModelType
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .config import get_model_config, configs, get_context_window_for_model, get_max_tokens_for_model
from .utils import count_tokens, get_file_content, truncate_prompt_to_fit
from .rag import RAG

# Configure logging
from .logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

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
    provider: str = Field("vllm", description="Model provider (vllm only)")
    model: Optional[str] = Field(None, description="Model name for the specified provider")

    language: Optional[str] = Field("en", description="Language for content generation (English only)")

@app.post("/chat/completions/stream")
async def chat_completions_stream(request: ChatCompletionRequest):
    """Stream a chat completion response."""
    try:
        # Initialize RAG instance
        request_rag = RAG(provider=request.provider, model=request.model)

        # Prepare retriever
        try:
            request_rag.prepare_retriever(
                request.repo_url, request.type, request.token
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
        max_completion_tokens = get_max_tokens_for_model(request.provider, request.model)
        
        # Ensure max_tokens in model_kwargs doesn't exceed the configured limit
        if "max_tokens" in model_kwargs:
            model_kwargs["max_tokens"] = min(model_kwargs["max_tokens"], max_completion_tokens)
        else:
            model_kwargs["max_tokens"] = max_completion_tokens

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
