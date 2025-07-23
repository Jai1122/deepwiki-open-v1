import logging
from typing import List, Optional, AsyncGenerator
from urllib.parse import unquote
import json
import asyncio

import google.generativeai as genai
from adalflow.components.model_client.ollama_client import OllamaClient
from adalflow.core.types import ModelType, Document
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from .config import get_model_config, configs, get_context_window_for_model
from .openai_client import OpenAIClient
from .openrouter_client import OpenRouterClient
from .azureai_client import AzureAIClient
from .rag import RAG
from .utils import count_tokens, truncate_prompt_to_fit, get_file_content

logger = logging.getLogger(__name__)

# Pydantic Models for API
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    repo_url: str
    messages: List[ChatMessage]
    filePath: Optional[str] = None
    token: Optional[str] = None
    type: Optional[str] = "github"
    provider: str = "google"
    model: Optional[str] = None
    language: Optional[str] = "en"
    excluded_dirs: Optional[str] = None
    excluded_files: Optional[str] = None
    included_dirs: Optional[str] = None
    included_files: Optional[str] = None

async def handle_websocket_chat(websocket: WebSocket):
    """
    Handles the WebSocket connection for the chat, processing requests and streaming responses.
    """
    await websocket.accept()
    try:
        while True:
            # Receive and parse the request from the client
            data = await websocket.receive_text()
            request_data = json.loads(data)
            request = ChatCompletionRequest(**request_data)

            # Initialize RAG and model configurations
            rag_instance = RAG(provider=request.provider, model=request.model)
            model_config = get_model_config(provider=request.provider, model=request.model)

            # Asynchronously stream the response back to the client
            async for chunk in stream_response(request, websocket, rag_instance, model_config, request.language):
                await websocket.send_text(chunk)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected.")
    except Exception as e:
        logger.error(f"Error in WebSocket handler: {e}", exc_info=True)
        # Send an error message to the client before closing
        try:
            await websocket.send_text(json.dumps({"error": str(e)}))
        except Exception:
            pass # Ignore errors if the socket is already closed
        await websocket.close()

async def stream_response(
    request: ChatCompletionRequest,
    websocket: WebSocket,
    rag_instance: RAG,
    model_config: dict,
    language: str = "en"
) -> AsyncGenerator[str, None]:
    """
    Generates and streams the chat response, handling context retrieval and model interaction.
    """
    try:
        # Prepare the retriever
        rag_instance.prepare_retriever(
            request.repo_url,
            request.type,
            request.token,
            request.excluded_dirs.split(',') if request.excluded_dirs else None,
            request.excluded_files.split(',') if request.excluded_files else None,
            request.included_dirs.split(',') if request.included_dirs else None,
            request.included_files.split(',') if request.included_files else None,
        )
    except Exception as e:
        logger.error(f"Error preparing retriever: {e}", exc_info=True)
        yield json.dumps({"error": f"Failed to prepare repository data: {e}"})
        return

    # Extract the latest user query
    user_query = request.messages[-1].content if request.messages else ""
    
    # Perform RAG call to get context
    retrieved_docs, _ = rag_instance.call(user_query, language)
    context_text = "\n\n".join([doc.content for doc in retrieved_docs]) if retrieved_docs else ""

    # Get file content if a file path is provided
    file_content = ""
    if request.filePath:
        file_content = get_file_content(request.repo_url, request.filePath, request.type, request.token)

    # Prepare prompts and conversation history
    system_prompt = "You are a helpful assistant. Provide a detailed answer based on the context."
    conversation_history = "\n".join([f"{msg.role}: {msg.content}" for msg in request.messages[:-1]])
    
    # Get model parameters from config
    model_kwargs = model_config.get("model_kwargs", {})
    context_window = get_context_window_for_model(request.provider, request.model)
    max_completion_tokens = model_kwargs.get("max_tokens", 2048) # Use max_tokens from kwargs, which is our desired completion length

    # Truncate prompts to fit the model's context window
    file_content, context_text = truncate_prompt_to_fit(
        context_window=context_window,
        max_completion_tokens=max_completion_tokens,
        system_prompt=system_prompt,
        conversation_history=conversation_history,
        file_content=file_content,
        context_text=context_text,
        query=user_query
    )

    # Construct the final prompt
    final_prompt = f"{system_prompt}\n\nHistory:\n{conversation_history}\n\nFile Context:\n{file_content}\n\nRetrieved Context:\n{context_text}\n\nQuery: {user_query}"

    # Initialize the model client
    client = model_config["model_client"](**model_config.get("initialize_kwargs", {}))
    
    try:
        # Prepare the API call arguments
        api_kwargs = client.convert_inputs_to_api_kwargs(
            input=final_prompt,
            model_kwargs=model_kwargs,
            model_type=ModelType.LLM
        )
        
        # Generate and stream the response
        response_stream = await client.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
        
        is_streaming = model_kwargs.get("stream", False)
        
        if is_streaming:
            async for chunk in response_stream:
                # Handle different chunk formats from various clients
                content = ""
                if isinstance(chunk, str):
                    content = chunk
                elif hasattr(chunk, "choices") and chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta and hasattr(delta, "content"):
                        content = delta.content or ""
                elif hasattr(chunk, "text"):
                    content = chunk.text
                
                if content:
                    yield json.dumps({"content": content})
        else:
            # Handle non-streaming responses
            yield json.dumps({"content": str(response_stream)})

    except Exception as e:
        logger.error(f"Error in streaming response: {e}", exc_info=True)
        yield json.dumps({"error": f"Error generating response: {e}"})

    # Send a final message to indicate the end of the stream
    yield json.dumps({"status": "done"})