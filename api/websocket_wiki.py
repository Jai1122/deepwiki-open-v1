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
        rag_instance.prepare_retriever(
            request.repo_url, request.type, request.token,
            [d for d in request.excluded_dirs.split(',') if d] if request.excluded_dirs else None,
            [f for f in request.excluded_files.split(',') if f] if request.excluded_files else None,
            [d for d in request.included_dirs.split(',') if d] if request.included_dirs else None,
            [f for f in request.included_files.split(',') if f] if request.included_files else None,
        )
    except Exception as e:
        logger.error(f"Error preparing retriever: {e}", exc_info=True)
        yield json.dumps({"error": f"Failed to prepare repository data: {e}"})
        return

    query = request.messages[-1].content if request.messages else ""
    retrieved_docs, _ = rag_instance.call(query, language)
    context_text = "\n\n".join([doc.content for doc in retrieved_docs]) if retrieved_docs else ""

    file_content = get_file_content(request.repo_url, request.filePath, request.type, request.token) if request.filePath else ""

    system_prompt = "You are a helpful assistant..."
    conversation_history = "\n".join([f"{m.role}: {m.content}" for m in request.messages[:-1]])

    model_kwargs = model_config.get("model_kwargs", {})
    context_window = get_context_window_for_model(request.provider, request.model)
    max_completion_tokens = model_kwargs.get("max_tokens", 4096)

    logger.info(f"Context Window: {context_window}, Max Completion Tokens: {max_completion_tokens}")
    logger.info(f"Tokens - System: {count_tokens(system_prompt)}, History: {count_tokens(conversation_history)}, Query: {count_tokens(query)}")
    logger.info(f"Tokens before truncation - File: {count_tokens(file_content)}, RAG: {count_tokens(context_text)}")

    file_content, context_text = truncate_prompt_to_fit(
        context_window=context_window,
        max_completion_tokens=max_completion_tokens,
        system_prompt=system_prompt,
        conversation_history=conversation_history,
        file_content=file_content,
        context_text=context_text,
        query=query
    )

    logger.info(f"Tokens after truncation - File: {count_tokens(file_content)}, RAG: {count_tokens(context_text)}")

    prompt = f"{system_prompt}\n\nHistory: {conversation_history}\n\nFile Context:\n{file_content}\n\nRetrieved Context:\n{context_text}\n\nQuery: {query}"
    
    # Final defensive check
    prompt_tokens = count_tokens(prompt)
    if prompt_tokens + max_completion_tokens > context_window:
        excess_tokens = (prompt_tokens + max_completion_tokens) - context_window
        logger.warning(f"Prompt still exceeds context window by {excess_tokens} tokens after initial truncation. Performing hard truncation.")
        # Hard truncate the prompt itself
        prompt = prompt[:int(len(prompt) * ((context_window - max_completion_tokens) / prompt_tokens))]

    logger.info(f"Final prompt tokens: {count_tokens(prompt)}. Final model_kwargs: {model_kwargs}")

    client = model_config["model_client"](**model_config.get("initialize_kwargs", {}))
    
    try:
        api_kwargs = client.convert_inputs_to_api_kwargs(input=prompt, model_kwargs=model_kwargs, model_type=ModelType.LLM)
        response_stream = await client.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
        
        async for chunk in response_stream:
            content = ""
            if isinstance(chunk, str): content = chunk
            elif hasattr(chunk, "choices") and chunk.choices and hasattr(chunk.choices[0], "delta") and hasattr(chunk.choices[0].delta, "content"): content = chunk.choices[0].delta.content or ""
            elif hasattr(chunk, "text"): content = chunk.text
            if content: yield json.dumps({"content": content})

    except Exception as e:
        logger.error(f"Error in streaming response: {e}", exc_info=True)
        yield json.dumps({"error": f"Error generating response: {e}"})

    yield json.dumps({"status": "done"})