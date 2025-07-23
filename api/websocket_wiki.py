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
from adalflow.components.data_process import TextSplitter

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
    This version can now handle oversized queries by summarizing them first.
    """
    query = request.messages[-1].content if request.messages else ""
    
    model_kwargs = model_config.get("model_kwargs", {})
    context_window = get_context_window_for_model(request.provider, request.model)
    max_completion_tokens = model_kwargs.get("max_tokens", 4096)
    allowed_prompt_size = context_window - max_completion_tokens

    query_tokens = count_tokens(query)

    # If the query itself is too large, summarize it first.
    if query_tokens > allowed_prompt_size * 0.8:
        logger.warning(f"Query is too large ({query_tokens} tokens). Summarizing before proceeding.")
        yield json.dumps({"status": "summarizing", "message": "Query is very large, summarizing its content to proceed..."})
        
        try:
            # 1. Chunk the oversized query
            splitter_config = configs.get("text_splitter", {})
            text_splitter = TextSplitter(
                split_by=splitter_config.get("split_by", "word"),
                chunk_size=splitter_config.get("chunk_size", 1000),
                chunk_overlap=splitter_config.get("chunk_overlap", 200),
            )
            query_doc = Document(text=query)
            chunks = text_splitter([query_doc])
            
            # 2. Summarize each chunk
            summaries = []
            client = model_config["model_client"](**model_config.get("initialize_kwargs", {}))
            
            for i, chunk in enumerate(chunks):
                yield json.dumps({"status": "summarizing", "message": f"Summarizing chunk {i+1} of {len(chunks)}..."})
                summary_prompt = f"Summarize the following text concisely: {chunk.text}"
                
                # Use a non-streaming call for summarization
                summary_kwargs = model_kwargs.copy()
                summary_kwargs["stream"] = False
                
                api_kwargs = client.convert_inputs_to_api_kwargs(input=summary_prompt, model_kwargs=summary_kwargs, model_type=ModelType.LLM)
                summary_response = await client.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
                
                # Extract text from potentially complex response objects
                summary_text = ""
                if isinstance(summary_response, str):
                    summary_text = summary_response
                elif hasattr(summary_response, "text"):
                    summary_text = summary_response.text
                elif hasattr(summary_response, "choices") and summary_response.choices:
                    summary_text = summary_response.choices[0].message.content
                
                summaries.append(summary_text)

            # 3. Combine summaries and create a final summary
            combined_summary = " ".join(summaries)
            final_summary_prompt = f"Create a final, coherent summary from the following smaller summaries: {combined_summary}"
            
            api_kwargs = client.convert_inputs_to_api_kwargs(input=final_summary_prompt, model_kwargs=summary_kwargs, model_type=ModelType.LLM)
            final_summary_response = await client.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
            
            final_summary = ""
            if isinstance(final_summary_response, str):
                final_summary = final_summary_response
            elif hasattr(final_summary_response, "text"):
                final_summary = final_summary_response.text
            elif hasattr(final_summary_response, "choices") and final_summary_response.choices:
                final_summary = final_summary_response.choices[0].message.content

            query = final_summary # Replace original query with the summary
            logger.info(f"Original query summarized to {count_tokens(query)} tokens.")
            yield json.dumps({"status": "summarized", "message": "Summary complete. Proceeding with context retrieval."})

        except Exception as e:
            logger.error(f"Failed to summarize oversized query: {e}", exc_info=True)
            yield json.dumps({"error": "Failed to process the large query. Please try a shorter one."})
            return

    # Continue with the (potentially summarized) query
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

    retrieved_docs, _ = rag_instance.call(query, language)
    context_text = "\n\n".join([doc.content for doc in retrieved_docs]) if retrieved_docs else ""

    file_content = get_file_content(request.repo_url, request.filePath, request.type, request.token) if request.filePath else ""

    system_prompt = "You are a helpful assistant. Provide a detailed answer based on the context."
    conversation_history = "\n".join([f"{m.role}: {m.content}" for m in request.messages[:-1]])

    # model_kwargs = model_config.get("model_kwargs", {})
    # context_window = get_context_window_for_model(request.provider, request.model)
    # max_completion_tokens = model_kwargs.get("max_tokens", 4096)

    # Always enforce streaming for WebSocket responses
    model_kwargs["stream"] = True

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
    
    prompt_tokens = count_tokens(prompt)
    if prompt_tokens + max_completion_tokens > context_window:
        excess_tokens = (prompt_tokens + max_completion_tokens) - context_window
        logger.warning(f"Prompt still exceeds context window by {excess_tokens} tokens after initial truncation. Performing hard truncation.")
        ratio = (context_window - max_completion_tokens) / prompt_tokens
        prompt = prompt[:int(len(prompt) * ratio)]

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

    except TypeError as te:
        logger.error(f"TypeError during streaming, this may indicate a non-streaming response was received: {te}", exc_info=True)
        yield json.dumps({"error": "A non-streaming response was received from the model. Please check model configuration."})
    except Exception as e:
        logger.error(f"Error in streaming response: {e}", exc_info=True)
        yield json.dumps({"error": f"Error generating response: {e}"})

    yield json.dumps({"status": "done"})