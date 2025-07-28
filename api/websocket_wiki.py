import logging
import json
from typing import List, Optional, AsyncGenerator
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from adalflow.core.types import ModelType, Document
from adalflow.components.data_process import TextSplitter

from .config import get_model_config, configs, get_context_window_for_model
from .rag import RAG
from .utils import count_tokens, truncate_prompt_to_fit, get_file_content
from .wiki_prompts import WIKI_PAGE_GENERATION_PROMPT, ARCHITECTURE_OVERVIEW_PROMPT

logger = logging.getLogger(__name__)

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
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            request = ChatCompletionRequest(**json.loads(data))
            rag_instance = RAG(provider=request.provider, model=request.model)
            model_config = get_model_config(provider=request.provider, model=request.model)

            async for chunk in stream_response(request, rag_instance, model_config):
                await websocket.send_text(chunk)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected.")
    except Exception as e:
        logger.error(f"Error in WebSocket handler: {e}", exc_info=True)
        if not websocket.client_state.DISCONNECTED:
            await websocket.send_text(json.dumps({"error": str(e)}))
        await websocket.close()

async def stream_response(
    request: ChatCompletionRequest,
    rag_instance: RAG,
    model_config: dict
) -> AsyncGenerator[str, None]:
    query = request.messages[-1].content if request.messages else ""
    model_kwargs = model_config.get("model_kwargs", {})
    context_window = get_context_window_for_model(request.provider, request.model)
    max_completion_tokens = model_kwargs.get("max_tokens", 4096)
    allowed_prompt_size = context_window - max_completion_tokens

    if count_tokens(query) > allowed_prompt_size * 0.8:
        yield json.dumps({"status": "summarizing", "message": "Query is very large, summarizing..."})
        try:
            query = await summarize_oversized_query(query, model_config, model_kwargs)
            yield json.dumps({"status": "summarized", "message": "Summary complete. Proceeding..."})
        except Exception as e:
            logger.error(f"Failed to summarize oversized query: {e}", exc_info=True)
            yield json.dumps({"error": "Failed to process the large query."})
            return

    try:
        rag_instance.prepare_retriever(
            request.repo_url, request.type, request.token,
            request.excluded_dirs.split(',') if request.excluded_dirs else None,
            request.excluded_files.split(',') if request.excluded_files else None,
            request.included_dirs.split(',') if request.included_dirs else None,
            request.included_files.split(',') if request.included_files else None,
        )
    except Exception as e:
        logger.error(f"Error preparing retriever: {e}", exc_info=True)
        yield json.dumps({"error": f"Failed to prepare repository data: {e}"})
        return

    retrieved_docs, _ = rag_instance.call(query, language=request.language)
    context_text = "\n\n".join([doc.text for doc in retrieved_docs]) if retrieved_docs else ""
    file_content = get_file_content(request.repo_url, request.filePath, request.type, request.token) if request.filePath else ""
    # Intelligently choose prompt based on query type
    query_lower = query.lower()
    
    if any(keyword in query_lower for keyword in ['architecture', 'overview', 'system', 'structure', 'file tree']):
        # Use architecture overview prompt for system-level queries
        system_prompt = ARCHITECTURE_OVERVIEW_PROMPT.format(
            file_tree=context_text[:5000] if context_text else "File tree not available",
            readme=file_content[:2000] if file_content else "README not available", 
            context=context_text[:3000] if context_text else "Repository context not available"
        )
    else:
        # Use detailed page generation prompt for component-specific queries
        system_prompt = WIKI_PAGE_GENERATION_PROMPT.format(
            context=context_text[:5000] if context_text else "Context not available",
            file_content=file_content[:10000] if file_content else "File content not available",
            page_topic=query
        )
    conversation_history = "\n".join([f"{m.role}: {m.content}" for m in request.messages[:-1]])

    file_content, context_text = truncate_prompt_to_fit(
        context_window, max_completion_tokens, system_prompt, conversation_history, file_content, context_text, query
    )

    prompt = f"{system_prompt}\n\nHistory: {conversation_history}\n\nFile Context:\n{file_content}\n\nRetrieved Context:\n{context_text}\n\nQuery: {query}"
    
    client = model_config["model_client"](**model_config.get("initialize_kwargs", {}))
    model_kwargs["stream"] = True

    logger.info("Preparing to stream response from LLM.")
    try:
        api_kwargs = client.convert_inputs_to_api_kwargs(input=prompt, model_kwargs=model_kwargs, model_type=ModelType.LLM)
        response_stream = await client.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
        
        logger.info("Starting to iterate over response stream.")
        chunk_count = 0
        async for chunk in response_stream:
            chunk_count += 1
            content = ""
            if isinstance(chunk, str): content = chunk
            elif hasattr(chunk, "choices") and chunk.choices and hasattr(chunk.choices[0], "delta") and hasattr(chunk.choices[0].delta, "content"): content = chunk.choices[0].delta.content or ""
            elif hasattr(chunk, "text"): content = chunk.text
            
            if content:
                # logger.debug(f"Yielding chunk {chunk_count}: {content}")
                yield json.dumps({"content": content})
        
        logger.info(f"Finished iterating over response stream after {chunk_count} chunks.")

    except Exception as e:
        logger.error(f"Error in streaming response: {e}", exc_info=True)
        yield json.dumps({"error": f"Error generating response: {e}"})

    logger.info("Sending 'done' status.")
    yield json.dumps({"status": "done"})

async def summarize_oversized_query(query: str, model_config: dict, model_kwargs: dict) -> str:
    """Chunks and summarizes a very large query."""
    logger.info("Summarizing oversized query...")
    splitter_config = configs.get("text_splitter", {})
    text_splitter = TextSplitter(
        split_by=splitter_config.get("split_by", "word"),
        chunk_size=splitter_config.get("chunk_size", 2000),
        chunk_overlap=splitter_config.get("chunk_overlap", 200),
    )
    # Use the splitter to get text chunks directly
    chunks = text_splitter.split_text(query)
    
    summaries = []
    client = model_config["model_client"](**model_config.get("initialize_kwargs", {}))
    summary_kwargs = {**model_kwargs, "stream": False}

    for chunk_text in chunks:
        summary_prompt = f"Summarize the following text concisely: {chunk_text}"
        
        # Defensively check prompt size before sending to summarizer
        if count_tokens(summary_prompt) >= get_context_window_for_model(model_config.get("provider"), model_kwargs.get("model")):
            logger.warning("Skipping a chunk during summarization as it exceeds model context window.")
            continue

        api_kwargs = client.convert_inputs_to_api_kwargs(input=summary_prompt, model_kwargs=summary_kwargs, model_type=ModelType.LLM)
        summary_response = await client.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
        
        summary_text = ""
        if isinstance(summary_response, str): summary_text = summary_response
        elif hasattr(summary_response, "text"): summary_text = summary_response.text
        elif hasattr(summary_response, "choices") and summary_response.choices: summary_text = summary_response.choices[0].message.content
        summaries.append(summary_text)

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
    
    logger.info(f"Original query summarized to {count_tokens(final_summary)} tokens.")
    return final_summary
