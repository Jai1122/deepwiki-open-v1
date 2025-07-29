import logging
import json
import asyncio
from typing import List, Optional, AsyncGenerator
from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from pydantic import BaseModel, Field

from adalflow.core.types import ModelType, Document
from adalflow.components.data_process import TextSplitter

from .config import get_model_config, configs, get_context_window_for_model
from .rag import RAG
from .utils import count_tokens, truncate_prompt_to_fit, get_file_content
from .wiki_prompts import WIKI_PAGE_GENERATION_PROMPT, ARCHITECTURE_OVERVIEW_PROMPT

logger = logging.getLogger(__name__)

def is_websocket_connected(websocket: WebSocket) -> bool:
    """Safely check if WebSocket is still connected"""
    try:
        # Check application state first
        if hasattr(websocket, 'application_state'):
            if websocket.application_state == WebSocketState.DISCONNECTED:
                return False
                
        # Check client state as fallback
        if hasattr(websocket, 'client_state'):
            if websocket.client_state == WebSocketState.DISCONNECTED:
                return False
        
        # Additional check for connection state
        if hasattr(websocket, '_state'):
            if websocket._state == WebSocketState.DISCONNECTED:
                return False
                
        # If we can't determine state, assume connected (let operations fail gracefully)
        return True
    except Exception as e:
        # Log the exception for debugging but don't fail
        logger.debug(f"Exception checking WebSocket state: {e}")
        # If any error checking state, assume disconnected to be safe
        return False

def is_xml_complete(xml_text: str) -> bool:
    """Check if XML response appears to be complete - very conservative validation"""
    if not xml_text.strip():
        return False
    
    text = xml_text.strip()
    
    # Only check for very obvious incomplete patterns to avoid false positives
    
    # Check for specific XML structures that must be complete
    if '<wiki_structure>' in text and not '</wiki_structure>' in text:
        return False
    
    # Check for obvious truncation patterns
    if text.endswith('<'):  # Ends with incomplete opening tag
        return False
    
    # Check for severely unbalanced brackets (more than 10% difference)
    open_count = text.count('<')
    close_count = text.count('>')
    if open_count > 0 and close_count > 0:
        imbalance_ratio = abs(open_count - close_count) / max(open_count, close_count)
        if imbalance_ratio > 0.1:  # More than 10% imbalance suggests truncation
            return False
    
    # Check for response that ends abruptly mid-word in XML context
    if text.startswith('<') and len(text) > 100:
        # If it's XML-like content that ends without proper closure, flag it
        last_50_chars = text[-50:]
        if '<' in last_50_chars and '>' not in last_50_chars:
            return False
    
    # Otherwise assume it's complete - be very conservative to avoid false positives
    return True

async def handle_streaming_response(response_stream):
    """Handle streaming response from LLM with timeout protection and response validation"""
    logger.info("Starting to iterate over response stream.")
    chunk_count = 0
    start_time = asyncio.get_event_loop().time()
    stream_timeout = 300  # Increased to 5 minutes for complex responses
    last_chunk_time = start_time
    response_buffer = ""  # Buffer to accumulate response for validation
    
    try:
        # Use asyncio.wait_for with the entire async for loop
        async def process_stream():
            nonlocal chunk_count, last_chunk_time, response_buffer
            async for chunk in response_stream:
                chunk_count += 1
                current_time = asyncio.get_event_loop().time()
                
                # Check if no chunks received recently (stalled stream)
                if current_time - last_chunk_time > 120:  # 2 minutes without chunks
                    logger.warning(f"Stream stalled - no chunks for {current_time - last_chunk_time:.1f}s")
                    yield json.dumps({"error": "Response stream stalled"})
                    return
                
                # Check if total stream time has exceeded limit
                if current_time - start_time > stream_timeout:
                    logger.warning(f"Stream timeout exceeded ({stream_timeout}s), terminating")
                    yield json.dumps({"error": "Response stream timed out"})
                    return
                
                content = ""
                if isinstance(chunk, str): 
                    content = chunk
                elif hasattr(chunk, "choices") and chunk.choices and hasattr(chunk.choices[0], "delta") and hasattr(chunk.choices[0].delta, "content"): 
                    content = chunk.choices[0].delta.content or ""
                elif hasattr(chunk, "text"): 
                    content = chunk.text
                
                if content:
                    response_buffer += content
                    logger.debug(f"Yielding chunk {chunk_count}: {len(content)} chars, buffer: {len(response_buffer)} chars")
                    yield json.dumps({"content": content})
                
                # Update last chunk time on successful processing
                last_chunk_time = current_time
        
        # Process the stream with overall timeout
        try:
            stream_gen = process_stream()
            while True:
                try:
                    chunk_result = await asyncio.wait_for(stream_gen.__anext__(), timeout=90)  # 90 second chunk timeout for complex responses
                    yield chunk_result
                except StopAsyncIteration:
                    break
        except asyncio.TimeoutError:
            logger.error(f"Chunk timeout exceeded - stream terminated prematurely")
            yield json.dumps({"error": "Stream processing timed out"})
            return  # Don't validate incomplete buffer
            
        # Validate response completeness after streaming is done
        if response_buffer:
            is_complete = is_xml_complete(response_buffer)
            if not is_complete:
                logger.warning(f"Potentially incomplete XML response detected (length: {len(response_buffer)})")
                logger.debug(f"Response buffer starts with: {response_buffer[:200]}...")
                logger.debug(f"Response buffer ends with: ...{response_buffer[-200:]}")
                # For incomplete responses, yield an error to prevent client processing
                yield json.dumps({"error": "Incomplete response detected - please retry"})
                return
            else:
                logger.debug(f"Response completeness validation passed (length: {len(response_buffer)})")
            
        logger.info(f"Stream completed after {chunk_count} chunks, total response: {len(response_buffer)} chars")
    
    except Exception as stream_error:
        logger.error(f"Error in stream iteration: {stream_error}")
        yield json.dumps({"error": f"Stream processing error: {stream_error}"})
    
    logger.info(f"Finished iterating over response stream after {chunk_count} chunks.")

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
    model: Optional[str] = "gemini-2.0-flash"
    language: Optional[str] = "en"
    excluded_dirs: Optional[str] = None
    excluded_files: Optional[str] = None
    included_dirs: Optional[str] = None
    included_files: Optional[str] = None

async def handle_websocket_chat(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection accepted")
    
    try:
        while True:
            # Check if WebSocket is still connected before trying to receive
            if not is_websocket_connected(websocket):
                logger.info("WebSocket client disconnected, breaking loop")
                break
                
            # Add timeout to websocket receive to prevent indefinite waiting
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=300)  # 5 minute timeout
            except asyncio.TimeoutError:
                logger.warning("WebSocket receive timeout - checking connection and sending heartbeat")
                # Check if still connected before sending heartbeat
                if not is_websocket_connected(websocket):
                    logger.info("WebSocket disconnected during timeout, breaking loop")
                    break
                    
                try:
                    await websocket.send_text(json.dumps({"status": "heartbeat", "message": "Connection alive"}))
                except WebSocketDisconnect as disconnect_error:
                    logger.info(f"WebSocket client disconnected during heartbeat: {disconnect_error}")
                    break
                except Exception as heartbeat_error:
                    # Check if this is a connection close event
                    error_str = str(heartbeat_error)
                    if ("CloseCode" in error_str or "NO_STATUS_RCVD" in error_str or 
                        "connection closed" in error_str.lower() or 
                        "1005" in error_str or "1006" in error_str or 
                        "abnormal closure" in error_str.lower()):
                        logger.info(f"WebSocket connection closed during heartbeat: {heartbeat_error}")
                    else:
                        logger.warning(f"Failed to send heartbeat: {heartbeat_error}")
                    break
                continue
            except WebSocketDisconnect as disconnect_error:
                logger.info(f"WebSocket client disconnected: {disconnect_error}")
                break
            except Exception as receive_error:
                # Check if this is a connection close event
                error_str = str(receive_error)
                # Handle various WebSocket close scenarios including 1005, 1006 (abnormal closures)
                if ("CloseCode" in error_str or "NO_STATUS_RCVD" in error_str or 
                    "connection closed" in error_str.lower() or 
                    "1005" in error_str or "1006" in error_str or 
                    "abnormal closure" in error_str.lower()):
                    logger.info(f"WebSocket connection closed by client: {receive_error}")
                    break
                else:
                    logger.error(f"Error receiving WebSocket data: {receive_error}")
                    break
                
            logger.info("Received WebSocket request")
            request = ChatCompletionRequest(**json.loads(data))
            
            # Validate and set default provider if empty
            provider = request.provider.strip() if request.provider else "google"
            if not provider:
                provider = "google"
                logger.warning(f"Empty provider received, defaulting to: {provider}")
            
            # Validate and set default model if empty
            model = request.model.strip() if request.model else None
            if not model:
                # Set default model based on provider
                if provider == "google":
                    model = "gemini-2.0-flash"
                elif provider == "openai":
                    model = "gpt-4"
                else:
                    model = "gemini-2.0-flash"  # fallback
                logger.warning(f"Empty model received, defaulting to: {model}")
            
            logger.info(f"Using provider: {provider}, model: {model}")
            
            try:
                rag_instance = RAG(provider=provider, model=model)
                model_config = get_model_config(provider=provider, model=model)
            except Exception as config_error:
                logger.error(f"Configuration error for provider '{provider}', model '{model}': {config_error}")
                # Try with default fallback
                provider = "google"
                model = "gemini-2.0-flash"
                logger.info(f"Falling back to default provider: {provider}, model: {model}")
                rag_instance = RAG(provider=provider, model=model)
                model_config = get_model_config(provider=provider, model=model)

            # Create updated request with validated provider and model
            validated_request = ChatCompletionRequest(
                repo_url=request.repo_url,
                messages=request.messages,
                filePath=request.filePath,
                token=request.token,
                type=request.type,
                provider=provider,
                model=model,
                language=request.language,
                excluded_dirs=request.excluded_dirs,
                excluded_files=request.excluded_files,
                included_dirs=request.included_dirs,
                included_files=request.included_files
            )

            chunk_count = 0
            start_time = asyncio.get_event_loop().time()
            
            try:
                async for chunk in stream_response(validated_request, rag_instance, model_config):
                    chunk_count += 1
                    
                    # Check if websocket is still connected before sending
                    if not is_websocket_connected(websocket):
                        logger.warning("Client disconnected during stream, stopping generation")
                        break
                    
                    try:
                        await websocket.send_text(chunk)
                    except WebSocketDisconnect as disconnect_error:
                        logger.info(f"WebSocket client disconnected during send: {disconnect_error}")
                        break
                    except Exception as send_error:
                        # Check if this is a connection close event
                        error_str = str(send_error)
                        if ("CloseCode" in error_str or "NO_STATUS_RCVD" in error_str or 
                            "connection closed" in error_str.lower() or 
                            "1005" in error_str or "1006" in error_str or 
                            "abnormal closure" in error_str.lower()):
                            logger.info(f"WebSocket connection closed during send: {send_error}")
                        else:
                            logger.warning(f"Failed to send chunk {chunk_count}: {send_error}")
                        # If we can't send, client probably disconnected
                        break
                    
                    # Send periodic heartbeats and connection checks during long streams
                    if chunk_count % 25 == 0:  # More frequent checks
                        current_time = asyncio.get_event_loop().time()
                        elapsed = current_time - start_time
                        logger.debug(f"Stream progress: {chunk_count} chunks, {elapsed:.1f}s elapsed")
                        
                        # Additional connection health check for long streams
                        if elapsed > 60 and not is_websocket_connected(websocket):
                            logger.warning("WebSocket connection lost during long stream, terminating")
                            break
                        
            except Exception as stream_error:
                logger.error(f"Error in stream processing: {stream_error}")
                try:
                    # Check if websocket is still connected before sending error
                    if is_websocket_connected(websocket):
                        await websocket.send_text(json.dumps({"error": f"Stream processing failed: {stream_error}"}))
                    else:
                        logger.warning("WebSocket disconnected, cannot send error message")
                except Exception as error_send_error:
                    logger.warning(f"Failed to send error message: {error_send_error}")
            
            # Send completion signal if connection is still active
            try:
                if is_websocket_connected(websocket):
                    await websocket.send_text(json.dumps({"status": "done", "message": f"Response completed with {chunk_count} chunks"}))
                    logger.info(f"Sent completion signal for {chunk_count} chunks")
            except Exception as completion_error:
                logger.debug(f"Failed to send completion signal: {completion_error}")
            
            logger.info(f"Completed WebSocket request: {chunk_count} chunks sent")
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected cleanly.")
    except Exception as e:
        # Check if this is a connection close event
        error_str = str(e)
        if ("CloseCode" in error_str or "NO_STATUS_RCVD" in error_str or 
            "connection closed" in error_str.lower() or 
            "1005" in error_str or "1006" in error_str or 
            "abnormal closure" in error_str.lower()):
            logger.info(f"WebSocket connection closed: {e}")
        else:
            logger.error(f"Error in WebSocket handler: {e}", exc_info=True)
            # Try to send error message if still connected
            try:
                if is_websocket_connected(websocket):
                    await websocket.send_text(json.dumps({"error": str(e)}))
            except Exception:
                pass  # Ignore errors when trying to send error message
        
        # Try to close the connection gracefully
        try:
            if is_websocket_connected(websocket):
                await websocket.close()
        except Exception:
            pass  # Ignore errors when trying to close

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

    # Check early if this is a structure query to skip unnecessary RAG preparation
    is_structure_query = '<file_tree>' in query and '</file_tree>' in query
    
    if is_structure_query:
        logger.info("🏗️  Detected wiki structure determination query - skipping RAG retrieval")
        # For structure queries, we don't need RAG content, just pass through the query
        prompt = query
        logger.info(f"📋 Structure query prompt length: {len(prompt)} characters")
        
        client = model_config["model_client"](**model_config.get("initialize_kwargs", {}))
        model_kwargs["stream"] = True
        
        logger.info("Preparing to stream response from LLM (structure query).")
        try:
            api_kwargs = client.convert_inputs_to_api_kwargs(input=prompt, model_kwargs=model_kwargs, model_type=ModelType.LLM)
            logger.debug(f"API kwargs prepared: {list(api_kwargs.keys())}")
            
            # Add timeout to the initial LLM call to prevent hanging on connection
            try:
                response_stream = await asyncio.wait_for(
                    client.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM),
                    timeout=60  # 60 second timeout for initial connection
                )
                logger.info("Successfully connected to LLM API and received response stream (structure query)")
            except asyncio.TimeoutError:
                logger.error("LLM API connection timeout - no response within 60 seconds (structure query)")
                yield json.dumps({"error": "LLM API connection timeout"})
                return
                
            # Handle streaming response (reuse existing streaming logic)
            async for chunk in handle_streaming_response(response_stream):
                yield chunk
                
        except Exception as e:
            logger.error(f"Error in streaming response (structure query): {e}", exc_info=True)
            yield json.dumps({"error": f"Error generating response: {e}"})

        logger.info("Sending 'done' status (structure query).")
        yield json.dumps({"status": "done"})
        return

    # For content generation queries, continue with RAG preparation
    try:
        rag_instance.prepare_retriever(
            request.repo_url, request.type, request.token,
            request.excluded_dirs.split(',') if request.excluded_dirs else None,
            request.excluded_files.split(',') if request.excluded_files else None,
            request.included_dirs.split(',') if request.included_dirs else None,
            request.included_files.split(',') if request.included_files else None,
        )
        
        # Diagnostic check for retriever state
        if not hasattr(rag_instance, 'retriever') or rag_instance.retriever is None:
            logger.error("❌ Retriever not initialized after prepare_retriever call")
            yield json.dumps({"error": "Repository retriever failed to initialize"})
            return
            
        if not hasattr(rag_instance, 'transformed_docs') or not rag_instance.transformed_docs:
            logger.error("❌ No transformed documents available for retrieval")
            yield json.dumps({"error": "No repository documents available for wiki generation"})
            return
            
        logger.info(f"✅ Retriever ready with {len(rag_instance.transformed_docs)} documents")
        
    except Exception as e:
        logger.error(f"Error preparing retriever: {e}", exc_info=True)
        yield json.dumps({"error": f"Failed to prepare repository data: {e}"})
        return

    # For wiki generation, we need comprehensive repository content, not just query-specific matches
    # Use multiple broad queries to get comprehensive codebase coverage
    comprehensive_queries = [
        query,  # Original user query
        "main application code functions classes implementation",  # Core functionality
        "API endpoints routes handlers controllers", # API/web layer
        "database models data structures schemas", # Data layer  
        "configuration settings environment setup", # Configuration
        "utility functions helpers common code", # Utilities
    ]
    
    all_retrieved_docs = []
    seen_sources = set()  # Avoid duplicate content from same source
    context_text = ""  # Initialize to prevent UnboundLocalError
    
    for broad_query in comprehensive_queries:
        try:
            result = rag_instance.call(broad_query, language=request.language)
            logger.debug(f"RAG call result type: {type(result)}, content: {result}")
            
            # Handle different return formats from RAG
            if result is None:
                logger.warning(f"RAG returned None for query: '{broad_query}'")
                continue
                
            if isinstance(result, tuple) and len(result) >= 2:
                docs, _ = result
            else:
                logger.warning(f"Unexpected RAG result format: {type(result)} for query: '{broad_query}'")
                continue
            
            # Ensure docs is a list
            if docs is None:
                logger.warning(f"RAG returned None docs for query: '{broad_query}'")
                continue
                
            if not isinstance(docs, list):
                logger.warning(f"RAG docs is not a list: {type(docs)} for query: '{broad_query}'")
                continue
            
            logger.info(f"✅ Retrieved {len(docs)} docs for query: '{broad_query}'")
            
            for doc in docs:
                if doc is None:
                    continue
                    
                source = doc.metadata.get('source', 'unknown') if hasattr(doc, 'metadata') and doc.metadata else 'unknown'
                if source not in seen_sources:
                    all_retrieved_docs.append(doc)
                    seen_sources.add(source)
                    logger.debug(f"Added doc from source: {source}")
                    
        except Exception as e:
            logger.warning(f"Failed to retrieve docs for query '{broad_query}': {e}")
            logger.debug(f"Exception details: {type(e).__name__}: {str(e)}")
            continue
    
    # If we still don't have enough content, try to get more documents with a very broad query
    if len(all_retrieved_docs) < 10:
        try:
            # Use a very broad query to capture more of the codebase
            result = rag_instance.call("source code implementation functions classes", language=request.language)
            
            if result is not None and isinstance(result, tuple) and len(result) >= 2:
                broad_docs, _ = result
                if broad_docs and isinstance(broad_docs, list):
                    logger.info(f"🔄 Fallback retrieved {len(broad_docs)} additional docs")
                    for doc in broad_docs[:15]:  # Add up to 15 more docs
                        if doc is None:
                            continue
                        source = doc.metadata.get('source', 'unknown') if hasattr(doc, 'metadata') and doc.metadata else 'unknown'
                        if source not in seen_sources:
                            all_retrieved_docs.append(doc)
                            seen_sources.add(source)
                else:
                    logger.warning("Fallback RAG call returned invalid docs format")
            else:
                logger.warning("Fallback RAG call returned invalid result format")
        except Exception as e:
            logger.warning(f"Failed to retrieve additional broad docs: {e}")
            logger.debug(f"Fallback exception details: {type(e).__name__}: {str(e)}")
    
    # ENHANCED FALLBACK: For wiki generation, we need comprehensive content regardless of similarity scores
    # Start by getting more comprehensive coverage from the repository
    if not is_structure_query:  # Only for content generation, not structure determination
        try:
            # Access the transformed documents directly from the RAG instance for comprehensive coverage
            if hasattr(rag_instance, 'transformed_docs') and rag_instance.transformed_docs:
                direct_docs = rag_instance.transformed_docs
                logger.info(f"📚 Found {len(direct_docs)} total documents in repository database")
                
                # For wiki generation, we want comprehensive coverage, not just similarity matches
                # Add more documents, prioritizing different file types for comprehensive coverage
                doc_count_by_type = {}
                
                # First pass: add documents we don't already have, up to reasonable limits per file type
                for doc in direct_docs:
                    if len(all_retrieved_docs) >= 50:  # Reasonable upper limit
                        break
                        
                    source = doc.metadata.get('source', 'unknown') if hasattr(doc, 'metadata') else 'unknown'
                    if source not in seen_sources:
                        # Determine file type for balanced coverage
                        file_ext = source.split('.')[-1].lower() if '.' in source else 'unknown'
                        
                        # Limit per file type to ensure diversity
                        type_limit = 8 if file_ext in ['py', 'js', 'ts', 'go', 'java', 'cpp', 'c'] else 3
                        current_count = doc_count_by_type.get(file_ext, 0)
                        
                        if current_count < type_limit:
                            all_retrieved_docs.append(doc)
                            seen_sources.add(source)
                            doc_count_by_type[file_ext] = current_count + 1
                            logger.debug(f"Added doc from source: {source} (type: {file_ext})")
                        
                logger.info(f"📈 Enhanced wiki context: Now have {len(all_retrieved_docs)} documents with balanced coverage")
                logger.info(f"📊 File type distribution: {doc_count_by_type}")
        except Exception as e:
            logger.warning(f"Direct document access failed: {e}")
    
    # Final fallback check
    if len(all_retrieved_docs) < 3:
        logger.warning(f"⚠️  Still insufficient content after all attempts (only {len(all_retrieved_docs)} docs)")
        logger.info("🔄 Making final attempt with very broad retrieval...")
        
        try:
            # Last resort: get any available documents
            if hasattr(rag_instance, 'transformed_docs') and rag_instance.transformed_docs:
                for doc in rag_instance.transformed_docs[:20]:  # Just get the first 20 documents
                    source = doc.metadata.get('source', 'unknown') if hasattr(doc, 'metadata') else 'unknown'
                    if source not in seen_sources:
                        all_retrieved_docs.append(doc)
                        seen_sources.add(source)
                        
                logger.info(f"🔄 Final fallback added {len(all_retrieved_docs)} documents")
        except Exception as e:
            logger.error(f"Final fallback failed: {e}")
    
    # Combine all retrieved content
    context_text = "\n\n".join([doc.text for doc in all_retrieved_docs]) if all_retrieved_docs else ""
    
    logger.info(f"📊 Wiki generation final stats: {len(all_retrieved_docs)} documents from {len(seen_sources)} unique sources")
    logger.info(f"📝 Total context length: {len(context_text)} characters")
    
    # Enhanced logging for debugging
    if len(seen_sources) > 0:
        logger.info(f"📁 Source files included: {list(seen_sources)[:10]}{'...' if len(seen_sources) > 10 else ''}")
    else:
        logger.error("❌ No source files retrieved - wiki will be generic!")
    file_content = get_file_content(request.repo_url, request.filePath, request.type, request.token) if request.filePath else ""
    
    if is_structure_query:
        # For structure determination, use the query as-is (it already contains file tree + README)
        logger.info("🏗️  Processing wiki structure determination query")
        system_prompt = query  # The query already contains the proper structure prompt
    else:
        # For content generation, use intelligently chosen prompts based on query type
        query_lower = query.lower()
        
        if any(keyword in query_lower for keyword in ['architecture', 'overview', 'system', 'structure']):
            # Use architecture overview prompt for system-level queries
            logger.info("🏛️  Using architecture overview prompt for comprehensive analysis")
            system_prompt = ARCHITECTURE_OVERVIEW_PROMPT.format(
                file_tree=context_text[:5000] if context_text else "File tree not available",
                readme=file_content[:2000] if file_content else "README not available", 
                context=context_text[:10000] if context_text else "Repository context not available"
            )
        else:
            # Use detailed page generation prompt for component-specific queries
            system_prompt = WIKI_PAGE_GENERATION_PROMPT.format(
                context=context_text[:5000] if context_text else "Context not available",
                file_content=file_content[:10000] if file_content else "File content not available",
                page_topic=query
            )
    conversation_history = "\n".join([f"{m.role}: {m.content}" for m in request.messages[:-1]])

    if is_structure_query:
        # For structure queries, the system_prompt already contains everything we need
        prompt = system_prompt
        logger.info(f"📋 Structure query prompt length: {len(prompt)} characters")
    else:
        # For content generation, combine all context
        file_content, context_text = truncate_prompt_to_fit(
            context_window, max_completion_tokens, system_prompt, conversation_history, file_content, context_text, query
        )
        prompt = f"{system_prompt}\n\nHistory: {conversation_history}\n\nFile Context:\n{file_content}\n\nRetrieved Context:\n{context_text}\n\nQuery: {query}"
        logger.info(f"📝 Content generation prompt length: {len(prompt)} characters")
        logger.info(f"🔍 Context breakdown - File: {len(file_content)}, Retrieved: {len(context_text)}, History: {len(conversation_history)}")
    
    client = model_config["model_client"](**model_config.get("initialize_kwargs", {}))
    model_kwargs["stream"] = True

    logger.info("Preparing to stream response from LLM.")
    try:
        api_kwargs = client.convert_inputs_to_api_kwargs(input=prompt, model_kwargs=model_kwargs, model_type=ModelType.LLM)
        logger.debug(f"API kwargs prepared: {list(api_kwargs.keys())}")
        
        # Add timeout to the initial LLM call to prevent hanging on connection
        try:
            response_stream = await asyncio.wait_for(
                client.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM),
                timeout=60  # 60 second timeout for initial connection
            )
            logger.info("Successfully connected to LLM API and received response stream")
        except asyncio.TimeoutError:
            logger.error("LLM API connection timeout - no response within 60 seconds")
            yield json.dumps({"error": "LLM API connection timeout"})
            return
        
        # Use the shared streaming handler
        async for chunk in handle_streaming_response(response_stream):
            yield chunk

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
