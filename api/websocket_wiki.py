import logging
import json
import asyncio
from typing import List, Optional, AsyncGenerator
from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from pydantic import BaseModel, Field

from adalflow.core.types import ModelType, Document
from adalflow.components.data_process import TextSplitter

from .config import get_model_config, configs, get_context_window_for_model, get_max_tokens_for_model
from .rag import RAG
from .utils import count_tokens, truncate_prompt_to_fit, get_file_content
from .wiki_prompts import WIKI_PAGE_GENERATION_PROMPT, ARCHITECTURE_OVERVIEW_PROMPT

logger = logging.getLogger(__name__)

async def safe_websocket_send(websocket: WebSocket, message: dict) -> bool:
    """
    Safely send a message via WebSocket with error handling.
    Returns True if successful, False if failed.
    """
    if not is_websocket_connected(websocket):
        logger.warning("Attempted to send message to disconnected WebSocket")
        return False
    
    try:
        await websocket.send_text(json.dumps(message))
        return True
    except WebSocketDisconnect:
        logger.info("Client disconnected during message send")
        return False
    except Exception as e:
        logger.error(f"Failed to send WebSocket message: {e}")
        return False

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
        logger.debug("XML incomplete: empty response")
        return False
    
    text = xml_text.strip()
    
    # Only flag very obvious incomplete patterns to minimize false positives
    
    # Check for specific XML structures that must be complete
    if '<wiki_structure>' in text and not '</wiki_structure>' in text:
        logger.warning("XML incomplete: wiki_structure tag not closed")
        return False
    
    # Check for obvious truncation patterns
    if text.endswith('<'):  # Ends with incomplete opening tag
        logger.warning("XML incomplete: ends with incomplete opening tag")
        return False
    
    # Check for response that ends abruptly with incomplete XML content
    if text.startswith('<') and len(text) > 100:
        last_200_chars = text[-200:]
        # Look for incomplete tag at the very end (more characters to be sure)
        if '<' in last_200_chars:
            last_open_pos = last_200_chars.rfind('<')
            last_close_pos = last_200_chars.rfind('>')
            # If there's an unclosed tag at the end
            if last_open_pos > last_close_pos:
                # Check if it might be incomplete XML content
                incomplete_part = last_200_chars[last_open_pos:]
                if len(incomplete_part) > 1 and not incomplete_part.strip().endswith('>'):
                    logger.warning(f"XML incomplete: unclosed tag at end: '{incomplete_part[:50]}...'")
                    return False
    
    # More lenient bracket balance check - only flag severe imbalances
    open_count = text.count('<')
    close_count = text.count('>')
    if open_count > 0 and close_count > 0:
        imbalance_ratio = abs(open_count - close_count) / max(open_count, close_count)
        # For longer responses, be even more lenient
        threshold = 0.5 if len(text) > 10000 else 0.4  # Increased thresholds
        if imbalance_ratio > threshold:
            logger.warning(f"XML incomplete: severe bracket imbalance (open: {open_count}, close: {close_count}, ratio: {imbalance_ratio:.2f})")
            return False
    
    # If response is suspiciously short for XML structure generation, flag it
    if '<wiki_structure>' in text and len(text) < 100:  # Reduced minimum to be more lenient
        logger.warning("XML incomplete: structure too short")
        return False
    
    # Check for common incomplete XML patterns in long responses
    if len(text) > 5000:
        # For long responses, be extra conservative - only flag if there are major structural issues
        if not text.startswith('<'):
            # Might be a markdown or other format response that's not XML
            logger.debug("Long response doesn't start with XML - might be valid non-XML content")
            return True
    
    # Otherwise assume it's complete - be very conservative
    logger.debug("XML appears complete")
    return True

async def handle_streaming_response(response_stream, validate_xml=False):
    """Handle streaming response from LLM with timeout protection and response validation
    
    Args:
        response_stream: The LLM response stream
        validate_xml: Whether to validate response as XML (for structure queries)
    """
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
                if current_time - last_chunk_time > 150:  # 2.5 minutes without chunks
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
                    chunk_result = await asyncio.wait_for(stream_gen.__anext__(), timeout=180)  # 3 minute chunk timeout (increased)
                    yield chunk_result
                except StopAsyncIteration:
                    break
        except asyncio.TimeoutError:
            logger.error(f"Chunk timeout exceeded - stream terminated prematurely")
            yield json.dumps({"error": "Stream processing timed out - the repository may be too complex. Please try with a smaller repository or contact support."})
            return  # Don't validate incomplete buffer
            
        # Validate response completeness after streaming is done (only for XML responses)
        if response_buffer and validate_xml:
            is_complete = is_xml_complete(response_buffer)
            if not is_complete:
                logger.warning(f"Potentially incomplete XML response detected (length: {len(response_buffer)})")
                logger.warning(f"Response buffer starts with: {response_buffer[:500]}...")
                logger.warning(f"Response buffer ends with: ...{response_buffer[-500:]}")
                
                # Check if this might be a false positive - be more lenient with longer responses
                should_allow = False
                
                # Allow if it has complete wiki_structure tags (most important)
                if '<wiki_structure>' in response_buffer and '</wiki_structure>' in response_buffer:
                    logger.info("Response has complete wiki_structure - allowing despite other validation issues")
                    should_allow = True
                
                # Allow if response is reasonably long and contains substantial content
                elif len(response_buffer) > 8000:
                    # Check if it contains meaningful XML content
                    if response_buffer.count('<') > 10 and response_buffer.count('>') > 10:
                        logger.info(f"Long response ({len(response_buffer)} chars) with XML content - allowing despite validation issues")
                        should_allow = True
                
                # Allow if response ends with proper XML closing pattern
                elif response_buffer.strip().endswith('>') and not response_buffer.strip().endswith('<'):
                    logger.info("Response ends with proper XML closing - allowing")
                    should_allow = True
                    
                if not should_allow:
                    # For incomplete responses, yield an error to prevent client processing
                    yield json.dumps({"error": "Incomplete response detected - please retry"})
                    return
            else:
                logger.debug(f"XML response completeness validation passed (length: {len(response_buffer)})")
        elif response_buffer and not validate_xml:
            logger.debug(f"Non-XML response completed (length: {len(response_buffer)} chars) - skipping XML validation")
            
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
    provider: str = "vllm"
    model: Optional[str] = "/app/models/Qwen3-32B"
    language: Optional[str] = "en"
    excluded_dirs: Optional[str] = None
    excluded_files: Optional[str] = None
    included_dirs: Optional[str] = None
    included_files: Optional[str] = None

async def handle_websocket_chat(websocket: WebSocket):
    logger.info("WebSocket connection request received")
    await websocket.accept()
    logger.info("WebSocket connection accepted")
    
    # Send immediate confirmation that connection is established
    try:
        await websocket.send_text(json.dumps({"status": "connected", "message": "WebSocket connection established"}))
        logger.info("Sent connection confirmation to client")
    except Exception as e:
        logger.error(f"Failed to send connection confirmation: {e}")
        return
    
    try:
        while True:
            # Check if WebSocket is still connected before trying to receive
            if not is_websocket_connected(websocket):
                logger.info("WebSocket client disconnected, breaking loop")
                break
                
            # Add timeout to websocket receive to prevent indefinite waiting
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=600)  # 10 minute timeout for large repos
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
                
            logger.info(f"Received WebSocket message: {data[:200]}..." if len(data) > 200 else f"Received WebSocket message: {data}")
            
            # Parse the incoming message
            try:
                message_data = json.loads(data)
                logger.info(f"Parsed message data keys: {list(message_data.keys()) if isinstance(message_data, dict) else type(message_data)}")
                logger.info(f"🔍 Full message data: {message_data}")
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON received: {e}")
                await websocket.send_text(json.dumps({"error": "Invalid JSON format"}))
                continue
            
            # Handle ping/pong messages for keepalive
            if message_data.get("type") == "ping":
                logger.debug("Received ping, sending pong")
                await websocket.send_text(json.dumps({"type": "pong"}))
                continue
            
            # Handle any other control messages
            if message_data.get("type"):
                logger.debug(f"Received control message type: {message_data.get('type')}")
                # For now, just ignore unknown control message types
                continue
            
            # Validate that this is a chat completion request before processing
            if not isinstance(message_data, dict) or "repo_url" not in message_data:
                logger.warning(f"Invalid message format received: {message_data}")
                logger.warning(f"Message type: {type(message_data)}, has repo_url: {'repo_url' in message_data if isinstance(message_data, dict) else 'N/A'}")
                await websocket.send_text(json.dumps({"error": "Invalid request format - missing repo_url or invalid structure"}))
                continue
            
            # Process regular chat completion request
            try:
                request = ChatCompletionRequest(**message_data)
            except Exception as validation_error:
                logger.error(f"Validation error for ChatCompletionRequest: {validation_error}")
                await websocket.send_text(json.dumps({"error": f"Invalid request: {validation_error}"}))
                continue
            
            # Validate and set default provider if empty
            provider = request.provider.strip() if request.provider else "vllm"
            if not provider:
                provider = "vllm"
                logger.warning(f"Empty provider received, defaulting to: {provider}")
            
            # Validate and set default model if empty
            model = request.model.strip() if request.model else None
            if not model:
                # Set default model based on provider
                if provider == "vllm":
                    model = "/app/models/Qwen3-32B"
                elif provider == "google":
                    model = "gemini-2.0-flash"
                elif provider == "openai":
                    model = "gpt-4"
                else:
                    model = "/app/models/Qwen3-32B"  # fallback
                logger.warning(f"Empty model received, defaulting to: {model}")
            
            logger.info(f"🎯 Request details - Provider: '{provider}', Model: '{model}', Type: '{request.type}', URL: '{request.repo_url}'")
            
            # Send status update that we're starting processing
            await websocket.send_text(json.dumps({"status": "processing", "message": f"Starting processing with {provider} {model}"}))
            
            # Try multiple providers with fallback
            fallback_providers = [
                (provider, model),  # Try requested first
                ("google", "gemini-2.0-flash"),
                ("vllm", "/app/models/Qwen3-32B"), 
                ("openai", "gpt-4o"),
                ("azure", "gpt-4o")
            ]
            
            rag_instance = None
            model_config = None
            successful_provider = None
            successful_model = None
            
            for try_provider, try_model in fallback_providers:
                try:
                    logger.info(f"🔧 Attempting provider: {try_provider}, model: {try_model}")
                    test_rag = RAG(provider=try_provider, model=try_model)
                    test_config = get_model_config(provider=try_provider, model=try_model)
                    
                    # If we get here, configuration succeeded
                    rag_instance = test_rag
                    model_config = test_config
                    successful_provider = try_provider
                    successful_model = try_model
                    
                    logger.info(f"✅ Successfully configured provider: {try_provider}, model: {try_model}")
                    await websocket.send_text(json.dumps({"status": "configured", "message": f"Successfully configured {try_provider} {try_model}"}))
                    break
                    
                except Exception as config_error:
                    logger.warning(f"Provider {try_provider} failed: {config_error}")
                    error_msg = str(config_error).lower()
                    if any(keyword in error_msg for keyword in ['api key', 'authentication', 'unauthorized', 'invalid key', 'missing key']):
                        await websocket.send_text(json.dumps({"status": "provider_failed", "message": f"{try_provider} API key invalid, trying next provider..."}))
                    else:
                        await websocket.send_text(json.dumps({"status": "provider_failed", "message": f"{try_provider} configuration failed, trying next provider..."}))
                    continue
            
            # Check if any provider worked
            if rag_instance is None or model_config is None:
                logger.error("🚨 All LLM providers failed!")
                await websocket.send_text(json.dumps({"error": "No working LLM providers available. Please check your .env file and ensure you have configured valid API keys for at least one LLM provider (Google, OpenAI, vLLM, etc.)"}))
                continue
                
            # Update request with successful provider/model
            provider = successful_provider
            model = successful_model

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
                logger.info(f"🚀 Starting stream_response with request: {validated_request.repo_url}")
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
    logger.info(f"🔄 stream_response started for repo: {request.repo_url}")
    query = request.messages[-1].content if request.messages else ""
    logger.info(f"📝 Query length: {len(query)} characters")
    model_kwargs = model_config.get("model_kwargs", {})
    context_window = get_context_window_for_model(request.provider, request.model)
    max_completion_tokens = get_max_tokens_for_model(request.provider, request.model)
    allowed_prompt_size = context_window - max_completion_tokens
    logger.info(f"🔧 Model config - Context: {context_window}, Max tokens: {max_completion_tokens}")
    
    # Ensure max_tokens in model_kwargs doesn't exceed the configured limit
    if "max_tokens" in model_kwargs:
        model_kwargs["max_tokens"] = min(model_kwargs["max_tokens"], max_completion_tokens)
    else:
        model_kwargs["max_tokens"] = max_completion_tokens

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
    logger.info(f"🔍 Structure query check: {is_structure_query}")
    
    if is_structure_query:
        logger.info("🏗️  Detected wiki structure determination query - skipping RAG retrieval")
        # For structure queries, we don't need RAG content, just pass through the query
        prompt = query
        logger.info(f"📋 Structure query prompt length: {len(prompt)} characters")
        
        client = model_config["model_client"](**model_config.get("initialize_kwargs", {}))
        model_kwargs["stream"] = True
        
        logger.info("Preparing to stream response from LLM (structure query).")
        try:
            logger.info(f"🔧 Creating API kwargs for structure query - provider: {request.provider}, model: {request.model}")
            api_kwargs = client.convert_inputs_to_api_kwargs(input=prompt, model_kwargs=model_kwargs, model_type=ModelType.LLM)
            logger.info(f"✅ API kwargs prepared: {list(api_kwargs.keys())}")
            logger.info(f"🚀 About to call LLM API for structure query...")
            
            # Add timeout to the initial LLM call to prevent hanging on connection
            try:
                response_stream = await asyncio.wait_for(
                    client.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM),
                    timeout=300  # 5 minute timeout for initial connection (increased from 3 min)
                )
                logger.info("Successfully connected to LLM API and received response stream (structure query)")
            except asyncio.TimeoutError:
                logger.error("LLM API connection timeout - no response within 5 minutes (structure query)")
                yield json.dumps({"error": "LLM API connection timeout - repository may be too large. Please try a smaller repository or contact support."})
                return
                
            # Handle streaming response with XML validation for structure queries
            async for chunk in handle_streaming_response(response_stream, validate_xml=True):
                yield chunk
                
        except Exception as e:
            logger.error(f"Error in streaming response (structure query): {e}", exc_info=True)
            yield json.dumps({"error": f"Error generating response: {e}"})

        logger.info("Sending 'done' status (structure query).")
        yield json.dumps({"status": "done"})
        return

    # For content generation queries, continue with RAG preparation
    logger.info("🔧 Starting RAG preparation for content generation")
    yield json.dumps({"status": "preparing_rag", "message": "Analyzing repository structure and content..."})
    
    try:
        logger.info(f"📂 Preparing retriever for: {request.repo_url} (type: {request.type})")
        rag_instance.prepare_retriever(
            request.repo_url, request.type, request.token,
            request.excluded_dirs.split(',') if request.excluded_dirs else None,
            request.excluded_files.split(',') if request.excluded_files else None,
            request.included_dirs.split(',') if request.included_dirs else None,
            request.included_files.split(',') if request.included_files else None,
        )
        logger.info("✅ RAG retriever preparation completed")
        yield json.dumps({"status": "rag_ready", "message": "Repository analysis complete"})
        
        # Diagnostic check for retriever state
        if not hasattr(rag_instance, 'retriever') or rag_instance.retriever is None:
            logger.error("❌ Retriever not initialized after prepare_retriever call")
            yield json.dumps({"error": "Repository retriever failed to initialize"})
            return
            
        if not hasattr(rag_instance, 'transformed_docs') or not rag_instance.transformed_docs:
            logger.error("❌ No transformed documents available for retrieval")
            yield json.dumps({"error": "No repository documents available for wiki generation"})
            return
            
    except Exception as rag_error:
        logger.error(f"RAG preparation failed: {rag_error}", exc_info=True)
        yield json.dumps({"error": f"Repository processing failed: {rag_error}"})
        return
        
    logger.info(f"✅ Retriever ready with {len(rag_instance.transformed_docs)} documents")
    yield json.dumps({"status": "retrieving_context", "message": f"Retrieving relevant context from {len(rag_instance.transformed_docs)} documents..."})

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
    yield json.dumps({"status": "context_ready", "message": f"Retrieved {len(context_text)} characters of context from {len(seen_sources)} files"})
    
    # Enhanced logging for debugging and content validation
    if len(seen_sources) > 0:
        logger.info(f"📁 Source files included: {list(seen_sources)[:10]}{'...' if len(seen_sources) > 10 else ''}")
        
        # Check if we have actual source code files (not just README/docs)
        source_code_files = [s for s in seen_sources if any(s.endswith(ext) for ext in ['.py', '.js', '.ts', '.go', '.java', '.cpp', '.c', '.h', '.json', '.yaml', '.yml'])]
        doc_files = [s for s in seen_sources if any(s.lower().endswith(ext) for ext in ['.md', '.txt', '.rst'])]
        
        logger.info(f"📊 Content breakdown: {len(source_code_files)} source files, {len(doc_files)} documentation files")
        
        if len(source_code_files) == 0 and len(doc_files) > 0:
            logger.warning("⚠️  Only documentation files found - may trigger README-only error")
        elif len(source_code_files) > 0:
            logger.info(f"✅ Found actual source code files: {source_code_files[:5]}{'...' if len(source_code_files) > 5 else ''}")
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
                file_tree=context_text[:15000] if context_text else "File tree not available",
                readme=file_content[:5000] if file_content else "README not available", 
                context=context_text[:50000] if context_text else "Repository context not available"  # Increased from 10K to 50K
            )
        else:
            # Use detailed page generation prompt for component-specific queries
            system_prompt = WIKI_PAGE_GENERATION_PROMPT.format(
                context=context_text[:30000] if context_text else "Context not available",  # Increased from 5K to 30K
                file_content=file_content[:15000] if file_content else "File content not available",  # Increased from 10K to 15K
                page_topic=query
            )
    conversation_history = "\n".join([f"{m.role}: {m.content}" for m in request.messages[:-1]])

    if is_structure_query:
        # For structure queries, the system_prompt already contains everything we need
        prompt = system_prompt
        logger.info(f"📋 Structure query prompt length: {len(prompt)} characters")
        
        # Final safety check for structure queries too
        prompt_tokens = count_tokens(prompt)
        total_tokens = prompt_tokens + max_completion_tokens
        if total_tokens > context_window:
            logger.warning(f"⚠️  Structure query prompt too large: {prompt_tokens} + {max_completion_tokens} = {total_tokens} > {context_window}")
            # Emergency truncation for structure queries
            max_prompt_tokens = context_window - max_completion_tokens - 200  # Extra safety buffer for complete responses
            if max_prompt_tokens > 0:
                truncation_ratio = max_prompt_tokens / prompt_tokens
                if truncation_ratio < 1:
                    truncated_length = int(len(prompt) * truncation_ratio * 0.9)  # 0.9 for safety
                    prompt = prompt[:truncated_length] + "\n\n[Content truncated due to length]"
                    logger.warning(f"Applied emergency truncation to structure query: {len(prompt)} characters")
            else:
                logger.error("Cannot fit structure query. Context window too small.")
                yield json.dumps({"error": "Context window too small for this structure query"})
                return
    else:
        # For content generation, combine all context
        file_content, context_text = truncate_prompt_to_fit(
            context_window, max_completion_tokens, system_prompt, conversation_history, file_content, context_text, query
        )
        prompt = f"{system_prompt}\n\nHistory: {conversation_history}\n\nFile Context:\n{file_content}\n\nRetrieved Context:\n{context_text}\n\nQuery: {query}"
        logger.info(f"📝 Content generation prompt length: {len(prompt)} characters")
        logger.info(f"🔍 Context breakdown - File: {len(file_content)}, Retrieved: {len(context_text)}, History: {len(conversation_history)}")
        
        # Final safety check: ensure prompt + completion doesn't exceed context window
        prompt_tokens = count_tokens(prompt)
        total_tokens = prompt_tokens + max_completion_tokens
        if total_tokens > context_window:
            logger.warning(f"⚠️  Final prompt still too large: {prompt_tokens} + {max_completion_tokens} = {total_tokens} > {context_window}")
            # Emergency truncation: cut the prompt to fit
            max_prompt_tokens = context_window - max_completion_tokens - 200  # Extra safety buffer for complete responses
            if max_prompt_tokens > 0:
                # Simple truncation - keep the beginning (system prompt) and end (query)
                truncation_ratio = max_prompt_tokens / prompt_tokens
                if truncation_ratio < 1:
                    truncated_length = int(len(prompt) * truncation_ratio * 0.9)  # 0.9 for safety
                    prompt = prompt[:truncated_length] + "\n\n[Content truncated due to length]\n\nQuery: " + query
                    logger.warning(f"Applied emergency truncation: {len(prompt)} characters")
            else:
                logger.error("Cannot fit any meaningful content. Context window too small.")
                yield json.dumps({"error": "Context window too small for this query"})
                return
    
    client = model_config["model_client"](**model_config.get("initialize_kwargs", {}))
    model_kwargs["stream"] = True

    yield json.dumps({"status": "generating", "message": "Generating AI response..."})
    logger.info("Preparing to stream response from LLM.")
    try:
        logger.info(f"🔧 Creating API kwargs for content generation - provider: {request.provider}, model: {request.model}")
        api_kwargs = client.convert_inputs_to_api_kwargs(input=prompt, model_kwargs=model_kwargs, model_type=ModelType.LLM)
        logger.info(f"✅ API kwargs prepared: {list(api_kwargs.keys())}")
        logger.info(f"🚀 About to call LLM API for content generation...")
        
        # Add timeout to the initial LLM call to prevent hanging on connection
        try:
            response_stream = await asyncio.wait_for(
                client.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM),
                timeout=300  # 5 minute timeout for initial connection (increased from 3 min)
            )
            logger.info("Successfully connected to LLM API and received response stream")
        except asyncio.TimeoutError:
            logger.error("LLM API connection timeout - no response within 5 minutes")
            yield json.dumps({"error": "LLM API connection timeout - repository may be too large. Please try a smaller repository or contact support."})
            return
        
        # Use the shared streaming handler (no XML validation for page content)
        async for chunk in handle_streaming_response(response_stream, validate_xml=False):
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
