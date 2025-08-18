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
from .wiki_prompts import WIKI_PAGE_GENERATION_PROMPT, ARCHITECTURE_OVERVIEW_PROMPT, WIKI_STRUCTURE_ANALYSIS_PROMPT, HIERARCHICAL_WIKI_GENERATION_PROMPT

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
        # For longer responses, be even more lenient - especially for wiki structure generation
        if len(text) > 20000:  # Very large responses (likely wiki structures)
            threshold = 0.7  # Very lenient for large structures
        elif len(text) > 10000:
            threshold = 0.6  # More lenient for medium responses
        else:
            threshold = 0.5  # Standard threshold for smaller responses
            
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
                
                # Special handling for wiki structure responses that might be truncated but still useful
                elif '<wiki_structure>' in response_buffer and len(response_buffer) > 15000:
                    # Check if we have substantial structure content even if not perfectly closed
                    section_count = response_buffer.count('<section')
                    subsection_count = response_buffer.count('<subsection')
                    if section_count > 0 or subsection_count > 0:
                        logger.info(f"Large wiki structure response ({len(response_buffer)} chars) with {section_count} sections, {subsection_count} subsections - allowing")
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
    type: Optional[str] = "bitbucket"
    provider: str = "vllm"
    model: Optional[str] = "/app/models/Qwen3-32B"
    # Advanced file filtering options removed

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
            
            # Handle control messages (but NOT repository type messages)
            # Repository "type" (bitbucket, local) is part of chat requests, not control messages
            message_type = message_data.get("type")
            if message_type and message_type in ["ping", "pong", "heartbeat", "control"]:  # Only actual control message types
                logger.debug(f"Received control message type: {message_type}")
                # For now, just ignore unknown control message types
                continue
            
            # Log if we have a "type" field that might be repo type vs control type
            if message_type:
                logger.info(f"🔍 Message has type='{message_type}' - treating as repository type, not control message")
            
            # Validate that this is a chat completion request before processing
            if not isinstance(message_data, dict) or "repo_url" not in message_data:
                logger.warning(f"Invalid message format received: {message_data}")
                logger.warning(f"Message type: {type(message_data)}, has repo_url: {'repo_url' in message_data if isinstance(message_data, dict) else 'N/A'}")
                await websocket.send_text(json.dumps({"error": "Invalid request format - missing repo_url or invalid structure"}))
                continue
            
            # Process regular chat completion request
            try:
                logger.info(f"🔍 Attempting to create ChatCompletionRequest from message_data")
                logger.debug(f"Message data keys: {list(message_data.keys())}")
                logger.debug(f"Messages field: {message_data.get('messages', 'MISSING')}")
                
                request = ChatCompletionRequest(**message_data)
                logger.info(f"✅ Successfully created ChatCompletionRequest")
                logger.info(f"Request details: repo_url={request.repo_url[:50]}..., provider={request.provider}, model={request.model}")
                
            except Exception as validation_error:
                logger.error(f"🚨 Validation error for ChatCompletionRequest: {validation_error}", exc_info=True)
                logger.error(f"Failed message_data: {message_data}")
                await websocket.send_text(json.dumps({"error": f"Invalid request format: {validation_error}. Please check your request structure."}))
                continue
            
            # Validate and set default provider if empty
            provider = request.provider.strip() if request.provider else "vllm"
            if not provider:
                provider = "vllm"
                logger.warning(f"Empty provider received, defaulting to: {provider}")
            
            # Validate and set default model if empty
            model = request.model.strip() if request.model else None
            if not model:
                # Set default model for vLLM provider
                if provider == "vllm":
                    model = "/app/models/Qwen2.5-VL-7B-Instruct"
                else:
                    model = "/app/models/Qwen3-32B"  # fallback
                logger.warning(f"Empty model received, defaulting to: {model}")
            
            logger.info(f"🎯 Request details - Provider: '{provider}', Model: '{model}', Type: '{request.type}', URL: '{request.repo_url}'")
            
            # Send status update that we're starting processing
            await websocket.send_text(json.dumps({"status": "processing", "message": f"Starting processing with {provider} {model}"}))
            
            # Only vLLM provider is supported
            fallback_providers = [
                (provider, model),  # Try requested first
                ("vllm", "/app/models/Qwen2.5-VL-7B-Instruct"),
                ("vllm", "/app/models/Llama-3.3-70B-Instruct"),
                ("vllm", "/app/models/Qwen3-32B")
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
                model=model
            )

            chunk_count = 0
            start_time = asyncio.get_event_loop().time()
            
            # Add timeout protection for the entire stream_response process
            stream_timeout = 1200  # 20 minutes total timeout for entire process
            
            try:
                logger.info(f"🚀 Starting stream_response with request: {validated_request.repo_url}")
                
                # Wrap the entire stream processing in a timeout
                try:
                    stream_started = False
                    last_activity = asyncio.get_event_loop().time()
                    
                    async for chunk in stream_response(validated_request, rag_instance, model_config):
                        # Track first chunk received
                        if not stream_started:
                            logger.info("✅ Stream response generation started successfully")
                            stream_started = True
                        
                        chunk_count += 1
                        last_activity = asyncio.get_event_loop().time()
                        
                        # Check for overall timeout
                        if last_activity - start_time > stream_timeout:
                            logger.error(f"🚨 Stream processing exceeded {stream_timeout} second timeout")
                            if is_websocket_connected(websocket):
                                await websocket.send_text(json.dumps({
                                    "error": f"Request timed out after {stream_timeout//60} minutes. The repository may be too complex or the AI service is overloaded."
                                }))
                            break
                        
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
                    
                    # Check if stream ever started
                    if not stream_started:
                        logger.error("❌ Stream response generator never yielded any content")
                        if is_websocket_connected(websocket):
                            await websocket.send_text(json.dumps({
                                "error": "Stream response generator failed to start - no content was generated. Please check server logs."
                            }))
                        chunk_count = 0
                
                except Exception as stream_gen_error:
                    logger.error(f"🚨 Critical error in stream response generator: {stream_gen_error}", exc_info=True)
                    if is_websocket_connected(websocket):
                        await websocket.send_text(json.dumps({
                            "error": f"Stream generation failed: {stream_gen_error}. Please check server logs and try again."
                        }))
                    chunk_count = 0  # Ensure completion signal reflects error
                        
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
    
    # Add validation for required parameters
    if not request or not request.repo_url:
        logger.error("❌ Invalid request: missing repo_url")
        yield json.dumps({"error": "Invalid request: missing repository URL"})
        return
    
    if not rag_instance:
        logger.error("❌ Invalid request: RAG instance is None")
        yield json.dumps({"error": "Internal error: RAG instance not initialized"})
        return
        
    if not model_config:
        logger.error("❌ Invalid request: model_config is None")
        yield json.dumps({"error": "Internal error: model configuration not available"})
        return
    
    try:
        query = request.messages[-1].content if request.messages else ""
        logger.info(f"📝 Query length: {len(query)} characters")
        
        if not query.strip():
            logger.error("❌ Empty query received")
            yield json.dumps({"error": "Empty query - please provide a question or request"})
            return
        
        model_kwargs = model_config.get("model_kwargs", {})
        context_window = get_context_window_for_model(request.provider, request.model)
        max_completion_tokens = get_max_tokens_for_model(request.provider, request.model)
        allowed_prompt_size = context_window - max_completion_tokens
        logger.info(f"🔧 Model config - Context: {context_window}, Max tokens: {max_completion_tokens}")
        
        # Validate model configuration
        if context_window <= 0 or max_completion_tokens <= 0:
            logger.error(f"❌ Invalid model configuration: context_window={context_window}, max_tokens={max_completion_tokens}")
            yield json.dumps({"error": f"Invalid model configuration for {request.provider}/{request.model}"})
            return
            
    except Exception as init_error:
        logger.error(f"❌ Error in stream_response initialization: {init_error}", exc_info=True)
        yield json.dumps({"error": f"Failed to initialize response generation: {init_error}"})
        return
    
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
        
        # For wiki structure generation, we need more tokens to generate complete XML
        # Increase max_completion_tokens for structure queries to prevent truncation
        original_max_tokens = max_completion_tokens
        
        # Calculate dynamic token limit based on content complexity
        file_tree_size = len([line for line in query.split('\n') if line.strip() and not line.startswith('<')])
        estimated_structure_tokens = file_tree_size * 15  # Rough estimate: 15 tokens per file/folder
        
        # Set minimum safe token limit for structure generation
        min_structure_tokens = max(12000, estimated_structure_tokens)
        
        # Use higher limit for structure queries, but respect context window
        structure_max_tokens = min(min_structure_tokens, context_window - count_tokens(prompt) - 1000)
        
        if structure_max_tokens > max_completion_tokens:
            logger.info(f"🔧 Increasing completion tokens for structure query: {max_completion_tokens} → {structure_max_tokens}")
            max_completion_tokens = structure_max_tokens
            model_kwargs["max_tokens"] = structure_max_tokens
        
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
        
        # Add timeout protection for RAG preparation
        rag_preparation_timeout = 600  # 10 minutes for repository analysis
        
        try:
            await asyncio.wait_for(
                asyncio.to_thread(
                    rag_instance.prepare_retriever,
                    request.repo_url, 
                    request.type, 
                    request.token
                ),
                timeout=rag_preparation_timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"🚨 RAG preparation timed out after {rag_preparation_timeout} seconds")
            yield json.dumps({"error": f"Repository analysis timed out after {rag_preparation_timeout//60} minutes. The repository may be too large or complex. Please try with a smaller repository."})
            return
        except RuntimeError as runtime_error:
            logger.error(f"🚨 RAG CRITICAL ERROR during preparation: {runtime_error}", exc_info=True)
            yield json.dumps({"error": f"RAG system initialization failed: {runtime_error}. This indicates a critical configuration or dependency issue."})
            return
        except Exception as prep_error:
            logger.error(f"🚨 RAG preparation failed with exception: {prep_error}", exc_info=True)
            yield json.dumps({"error": f"Repository analysis failed: {prep_error}. Please check if the repository URL is accessible and try again."})
            return
        
        logger.info("✅ RAG retriever preparation completed")
        yield json.dumps({"status": "rag_ready", "message": "Repository analysis complete"})
        
        # Enhanced diagnostic checks for retriever state
        try:
            if not hasattr(rag_instance, 'retriever') or rag_instance.retriever is None:
                logger.error("❌ Retriever not initialized after prepare_retriever call")
                logger.error(f"RAG instance attributes: {dir(rag_instance)}")
                yield json.dumps({"error": "Repository retriever failed to initialize. Please check if the repository contains valid source code files."})
                return
                
            if not hasattr(rag_instance, 'transformed_docs') or not rag_instance.transformed_docs:
                logger.error("❌ No transformed documents available for retrieval")
                logger.error(f"RAG instance transformed_docs: {getattr(rag_instance, 'transformed_docs', 'MISSING')}")
                yield json.dumps({"error": "No repository documents available for wiki generation. The repository may be empty or contain only unsupported file types."})
                return
                
            logger.info(f"✅ RAG validation passed - retriever ready with {len(rag_instance.transformed_docs)} documents")
            
        except Exception as validation_error:
            logger.error(f"❌ RAG validation failed: {validation_error}", exc_info=True)
            yield json.dumps({"error": f"Repository validation failed: {validation_error}"})
            return
            
    except Exception as rag_error:
        logger.error(f"🚨 Critical RAG preparation error: {rag_error}", exc_info=True)
        yield json.dumps({"error": f"Repository processing failed: {rag_error}. Please check server logs and try again."})
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
    
    try:
        for broad_query in comprehensive_queries:
            try:
                logger.debug(f"🔍 Executing RAG query: '{broad_query[:50]}...'")
                result = rag_instance.call(broad_query)
                logger.debug(f"RAG call result type: {type(result)}, content preview: {str(result)[:100]}...")
                
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
                
                logger.info(f"✅ Retrieved {len(docs)} docs for query: '{broad_query[:30]}...'")
                
                for doc in docs:
                    if doc is None:
                        continue
                        
                    try:
                        source = doc.metadata.get('source', 'unknown') if hasattr(doc, 'metadata') and doc.metadata else 'unknown'
                        if source not in seen_sources:
                            all_retrieved_docs.append(doc)
                            seen_sources.add(source)
                            logger.debug(f"Added doc from source: {source}")
                    except Exception as doc_error:
                        logger.warning(f"Error processing document: {doc_error}")
                        continue
                        
            except RuntimeError as runtime_error:
                # These are critical RAG errors that should not be ignored
                logger.error(f"❌ RAG CRITICAL ERROR during query '{broad_query[:30]}...': {runtime_error}")
                logger.error("This is a critical RAG system failure - cannot continue with empty context")
                yield json.dumps({"error": f"RAG system failure: {runtime_error}. Please check your configuration."})
                return
            except Exception as query_error:
                logger.warning(f"Failed to retrieve docs for query '{broad_query[:30]}...': {query_error}")
                logger.debug(f"Query exception details: {type(query_error).__name__}: {str(query_error)}")
                continue
    except Exception as retrieval_error:
        logger.error(f"🚨 Critical error during context retrieval: {retrieval_error}", exc_info=True)
        yield json.dumps({"error": f"Context retrieval failed: {retrieval_error}. Please try again."})
        return
    
    # If we still don't have enough content, try to get more documents with a very broad query
    if len(all_retrieved_docs) < 10:
        try:
            # Use a very broad query to capture more of the codebase
            result = rag_instance.call("source code implementation functions classes")
            
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
    
    # Critical content validation - ensure we have meaningful context
    if len(all_retrieved_docs) < 5:
        logger.error(f"❌ INSUFFICIENT CONTENT: Only {len(all_retrieved_docs)} documents retrieved")
        logger.error("This will result in poor wiki quality with generic content")
        logger.info("🔄 Making aggressive retrieval attempt to get comprehensive context...")
        
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
    
    # Combine all retrieved content with smart token management
    context_text_parts = []
    total_context_tokens = 0
    max_context_tokens = 20000  # Increased limit for better content quality
    
    # Sort documents by relevance/importance if available
    sorted_docs = all_retrieved_docs
    
    for doc in sorted_docs:
        doc_tokens = count_tokens(doc.text)
        
        # If adding this doc would exceed limit, try to truncate it instead of skipping
        if total_context_tokens + doc_tokens > max_context_tokens:
            remaining_tokens = max_context_tokens - total_context_tokens
            if remaining_tokens > 500:  # Only if we have meaningful space left
                # Smart truncation - keep beginning of document
                truncate_ratio = remaining_tokens / doc_tokens
                truncated_length = int(len(doc.text) * truncate_ratio * 0.9)
                truncated_text = doc.text[:truncated_length] + "\n[Document truncated...]"
                context_text_parts.append(truncated_text)
                total_context_tokens += count_tokens(truncated_text)
                logger.info(f"Truncated document to fit: {doc_tokens} -> {count_tokens(truncated_text)} tokens")
            break
        
        context_text_parts.append(doc.text)
        total_context_tokens += doc_tokens
    
    context_text = "\n\n".join(context_text_parts)
    logger.info(f"📝 Final context: {total_context_tokens} tokens from {len(context_text_parts)} documents")
    
    logger.info(f"📊 Wiki generation final stats: {len(all_retrieved_docs)} documents from {len(seen_sources)} unique sources")
    logger.info(f"📝 Total context length: {len(context_text)} characters")
    
    # Content quality warning
    if len(all_retrieved_docs) < 5:
        logger.warning(f"⚠️  LOW CONTENT WARNING: Only {len(all_retrieved_docs)} docs retrieved - wiki quality may be poor")
        yield json.dumps({"status": "context_warning", "message": f"Warning: Limited context retrieved ({len(seen_sources)} files). Wiki may lack repository-specific details."})
    elif len(context_text) < 10000:
        logger.warning(f"⚠️  SHORT CONTENT WARNING: Only {len(context_text)} chars - wiki may lack depth")
        yield json.dumps({"status": "context_warning", "message": f"Warning: Limited context content ({len(context_text)} chars). Wiki may lack detailed explanations."})
    else:
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
    elif query.startswith("[CLEAN_WIKI_PREFORMATTED_PROMPT]"):
        # For clean wiki generation, use the pre-formatted prompt directly
        logger.info("🎯 Processing clean wiki with pre-formatted HIERARCHICAL_WIKI_GENERATION_PROMPT")
        system_prompt = query.replace("[CLEAN_WIKI_PREFORMATTED_PROMPT]\n", "")
        logger.info(f"✅ Using pre-formatted prompt: {len(system_prompt)} characters")
        
        # Allocate more tokens for comprehensive 6-chapter wiki generation
        estimated_wiki_tokens = 15000  # Need substantial tokens for 6 complete chapters
        wiki_max_tokens = min(estimated_wiki_tokens, context_window - count_tokens(system_prompt) - 500)
        
        if wiki_max_tokens > max_completion_tokens:
            logger.info(f"🔧 Increasing completion tokens for comprehensive wiki: {max_completion_tokens} → {wiki_max_tokens}")
            max_completion_tokens = wiki_max_tokens
            model_kwargs["max_tokens"] = wiki_max_tokens
    else:
        # For content generation, use intelligently chosen prompts based on query type
        query_lower = query.lower()
        
        if any(keyword in query_lower for keyword in ['architecture', 'overview', 'system', 'structure']):
            # Use architecture overview prompt for system-level queries
            logger.info("🏛️  Using architecture overview prompt for comprehensive analysis")
            
            # Extract file tree and README from structured query if present
            extracted_file_tree = "File tree not available"
            extracted_readme = "README not available"
            
            # Check if query contains structured file tree and README
            if '<file_tree>' in query and '</file_tree>' in query:
                try:
                    start_tag = query.find('<file_tree>') + len('<file_tree>')
                    end_tag = query.find('</file_tree>')
                    extracted_file_tree = query[start_tag:end_tag].strip()
                    logger.info(f"📁 Extracted file tree: {len(extracted_file_tree)} characters")
                except Exception as e:
                    logger.warning(f"Failed to extract file tree: {e}")
            
            if '<readme>' in query and '</readme>' in query:
                try:
                    start_tag = query.find('<readme>') + len('<readme>')
                    end_tag = query.find('</readme>')
                    extracted_readme = query[start_tag:end_tag].strip()
                    logger.info(f"📖 Extracted README: {len(extracted_readme)} characters")
                except Exception as e:
                    logger.warning(f"Failed to extract README: {e}")
            
            system_prompt = ARCHITECTURE_OVERVIEW_PROMPT.format(
                file_tree=extracted_file_tree[:15000] if extracted_file_tree != "File tree not available" else "File tree not available",
                readme=extracted_readme[:5000] if extracted_readme != "README not available" else "README not available", 
                context=context_text[:50000] if context_text else "Repository context not available"  # Increased from 10K to 50K
            )
            logger.info(f"✅ ARCHITECTURE_OVERVIEW_PROMPT formatted successfully:")
            logger.info(f"   - File tree length: {len(extracted_file_tree)} chars")
            logger.info(f"   - README length: {len(extracted_readme)} chars") 
            logger.info(f"   - Context length: {len(context_text)} chars")
            logger.info(f"   - Final prompt length: {len(system_prompt)} chars")
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
        logger.info(f"🔍 Structure token check: prompt={prompt_tokens}, completion={max_completion_tokens}, total={total_tokens}, limit={context_window}")
        
        if total_tokens > context_window:
            logger.warning(f"⚠️  Structure query prompt too large: {prompt_tokens} + {max_completion_tokens} = {total_tokens} > {context_window}")
            # Emergency truncation for structure queries with larger safety buffer
            max_prompt_tokens = context_window - max_completion_tokens - 500  # Larger safety buffer
            if max_prompt_tokens > 0:
                truncation_ratio = max_prompt_tokens / prompt_tokens
                if truncation_ratio < 1:
                    truncated_length = int(len(prompt) * truncation_ratio * 0.85)  # More conservative
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
        logger.info(f"🔍 Token check: prompt={prompt_tokens}, completion={max_completion_tokens}, total={total_tokens}, limit={context_window}")
        
        if total_tokens > context_window:
            logger.warning(f"⚠️  Final prompt still too large: {prompt_tokens} + {max_completion_tokens} = {total_tokens} > {context_window}")
            # Emergency truncation: cut the prompt to fit with larger safety buffer
            max_prompt_tokens = context_window - max_completion_tokens - 500  # Larger safety buffer
            if max_prompt_tokens > 0:
                # Simple truncation - keep the beginning (system prompt) and end (query)
                truncation_ratio = max_prompt_tokens / prompt_tokens
                if truncation_ratio < 1:
                    truncated_length = int(len(prompt) * truncation_ratio * 0.85)  # More conservative
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
        
        # Validate client is available
        if not client:
            logger.error("❌ LLM client is None")
            yield json.dumps({"error": "LLM client not available. Please check provider configuration."})
            return
            
        try:
            api_kwargs = client.convert_inputs_to_api_kwargs(input=prompt, model_kwargs=model_kwargs, model_type=ModelType.LLM)
            logger.info(f"✅ API kwargs prepared: {list(api_kwargs.keys())}")
        except Exception as kwargs_error:
            logger.error(f"❌ Failed to create API kwargs: {kwargs_error}", exc_info=True)
            yield json.dumps({"error": f"Failed to prepare API request: {kwargs_error}"})
            return
        
        logger.info(f"🚀 About to call LLM API for content generation...")
        
        # Add timeout to the initial LLM call to prevent hanging on connection
        try:
            response_stream = await asyncio.wait_for(
                client.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM),
                timeout=300  # 5 minute timeout for initial connection (increased from 3 min)
            )
            logger.info("✅ Successfully connected to LLM API and received response stream")
            
            # Validate response stream
            if response_stream is None:
                logger.error("❌ LLM API returned None response stream")
                yield json.dumps({"error": "LLM API returned empty response. Please try again."})
                return
                
        except asyncio.TimeoutError:
            logger.error("🚨 LLM API connection timeout - no response within 5 minutes")
            yield json.dumps({"error": "LLM API connection timeout - the repository may be too complex or the AI service is overloaded. Please try a smaller repository or contact support."})
            return
        except Exception as api_error:
            logger.error(f"🚨 LLM API call failed: {api_error}", exc_info=True)
            error_msg = str(api_error).lower()
            if "api key" in error_msg or "unauthorized" in error_msg or "authentication" in error_msg:
                yield json.dumps({"error": f"API authentication failed for {request.provider}. Please check your API key configuration."})
            elif "quota" in error_msg or "rate limit" in error_msg:
                yield json.dumps({"error": f"API quota exceeded for {request.provider}. Please try again later or use a different provider."})
            elif "timeout" in error_msg:
                yield json.dumps({"error": f"API request timed out for {request.provider}. Please try again."})
            else:
                yield json.dumps({"error": f"LLM API error ({request.provider}): {api_error}"})
            return
        
        # Use the shared streaming handler (no XML validation for page content)
        chunk_yielded = False
        try:
            async for chunk in handle_streaming_response(response_stream, validate_xml=False):
                chunk_yielded = True
                yield chunk
                
            if not chunk_yielded:
                logger.warning("⚠️  No chunks received from streaming handler")
                yield json.dumps({"error": "No response generated. The request may have failed silently."})
                
        except Exception as stream_error:
            logger.error(f"🚨 Error in streaming response handler: {stream_error}", exc_info=True)
            yield json.dumps({"error": f"Streaming response failed: {stream_error}"})
            return

    except Exception as generation_error:
        logger.error(f"🚨 Critical error in response generation: {generation_error}", exc_info=True)
        yield json.dumps({"error": f"Response generation failed: {generation_error}. Please check server logs and try again."})
        return

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
        
        summary_text = client.chat_completion_parser(summary_response)
        summaries.append(summary_text)

    combined_summary = " ".join(summaries)
    final_summary_prompt = f"Create a final, coherent summary from the following smaller summaries: {combined_summary}"
    api_kwargs = client.convert_inputs_to_api_kwargs(input=final_summary_prompt, model_kwargs=summary_kwargs, model_type=ModelType.LLM)
    final_summary_response = await client.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)

    final_summary = client.chat_completion_parser(final_summary_response)
    
    logger.info(f"Original query summarized to {count_tokens(final_summary)} tokens.")
    return final_summary


# --- Clean Wiki Generation WebSocket Handler ---

class CleanWikiGenerationRequest(BaseModel):
    """Clean request model for WebSocket wiki generation."""
    repo_url: str
    repo_type: str = "bitbucket"
    provider: str = "vllm" 
    model: str = "/app/models/Qwen2.5-VL-7B-Instruct"
    token: Optional[str] = None
    local_path: Optional[str] = None
    language: str = "en"


async def handle_clean_wiki_generation(websocket: WebSocket):
    """
    Clean WebSocket handler for wiki generation.
    
    Takes only repository details - handles all prompt engineering internally.
    No prompts are passed from frontend to backend.
    """
    logger.info("🎯 Clean wiki generation WebSocket connection request received")
    await websocket.accept()
    logger.info("✅ Clean wiki generation WebSocket connection accepted")
    
    # Send immediate confirmation
    try:
        await websocket.send_text(json.dumps({
            "status": "connected", 
            "message": "Ready for clean wiki generation"
        }))
        logger.info("📨 Sent connection confirmation to client")
    except Exception as e:
        logger.error(f"Failed to send connection confirmation: {e}")
        return
    
    try:
        while True:
            # Check if WebSocket is still connected
            if not is_websocket_connected(websocket):
                logger.info("WebSocket client disconnected, breaking loop")
                break
                
            # Receive wiki generation request
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=600)
            except asyncio.TimeoutError:
                logger.warning("WebSocket receive timeout - sending heartbeat")
                if not is_websocket_connected(websocket):
                    break
                try:
                    await websocket.send_text(json.dumps({"status": "heartbeat", "message": "Connection alive"}))
                except WebSocketDisconnect:
                    break
                continue
            except WebSocketDisconnect:
                logger.info("WebSocket client disconnected")
                break
            except Exception as receive_error:
                logger.error(f"Error receiving WebSocket data: {receive_error}")
                break
                
            logger.info(f"📨 Received wiki generation request")
            
            # Parse the request
            try:
                request_data = json.loads(data)
                request = CleanWikiGenerationRequest(**request_data)
                logger.info(f"✅ Parsed clean wiki request: {request.repo_url}")
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON received: {e}")
                await websocket.send_text(json.dumps({"error": "Invalid JSON format"}))
                continue
            except Exception as validation_error:
                logger.error(f"Request validation error: {validation_error}")
                await websocket.send_text(json.dumps({"error": f"Invalid request format: {validation_error}"}))
                continue
            
            # Handle ping/pong
            if request_data.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
                continue
            
            # Generate wiki internally using backend logic
            try:
                await generate_clean_wiki(websocket, request)
                
            except Exception as generation_error:
                logger.error(f"Wiki generation error: {generation_error}", exc_info=True)
                await safe_websocket_send(websocket, {
                    "error": f"Wiki generation failed: {str(generation_error)}"
                })
                
    except WebSocketDisconnect:
        logger.info("Clean wiki generation WebSocket disconnected")
    except Exception as e:
        logger.error(f"Unexpected error in clean wiki generation handler: {e}", exc_info=True)
    finally:
        # Clean up any resources
        try:
            await websocket.close()
        except:
            pass


async def generate_clean_wiki(websocket: WebSocket, request: CleanWikiGenerationRequest):
    """
    Generate comprehensive structured wiki using proper two-step process.
    
    This function handles:
    1. Repository analysis and file tree extraction
    2. Wiki structure generation using WIKI_STRUCTURE_ANALYSIS_PROMPT
    3. Content generation for each structured section
    4. Streaming response back to frontend
    """
    logger.info(f"🚀 Starting structured wiki generation for {request.repo_url}")
    
    # Send status update
    await safe_websocket_send(websocket, {
        "status": "initializing", 
        "message": "Starting repository analysis..."
    })
    
    try:
        # Step 1: Get repository structure
        logger.info("📁 Step 1: Analyzing repository structure")
        file_tree, readme_content = await get_repository_structure(request)
        
        await safe_websocket_send(websocket, {
            "status": "structure_analyzed", 
            "message": f"Repository structure analyzed: {len(file_tree.split())} files found"
        })
        
        # Step 2: Initialize RAG for content retrieval
        logger.info("🔍 Step 2: Initializing content retrieval system")
        rag_instance = await initialize_rag_system(request)
        
        await safe_websocket_send(websocket, {
            "status": "rag_initializing", 
            "message": "Processing repository content..."
        })
        
        # Step 2b: Prepare the RAG retriever for the repository  
        logger.info("📚 Step 2b: Preparing RAG retriever for repository")
        prepare_rag_for_repository(rag_instance, request)
        
        await safe_websocket_send(websocket, {
            "status": "rag_initialized", 
            "message": "Content retrieval system ready"
        })
        
        # Step 3: Generate wiki structure using WIKI_STRUCTURE_ANALYSIS_PROMPT
        logger.info("🏗️ Step 3: Generating detailed wiki structure")
        
        await safe_websocket_send(websocket, {
            "status": "analyzing_structure", 
            "message": "Analyzing codebase to create detailed topic structure..."
        })
        
        # Use WIKI_STRUCTURE_ANALYSIS_PROMPT to get structured outline
        structure_prompt = WIKI_STRUCTURE_ANALYSIS_PROMPT.format(
            file_tree=file_tree[:15000],
            readme=readme_content[:5000]
        )
        
        logger.info(f"✅ Using WIKI_STRUCTURE_ANALYSIS_PROMPT for structured topic generation")
        logger.info(f"   - File tree length: {len(file_tree)} chars (using {len(file_tree[:15000])})")
        logger.info(f"   - README length: {len(readme_content)} chars (using {len(readme_content[:5000])})")
        logger.info(f"   - Structure prompt length: {len(structure_prompt)} characters")
        
        # Get model configuration
        model_config = get_model_config(request.provider, request.model)
        
        # Generate wiki structure
        wiki_structure = await generate_wiki_structure(structure_prompt, model_config, request)
        
        await safe_websocket_send(websocket, {
            "status": "structure_generated", 
            "message": "Topic structure created - generating detailed content for each section..."
        })
        
        # Step 4: Generate comprehensive content using the detailed architecture prompt
        logger.info("📝 Step 4: Generating comprehensive structured wiki content")
        
        # Get comprehensive repository context
        repo_context = await get_repository_context(rag_instance)
        
        # Create enhanced prompt that combines structure awareness with comprehensive content
        enhanced_prompt = create_enhanced_wiki_prompt(file_tree, readme_content, repo_context, wiki_structure)
        
        logger.info(f"✅ Using enhanced structured wiki prompt:")
        logger.info(f"   - Enhanced prompt length: {len(enhanced_prompt)} characters")
        logger.info(f"   - Includes topic structure and comprehensive context")
        
        await safe_websocket_send(websocket, {
            "status": "generating", 
            "message": "Generating comprehensive wiki with structured topics..."
        })
        
        # Step 5: Stream the generated content
        logger.info("📝 Step 5: Streaming structured wiki content")
        response_buffer = ""
        
        # Create a proper chat request for the existing stream system
        marked_prompt = f"[CLEAN_WIKI_PREFORMATTED_PROMPT]\n{enhanced_prompt}"
        
        chat_request = ChatCompletionRequest(
            repo_url=request.repo_url,
            messages=[ChatMessage(role="user", content=marked_prompt)],
            type=request.repo_type,
            provider=request.provider,
            model=request.model,
            token=request.token
        )
        
        # Stream the response using existing infrastructure
        async for chunk in stream_response(chat_request, rag_instance, model_config):
            if not is_websocket_connected(websocket):
                break
                
            try:
                chunk_data = json.loads(chunk)
                if "content" in chunk_data:
                    response_buffer += chunk_data["content"]
                
                # Forward the chunk to the frontend
                await websocket.send_text(chunk)
                
            except json.JSONDecodeError:
                # Handle plain text chunks
                response_buffer += chunk
                await websocket.send_text(json.dumps({"content": chunk}))
        
        # Step 6: Finalize and send completion
        logger.info("✅ Structured wiki generation completed successfully")
        await safe_websocket_send(websocket, {
            "status": "completed", 
            "message": f"Structured wiki generation completed: {len(response_buffer)} characters generated with detailed topics"
        })
        
    except Exception as e:
        logger.error(f"Error in structured wiki generation: {e}", exc_info=True)
        await safe_websocket_send(websocket, {
            "error": f"Structured wiki generation failed: {str(e)}"
        })
        raise


async def get_repository_structure(request: CleanWikiGenerationRequest) -> tuple[str, str]:
    """Get file tree and README content for the repository."""
    logger.info(f"📁 Getting repository structure for {request.repo_url}")
    
    if request.repo_type == "local":
        # For local repositories, use the existing local repo API logic
        if not request.local_path:
            raise ValueError("local_path is required for local repositories")
            
        # Use the same logic as in api.py get_local_repo_structure
        import os
        import fnmatch
        from .config import configs, DEFAULT_EXCLUDED_DIRS, DEFAULT_EXCLUDED_FILES
        
        file_tree_lines = []
        readme_content = ""
        path = request.local_path
        
        # Use same exclusion logic as api.py
        final_excluded_dirs = DEFAULT_EXCLUDED_DIRS.copy()
        final_excluded_files = DEFAULT_EXCLUDED_FILES.copy()
        
        file_filters_config = configs.get("file_filters", {})
        excluded_filename_patterns = file_filters_config.get("excluded_filename_patterns", [])
        repo_excluded_dirs = file_filters_config.get("excluded_dirs", [])
        final_excluded_dirs.extend(repo_excluded_dirs)
        
        # Normalize exclusion patterns
        normalized_excluded_dirs = []
        for p in final_excluded_dirs:
            if not p or not p.strip():
                continue
            normalized = p.strip()
            if normalized.startswith('./'):
                normalized = normalized[2:]
            normalized = normalized.rstrip('/')
            if normalized and normalized not in normalized_excluded_dirs:
                normalized_excluded_dirs.append(normalized)
        
        # Walk the directory
        for root, dirs, files in os.walk(path, topdown=True):
            current_dir_relative = os.path.relpath(root, path)
            normalized_current_dir = current_dir_relative.replace('\\', '/')
            
            if normalized_current_dir == '.':
                normalized_current_dir = ''
            
            # Check if current directory should be skipped
            should_skip_dir = False
            if normalized_current_dir:
                path_components = normalized_current_dir.split('/')
                for component in path_components:
                    if component in normalized_excluded_dirs:
                        should_skip_dir = True
                        break
            
            if should_skip_dir:
                dirs.clear()
                continue
            
            dirs[:] = [d for d in dirs if d not in normalized_excluded_dirs]
            
            # Process files
            for file in files:
                file_path = os.path.join(root, file)
                rel_dir = os.path.relpath(root, path)
                rel_file = os.path.join(rel_dir, file) if rel_dir != '.' else file
                normalized_rel_file = rel_file.replace('\\', '/')
                
                # Check exclusions
                filename_excluded = False
                for pattern in excluded_filename_patterns:
                    if fnmatch.fnmatch(file, pattern) or fnmatch.fnmatch(normalized_rel_file, pattern):
                        filename_excluded = True
                        break
                
                if filename_excluded:
                    continue
                
                if any(fnmatch.fnmatch(normalized_rel_file, pattern) for pattern in final_excluded_files):
                    continue
                
                file_tree_lines.append(normalized_rel_file)
                
                # Find README
                if file.lower() == 'readme.md' and not readme_content:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            readme_content = f.read()
                    except Exception as e:
                        logger.warning(f"Could not read README.md: {str(e)}")
        
        file_tree = '\n'.join(sorted(file_tree_lines))
        
    else:
        # For remote repositories (Bitbucket)
        # The RAG system will handle repository cloning and analysis
        # For now, return placeholder - this will be enhanced by RAG analysis
        file_tree = "Repository structure will be analyzed during content processing"
        readme_content = "README content will be retrieved during repository analysis"
    
    logger.info(f"📊 Repository structure: {len(file_tree.split())} files, README: {len(readme_content)} chars")
    return file_tree, readme_content


async def initialize_rag_system(request: CleanWikiGenerationRequest) -> RAG:
    """Initialize RAG system for the repository."""
    logger.info(f"🔍 Initializing RAG system for {request.repo_url}")
    
    # Create RAG instance with correct parameters
    rag_instance = RAG(
        provider=request.provider,
        model=request.model,
        use_s3=False  # We're not using S3 for this implementation
    )
    
    # The RAG system will handle repository processing internally
    # This includes cloning, file analysis, embedding generation, etc.
    logger.info("✅ RAG system initialized successfully")
    
    return rag_instance


def prepare_rag_for_repository(rag_instance: RAG, request: CleanWikiGenerationRequest):
    """Prepare the RAG system for the specific repository."""
    logger.info(f"📚 Preparing RAG retriever for repository: {request.repo_url}")
    
    try:
        # Use the existing prepare_retriever method to process the repository
        repo_path = request.local_path if request.repo_type == "local" else request.repo_url
        logger.info(f"🔧 Calling prepare_retriever with: path={repo_path}, type={request.repo_type}")
        
        rag_instance.prepare_retriever(
            repo_url_or_path=repo_path,
            type=request.repo_type,
            access_token=request.token
        )
        
        # Verify that retriever was actually created
        if rag_instance.retriever:
            logger.info("✅ RAG retriever prepared successfully")
            logger.info(f"📊 RAG has {len(rag_instance.transformed_docs)} transformed documents")
        else:
            logger.error("❌ RAG retriever preparation appeared to succeed but retriever is None")
        
    except Exception as e:
        logger.error(f"❌ Failed to prepare RAG retriever: {e}", exc_info=True)
        # Don't raise the exception - we can still generate a basic wiki without RAG
        logger.warning("⚠️ Continuing without RAG retrieval - wiki may be less comprehensive")


async def generate_wiki_structure(structure_prompt: str, model_config: dict, request: CleanWikiGenerationRequest) -> str:
    """Generate wiki structure using WIKI_STRUCTURE_ANALYSIS_PROMPT."""
    logger.info("🏗️ Generating wiki structure with analysis prompt")
    
    try:
        client = model_config["model_client"](**model_config.get("initialize_kwargs", {}))
        model_kwargs = model_config.get("model_kwargs", {}).copy()
        model_kwargs["stream"] = False  # Non-streaming for structure analysis
        model_kwargs["max_tokens"] = min(model_kwargs.get("max_tokens", 4000), 8000)  # Reasonable limit
        
        # Make API call for structure generation
        api_kwargs = client.convert_inputs_to_api_kwargs(
            input=structure_prompt, 
            model_kwargs=model_kwargs, 
            model_type=ModelType.LLM
        )
        
        logger.info(f"🔧 Making API call for structure generation with kwargs: {list(api_kwargs.keys())}")
        
        # Make API call for structure generation (OpenAI client now properly handles async)
        logger.info("🔄 Making async API call for structure generation")
        response = await client.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
        
        # Extract structure text from response using the client's parser
        logger.info(f"📋 Processing response of type: {type(response)}")
        
        # Use the client's built-in chat completion parser
        structure_text = client.chat_completion_parser(response)
        
        if not structure_text:
            # Try to extract from response object
            logger.warning(f"Unexpected response format: {type(response)}")
            logger.warning(f"Response attributes: {dir(response)}")
            structure_text = str(response)
            
        # Clean and validate structure text
        if not structure_text or not structure_text.strip():
            logger.warning("⚠️ Empty structure text extracted, using fallback")
            raise ValueError("Empty response received from structure generation")
        
        logger.info(f"✅ Generated wiki structure: {len(structure_text)} characters")
        logger.debug(f"Structure preview: {structure_text[:200]}...")
        
        return structure_text
        
    except Exception as e:
        logger.error(f"Failed to generate wiki structure: {e}", exc_info=True)
        logger.error(f"Error type: {type(e).__name__}")
        
        # Return repository-agnostic fallback structure (simplified)
        fallback_structure = """{
  "repository_analysis": {
    "type": "unknown",
    "primary_technologies": ["to_be_determined"],
    "architecture_pattern": "Repository analysis required for specific architecture details"
  },
  "wiki_structure": {
    "Getting Started": {
      "description": "Setup, installation, and initial configuration guide",
      "subtopics": [
        {"title": "Prerequisites and Dependencies", "description": "Required tools, runtime versions, and dependencies"},
        {"title": "Installation Instructions", "description": "Step-by-step installation and setup process"},
        {"title": "Initial Configuration", "description": "Basic configuration required to run the application"},
        {"title": "Quick Start Examples", "description": "Simple examples to verify successful installation"}
      ]
    },
    "Core Concepts": {
      "description": "Fundamental architecture, design patterns, and system principles",
      "subtopics": [
        {"title": "System Architecture", "description": "High-level system design and component relationships"},
        {"title": "Design Patterns", "description": "Key architectural and design patterns used"},
        {"title": "Data Models and Structures", "description": "Core data models and their relationships"},
        {"title": "Business Logic Flow", "description": "Main application workflows and processes"}
      ]
    },
    "API Development": {
      "description": "API endpoints, services, and integration patterns",
      "subtopics": [
        {"title": "API Structure and Endpoints", "description": "Available endpoints and their organization"},
        {"title": "Request and Response Formats", "description": "Data formats, schemas, and validation"},
        {"title": "Authentication and Authorization", "description": "Security implementation and access control"},
        {"title": "Integration Patterns", "description": "How to integrate with external services"}
      ]
    },
    "Infrastructure and Configuration": {
      "description": "Deployment, environment setup, and system configuration",
      "subtopics": [
        {"title": "Environment Configuration", "description": "Environment variables and configuration management"},
        {"title": "Build and Deployment", "description": "Build processes and deployment strategies"},
        {"title": "Database Setup", "description": "Database configuration and migration processes"},
        {"title": "Monitoring and Logging", "description": "System monitoring, logging, and observability"}
      ]
    },
    "Testing and Debugging": {
      "description": "Testing strategies, debugging techniques, and troubleshooting",
      "subtopics": [
        {"title": "Testing Strategy", "description": "Testing frameworks, approaches, and best practices"},
        {"title": "Running Tests", "description": "How to execute different types of tests"},
        {"title": "Debugging Techniques", "description": "Debugging tools and troubleshooting approaches"},
        {"title": "Common Issues", "description": "Frequently encountered problems and solutions"}
      ]
    },
    "Utilities and Helpers": {
      "description": "Support functions, utilities, and development tools",
      "subtopics": [
        {"title": "Utility Functions", "description": "Common utility functions and helper methods"},
        {"title": "Development Tools", "description": "Tools and scripts for development workflow"},
        {"title": "Code Generation", "description": "Automated code generation and scaffolding tools"},
        {"title": "Performance Utilities", "description": "Performance monitoring and optimization tools"}
      ]
    }
  }
}"""
        
        logger.info("✅ Using fallback wiki structure")
        return fallback_structure


def create_enhanced_wiki_prompt(file_tree: str, readme_content: str, repo_context: str, wiki_structure: str) -> str:
    """Create enhanced hierarchical prompt that uses dynamic structure with sub-topics."""
    
    logger.info("🏗️ Creating enhanced hierarchical wiki prompt with dynamic sub-topics")
    
    # Try to parse the wiki structure to validate it's hierarchical
    parsed_structure = None
    try:
        # Handle different possible formats of wiki_structure
        if wiki_structure.strip().startswith('{'):
            # It's JSON - try to parse it
            import json
            parsed_structure = json.loads(wiki_structure)
            logger.info("✅ Successfully parsed JSON wiki structure")
            
            # Log the structure to verify it has the hierarchical format
            if 'wiki_structure' in parsed_structure:
                topics = list(parsed_structure['wiki_structure'].keys())
                logger.info(f"📋 Found hierarchical structure with topics: {topics}")
                
                # Count total sub-topics
                total_subtopics = 0
                for topic, topic_data in parsed_structure['wiki_structure'].items():
                    if isinstance(topic_data, dict) and 'subtopics' in topic_data:
                        subtopic_count = len(topic_data['subtopics'])
                        total_subtopics += subtopic_count
                        logger.info(f"   - {topic}: {subtopic_count} sub-topics")
                
                logger.info(f"📊 Total sub-topics found: {total_subtopics}")
            else:
                logger.warning("⚠️ Wiki structure doesn't have expected 'wiki_structure' key")
        else:
            # It's text format - use as-is
            logger.info("📝 Wiki structure is in text format, using directly")
            
    except json.JSONDecodeError as e:
        logger.warning(f"⚠️ Could not parse wiki structure as JSON: {e}")
        logger.info("📝 Using wiki structure as descriptive text")
    except Exception as e:
        logger.warning(f"⚠️ Error processing wiki structure: {e}")
    
    # Use the new hierarchical prompt that properly handles dynamic sub-topics
    enhanced_prompt = HIERARCHICAL_WIKI_GENERATION_PROMPT.format(
        wiki_structure=wiki_structure,
        context=repo_context,
        file_tree=file_tree,
        readme=readme_content
    )
    
    logger.info(f"✅ Created hierarchical wiki prompt:")
    logger.info(f"   - Uses HIERARCHICAL_WIKI_GENERATION_PROMPT template")
    logger.info(f"   - Wiki structure length: {len(wiki_structure)} chars")
    logger.info(f"   - Repo context length: {len(repo_context)} chars")
    logger.info(f"   - Final prompt length: {len(enhanced_prompt)} chars")
    
    return enhanced_prompt


async def get_repository_context(rag_instance: RAG) -> str:
    """Get comprehensive repository context from RAG system."""
    logger.info("📝 Retrieving comprehensive repository context")
    
    # Check if RAG retriever is properly initialized
    if not rag_instance.retriever:
        logger.error("❌ RAG retriever is not initialized - cannot retrieve context")
        return "Repository context not available: RAG retriever not prepared"
    
    logger.info(f"✅ RAG retriever found: {type(rag_instance.retriever)}")
    
    try:
        # Use broad queries to get comprehensive coverage
        broad_queries = [
            "system architecture components configuration",
            "main application logic business functions", 
            "API endpoints routes handlers",
            "database models data structures",
            "user interface components views",
            "authentication security middleware",
            "testing deployment configuration"
        ]
        
        all_docs = []
        seen_sources = set()
        
        for query in broad_queries:
            try:
                # Use the correct RAG.call() method instead of retrieve()
                result = rag_instance.call(query)
                
                if result is not None and isinstance(result, tuple) and len(result) >= 2:
                    retrieved_docs, _ = result
                    
                    if retrieved_docs:
                        for doc in retrieved_docs:
                            # Check if doc has source information
                            source = getattr(doc, 'source', 'unknown')
                            if hasattr(doc, 'metadata') and doc.metadata:
                                source = doc.metadata.get('source', source)
                            
                            if source not in seen_sources:
                                all_docs.append(doc)
                                seen_sources.add(source)
                    else:
                        logger.debug(f"Query '{query}' returned no documents")
                else:
                    logger.debug(f"Query '{query}' returned invalid result format")
                    
            except Exception as e:
                logger.warning(f"Query '{query}' failed: {e}")
                continue
        
        # Combine all retrieved context
        context_text = "\n\n".join([getattr(doc, 'text', getattr(doc, 'content', '')) for doc in all_docs]) if all_docs else ""
        
        logger.info(f"📊 Retrieved context: {len(context_text)} chars from {len(seen_sources)} sources")
        return context_text[:50000]  # Limit to 50K chars
        
    except Exception as e:
        logger.error(f"Failed to retrieve repository context: {e}")
        return "Repository context not available due to retrieval error"
