import logging
import os
from typing import List, Optional, Dict, Any
from urllib.parse import unquote

import google.generativeai as genai
from adalflow.components.model_client.ollama_client import OllamaClient
from adalflow.core.types import ModelType
from fastapi import WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel, Field

from api.config import get_model_config, configs, OPENROUTER_API_KEY, OPENAI_API_KEY
from api.data_pipeline import count_tokens, get_file_content
from api.openai_client import OpenAIClient
from api.openrouter_client import OpenRouterClient
from api.azureai_client import AzureAIClient
from api.rag import RAG
from api.context_manager import ContextManager
import inspect

# Configure logging
from api.logging_config import setup_logging

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
    provider: str = Field("google", description="Model provider (google, openai, openrouter, ollama, azure, vllm)")
    model: Optional[str] = Field(None, description="Model name for the specified provider")

    language: Optional[str] = Field("en", description="Language for content generation (e.g., 'en', 'ja', 'zh', 'es', 'kr', 'vi')")
    excluded_dirs: Optional[str] = Field(None, description="Comma-separated list of directories to exclude from processing")
    excluded_files: Optional[str] = Field(None, description="Comma-separated list of file patterns to exclude from processing")
    included_dirs: Optional[str] = Field(None, description="Comma-separated list of directories to include exclusively")
    included_files: Optional[str] = Field(None, description="Comma-separated list of file patterns to include exclusively")

async def prepare_rag(request: ChatCompletionRequest) -> RAG:
    """
    Prepare the RAG instance for the request.
    """
    logger.info("Preparing RAG instance...")
    request_rag = RAG(provider=request.provider, model=request.model)

    # Extract custom file filter parameters if provided
    excluded_dirs = None
    excluded_files = None
    included_dirs = None
    included_files = None

    if request.excluded_dirs:
        excluded_dirs = [unquote(dir_path) for dir_path in request.excluded_dirs.split('\n') if dir_path.strip()]
        logger.info(f"Using custom excluded directories: {excluded_dirs}")
    if request.excluded_files:
        excluded_files = [unquote(file_pattern) for file_pattern in request.excluded_files.split('\n') if file_pattern.strip()]
        logger.info(f"Using custom excluded files: {excluded_files}")
    if request.included_dirs:
        included_dirs = [unquote(dir_path) for dir_path in request.included_dirs.split('\n') if dir_path.strip()]
        logger.info(f"Using custom included directories: {included_dirs}")
    if request.included_files:
        included_files = [unquote(file_pattern) for file_pattern in request.included_files.split('\n') if file_pattern.strip()]
        logger.info(f"Using custom included files: {included_files}")

    from api.config import is_ollama_embedder
    is_ollama = is_ollama_embedder()

    request_rag.prepare_retriever(
        request.repo_url,
        request.type,
        request.token,
        is_ollama_embedder=is_ollama,
        excluded_dirs=excluded_dirs,
        excluded_files=excluded_files,
        included_dirs=included_dirs,
        included_files=included_files
    )
    logger.info(f"Retriever prepared for {request.repo_url}")
    return request_rag

async def handle_websocket_chat(websocket: WebSocket):
    """
    Handle WebSocket connection for chat completions.
    This replaces the HTTP streaming endpoint with a WebSocket connection.
    """
    await websocket.accept()
    try:
        request_data = await websocket.receive_json()
        request = ChatCompletionRequest(**request_data)

        input_too_large = False
        if request.messages and len(request.messages) > 0:
            last_message = request.messages[-1]
            if hasattr(last_message, 'content') and last_message.content:
                tokens = count_tokens(last_message.content, request.provider == "ollama")
                logger.info(f"Request size: {tokens} tokens")
                if tokens > 8000:
                    logger.warning(f"Request exceeds recommended token limit ({tokens} > 7500)")
                    input_too_large = True

        request_rag = await prepare_rag(request)

        if not request.messages or len(request.messages) == 0:
            await websocket.send_text("Error: No messages provided")
            return

        last_message = request.messages[-1]
        if last_message.role != "user":
            await websocket.send_text("Error: Last message must be from the user")
            return

        for i in range(0, len(request.messages) - 1, 2):
            if i + 1 < len(request.messages):
                user_msg = request.messages[i]
                assistant_msg = request.messages[i+1]
                if user_msg.role == "user" and assistant_msg.role == "assistant":
                    request_rag.memory.add_dialog_turn(user_query=user_msg.content, assistant_response=assistant_msg.content)

        is_deep_research = False
        research_iteration = 1
        for msg in request.messages:
            if hasattr(msg, 'content') and msg.content and "[DEEP RESEARCH]" in msg.content:
                is_deep_research = True
                if msg == request.messages[-1]:
                    msg.content = msg.content.replace("[DEEP RESEARCH]", "").strip()

        if is_deep_research:
            research_iteration = sum(1 for msg in request.messages if msg.role == 'assistant') + 1
            logger.info(f"Deep Research request detected - iteration {research_iteration}")
            if "continue" in last_message.content.lower() and "research" in last_message.content.lower():
                original_topic = None
                for msg in request.messages:
                    if msg.role == "user" and "continue" not in msg.content.lower():
                        original_topic = msg.content.replace("[DEEP RESEARCH]", "").strip()
                        logger.info(f"Found original research topic: {original_topic}")
                        break
                if original_topic:
                    last_message.content = original_topic
                    logger.info(f"Using original topic for research: {original_topic}")

        # Get the query from the last message
        query = last_message.content
        prompt = await build_prompt_for_request(request, request_rag, is_deep_research, research_iteration, query, input_too_large)
        await call_model_and_stream_response(websocket, request, prompt)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error in WebSocket handler: {str(e)}")
        if not websocket.client_state.name == 'DISCONNECTED':
            try:
                await websocket.send_text(f"Error: {str(e)}")
            except:
                pass
    finally:
        if not websocket.client_state.name == 'DISCONNECTED':
            await websocket.close()

async def build_prompt_for_request(request: ChatCompletionRequest, request_rag: RAG, is_deep_research: bool, research_iteration: int, query: str, input_too_large: bool) -> str:
    """
    Build the prompt for the request.
    """
    logger.info("Building prompt...")
    repo_url = request.repo_url
    repo_name = repo_url.split("/")[-1] if "/" in repo_url else repo_url
    repo_type = request.type
    language_code = request.language or configs["lang_config"]["default"]
    supported_langs = configs["lang_config"]["supported_languages"]
    language_name = supported_langs.get(language_code, "English")

    system_prompt = ""
    # ... (system prompt logic remains the same)

    file_content = ""
    if request.filePath:
        try:
            file_content = get_file_content(request.repo_url, request.filePath, request.type, request.token)
            logger.info(f"Successfully retrieved content for file: {request.filePath}")
        except Exception as e:
            logger.error(f"Error retrieving file content: {str(e)}")

    conversation_history = ""
    for turn_id, turn in request_rag.memory().items():
        if not isinstance(turn_id, int) and hasattr(turn, 'user_query') and hasattr(turn, 'assistant_response'):
            conversation_history += f"<turn>\n<user>{turn.user_query.query_str}</user>\n<assistant>{turn.assistant_response.response_str}</assistant>\n</turn>\n"

    retrieved_documents = None
    if not input_too_large:
        try:
            rag_query = query
            if request.filePath:
                rag_query = f"Contexts related to {request.filePath}"

            retrieved_docs_result = request_rag(rag_query, language=request.language)
            if retrieved_docs_result and retrieved_docs_result[0].documents:
                retrieved_documents = retrieved_docs_result[0].documents
        except Exception as e:
            logger.error(f"Error during RAG retrieval: {str(e)}")

    model_config_dict = get_model_config(request.provider, request.model)
    resolved_model_kwargs = model_config_dict.get("model_kwargs", {})
    model_max_tokens = resolved_model_kwargs.get("max_context_tokens", 8192)

    context_manager = ContextManager(model_provider=request.provider)
    prompt = context_manager.build_prompt(
        system_prompt=system_prompt,
        query=query,
        conversation_history=conversation_history,
        file_content=file_content,
        file_path=request.filePath,
        retrieved_documents=retrieved_documents,
        model_max_tokens=model_max_tokens
    )
    logger.info(f"Final prompt tokens from websocket: {count_tokens(prompt, request.provider == 'ollama')}")
    return prompt

async def call_model_and_stream_response(websocket: WebSocket, request: ChatCompletionRequest, prompt: str):
    """
    Call the model and stream the response.
    """
    logger.info("Calling model and streaming response...")
    model_config_dict = get_model_config(request.provider, request.model)

    resolved_client_class = model_config_dict["model_client"]
    resolved_model_kwargs_from_config = model_config_dict["model_kwargs"]

    llm_client_instance: Any = None
    api_input_construction_kwargs = resolved_model_kwargs_from_config.copy()
    api_input_construction_kwargs["stream"] = True
    api_input_construction_kwargs.pop('max_context_tokens', None)

    # Instantiate clients based on provider
    if request.provider == "vllm":
        logger.info(f"Using vLLM with model: {api_input_construction_kwargs.get('model')}")
        vllm_base_url = os.environ.get('VLLM_API_BASE_URL')
        vllm_api_key = os.environ.get('VLLM_API_KEY')
        if not vllm_base_url:
            # Send error to client and close WebSocket
            error_msg = "VLLM_API_BASE_URL is not set for vLLM provider."
            logger.error(error_msg)
            await websocket.send_text(f"Error: {error_msg}")
            await websocket.close()
            return
        # resolved_client_class is OpenAIClient
        llm_client_instance = resolved_client_class(base_url=vllm_base_url, api_key=vllm_api_key)

    elif request.provider == "ollama":
        logger.info(f"Using Ollama with model: {api_input_construction_kwargs.get('model')}")
        if not prompt.endswith(" /no_think"): # Avoid adding it multiple times
            prompt += " /no_think" # Ollama specific prompt adjustment
        # resolved_client_class is OllamaClient
        llm_client_instance = resolved_client_class() # OllamaClient picks up OLLAMA_HOST from env
                                                   # and expects headers/options in model_kwargs for convert_inputs...
                                                   # api_input_construction_kwargs for ollama is already correctly structured by get_model_config

    elif request.provider == "openrouter":
        logger.info(f"Using OpenRouter with model: {api_input_construction_kwargs.get('model')}")
        # resolved_client_class is OpenRouterClient
        llm_client_instance = resolved_client_class(api_key=OPENROUTER_API_KEY) # Pass key if constructor takes it
        if not OPENROUTER_API_KEY:
             logger.warning("OPENROUTER_API_KEY not configured, OpenRouter call might fail.")

    elif request.provider == "openai":
        logger.info(f"Using OpenAI protocol with model: {api_input_construction_kwargs.get('model')}")
        # resolved_client_class is OpenAIClient
        llm_client_instance = resolved_client_class(api_key=OPENAI_API_KEY) # Pass key if constructor takes it
        if not OPENAI_API_KEY:
             logger.warning("OPENAI_API_KEY not configured, OpenAI call might fail.")

    elif request.provider == "azure":
        logger.info(f"Using Azure AI with model: {api_input_construction_kwargs.get('model')}")
        # resolved_client_class is AzureAIClient
        llm_client_instance = resolved_client_class() # AzureAIClient reads its specific env vars

    elif request.provider == "google":
        logger.info(f"Using Google Gemini with model: {api_input_construction_kwargs.get('model')}")
        # Google's genai client is instantiated and used differently
        llm_client_instance = genai.GenerativeModel(
            model_name=api_input_construction_kwargs.get("model"),
            generation_config={
                "temperature": api_input_construction_kwargs.get("temperature"),
                "top_p": api_input_construction_kwargs.get("top_p"),
                "top_k": api_input_construction_kwargs.get("top_k")
            }
        )
        # api_kwargs_for_call is not used for Google in the same way
    else:
        logger.error(f"Unknown provider in websocket: {request.provider}")
        await websocket.send_text(f"Error: Unknown provider {request.provider}")
        await websocket.close()
        return

    try:
        if request.provider in ["vllm", "ollama", "openrouter", "openai", "azure"]:
            api_kwargs_for_call = llm_client_instance.convert_inputs_to_api_kwargs(
                input=prompt,
                model_kwargs=api_input_construction_kwargs,
                model_type=ModelType.LLM
            )
            response_stream = await llm_client_instance.acall(api_kwargs=api_kwargs_for_call, model_type=ModelType.LLM)

            if request.provider in ["vllm", "openai", "azure", "openrouter"]:
                async for chunk in response_stream:
                    choices = getattr(chunk, "choices", [])
                    if len(choices) > 0:
                        delta = getattr(choices[0], "delta", None)
                        if delta is not None:
                            text_content = getattr(delta, "content", None)
                            if text_content is not None:
                                await websocket.send_text(text_content)
            elif request.provider == "ollama":
                async for chunk in response_stream:
                    text = getattr(chunk, 'response', None)
                    if text is None:
                        message_attr = getattr(chunk, 'message', None)
                        if message_attr:
                            text = getattr(message_attr, 'content', None)
                        else:
                            text = getattr(chunk, 'text', None) or \
                                   (str(chunk) if not (hasattr(chunk, 'model') and hasattr(chunk, 'created_at')) else None)

                    if text:
                       text = text.replace('<think>', '').replace('</think>', '')
                       await websocket.send_text(text)
        elif request.provider == "google":
            response_stream = llm_client_instance.generate_content(prompt, stream=True)
            for chunk in response_stream:
                if hasattr(chunk, 'text'):
                    await websocket.send_text(chunk.text)

    except Exception as e_outer:
        logger.error(f"Error in streaming response for provider {request.provider}: {str(e_outer)}")
        if not websocket.client_state.name == 'DISCONNECTED':
            await websocket.send_text(f"\nError: {str(e_outer)}")

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error in WebSocket handler: {str(e)}")
        if not websocket.client_state.name == 'DISCONNECTED':
            try:
                await websocket.send_text(f"Error: {str(e)}")
            except:
                pass
