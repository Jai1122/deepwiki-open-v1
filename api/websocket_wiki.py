import logging
from typing import List, Optional
from urllib.parse import unquote

import google.generativeai as genai
from adalflow.components.model_client.ollama_client import OllamaClient
from adalflow.core.types import ModelType
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from api.config import get_model_config, configs, OPENROUTER_API_KEY, OPENAI_API_KEY
from api.data_pipeline import count_tokens, get_file_content
from api.openai_client import OpenAIClient
from api.openrouter_client import OpenRouterClient
from api.azureai_client import AzureAIClient
from api.vllm_client import VLLMClient
from api.websocket_helpers import (
    prepare_rag_retriever,
    get_rag_context,
    get_system_prompt,
    handle_google_provider,
    handle_ollama_provider,
    handle_openrouter_provider,
    handle_openai_provider,
    handle_azure_provider,
    handle_vllm_provider,
)

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

async def handle_websocket_chat(websocket: WebSocket):
    """
    Handle WebSocket connection for chat completions.
    This replaces the HTTP streaming endpoint with a WebSocket connection.
    """
    await websocket.accept()

    try:
        # 1. Receive and parse the request data
        request_data = await websocket.receive_json()
        request = ChatCompletionRequest(**request_data)

        # 2. Check for large input
        input_too_large = False
        if request.messages and len(request.messages) > 0:
            last_message = request.messages[-1]
            if hasattr(last_message, 'content') and last_message.content:
                tokens = count_tokens(last_message.content, request.provider == "ollama")
                logger.info(f"Request size: {tokens} tokens")
                if tokens > 8000:
                    logger.warning(f"Request exceeds recommended token limit ({tokens} > 7500)")
                    input_too_large = True

        # 3. Prepare the RAG retriever
        try:
            request_rag = prepare_rag_retriever(
                request.repo_url,
                request.type,
                request.token,
                request.excluded_dirs,
                request.excluded_files,
                request.included_dirs,
                request.included_files,
                request.provider,
                request.model,
            )
        except Exception as e:
            await websocket.send_text(f"Error preparing retriever: {str(e)}")
            return

        # 4. Validate request
        if not request.messages or len(request.messages) == 0:
            await websocket.send_text("Error: No messages provided")
            return
        last_message = request.messages[-1]
        if last_message.role != "user":
            await websocket.send_text("Error: Last message must be from the user")
            return

        # 5. Process conversation history
        for i in range(0, len(request.messages) - 1, 2):
            if i + 1 < len(request.messages):
                user_msg = request.messages[i]
                assistant_msg = request.messages[i + 1]
                if user_msg.role == "user" and assistant_msg.role == "assistant":
                    request_rag.memory.add_dialog_turn(user_query=user_msg.content, assistant_response=assistant_msg.content)

        # 6. Handle Deep Research requests
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

        # 7. Get query and context
        query = last_message.content
        context_text = ""
        if not input_too_large:
            context_text = get_rag_context(request_rag, query, request.filePath, request.language)

        # 8. Get repository and language information
        repo_url = request.repo_url
        repo_name = repo_url.split("/")[-1] if "/" in repo_url else repo_url
        repo_type = request.type
        language_code = request.language or configs["lang_config"]["default"]
        supported_langs = configs["lang_config"]["supported_languages"]
        language_name = supported_langs.get(language_code, "English")

        # 9. Get system prompt
        system_prompt = get_system_prompt(repo_type, repo_url, repo_name, language_name, is_deep_research, research_iteration)

        # 10. Fetch file content
        file_content = ""
        if request.filePath:
            try:
                file_content = get_file_content(request.repo_url, request.filePath, request.type, request.token)
                logger.info(f"Successfully retrieved content for file: {request.filePath}")
            except Exception as e:
                logger.error(f"Error retrieving file content: {str(e)}")

        # 11. Format conversation history
        conversation_history = ""
        for turn_id, turn in request_rag.memory().items():
            if not isinstance(turn_id, int) and hasattr(turn, 'user_query') and hasattr(turn, 'assistant_response'):
                conversation_history += f"<turn>\\n<user>{turn.user_query.query_str}</user>\\n<assistant>{turn.assistant_response.response_str}</assistant>\\n</turn>\\n"

        # 12. Create the final prompt
        prompt = f"/no_think {system_prompt}\\n\\n"
        if conversation_history:
            prompt += f"<conversation_history>\\n{conversation_history}</conversation_history>\\n\\n"
        if file_content:
            prompt += f"<currentFileContent path=\\\"{request.filePath}\\\">\\n{file_content}\\n</currentFileContent>\\n\\n"
        if context_text.strip():
            prompt += f"<START_OF_CONTEXT>\\n{context_text}\\n</END_OF_CONTEXT>\\n\\n"
        else:
            logger.info("No context available from RAG")
            prompt += "<note>Answering without retrieval augmentation.</note>\\n\\n"
        prompt += f"<query>\\n{query}\\n</query>\\n\\nAssistant: "

        # 13. Get model configuration and handle provider-specific logic
        model_config = get_model_config(request.provider, request.model)["model_kwargs"]
        try:
            if request.provider == "google":
                model = genai.GenerativeModel(model_name=model_config["model"], generation_config={"temperature": model_config["temperature"], "top_p": model_config["top_p"], "top_k": model_config["top_k"]})
                await handle_google_provider(model, prompt, websocket)
            else:
                provider_handlers = {"ollama": handle_ollama_provider, "openrouter": handle_openrouter_provider, "openai": handle_openai_provider, "azure": handle_azure_provider, "vllm": handle_vllm_provider}
                client_classes = {"ollama": OllamaClient, "openrouter": OpenRouterClient, "openai": OpenAIClient, "azure": AzureAIClient, "vllm": VLLMClient}
                model = client_classes[request.provider]()
                model_kwargs = {"model": request.model, "stream": True}
                if request.provider == "ollama":
                    prompt += " /no_think"
                    model_kwargs["options"] = {"temperature": model_config["temperature"], "top_p": model_config["top_p"], "num_ctx": model_config["num_ctx"]}
                else:
                    model_kwargs["temperature"] = model_config.get("temperature", 0.7)
                    if "top_p" in model_config:
                        model_kwargs["top_p"] = model_config["top_p"]

                api_kwargs = model.convert_inputs_to_api_kwargs(input=prompt, model_kwargs=model_kwargs, model_type=ModelType.LLM)
                await provider_handlers[request.provider](model, api_kwargs, websocket)

        except Exception as e_outer:
            logger.error(f"Error in streaming response: {str(e_outer)}")
            error_message = str(e_outer)
            if "maximum context length" in error_message or "token limit" in error_message or "too many tokens" in error_message:
                logger.warning("Token limit exceeded, retrying without context")
                try:
                    simplified_prompt = f"/no_think {system_prompt}\\n\\n"
                    if conversation_history:
                        simplified_prompt += f"<conversation_history>\\n{conversation_history}</conversation_history>\\n\\n"
                    if request.filePath and file_content:
                        simplified_prompt += f"<currentFileContent path=\\\"{request.filePath}\\\">\\n{file_content}\\n</currentFileContent>\\n\\n"
                    simplified_prompt += "<note>Answering without retrieval augmentation due to input size constraints.</note>\\n\\n"
                    simplified_prompt += f"<query>\\n{query}\\n</query>\\n\\nAssistant: "
                    if request.provider == "google":
                        await handle_google_provider(model, simplified_prompt, websocket)
                    else:
                        await provider_handlers[request.provider](model, api_kwargs, websocket)
                except Exception as e2:
                    logger.error(f"Error in fallback streaming response: {str(e2)}")
                    await websocket.send_text(f"\\nI apologize, but your request is too large for me to process. Please try a shorter query or break it into smaller parts.")
            else:
                await websocket.send_text(f"\\nError: {error_message}")

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error in WebSocket handler: {str(e)}")
        try:
            await websocket.send_text(f"Error: {str(e)}")
        except:
            pass
    finally:
        try:
            await websocket.close()
        except:
            pass
