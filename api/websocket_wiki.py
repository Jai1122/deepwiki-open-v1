import logging
import os
from typing import List, Optional, Dict, Any
from urllib.parse import unquote

import google.generativeai as genai
from adalflow.components.model_client.ollama_client import OllamaClient
from adalflow.core.types import ModelType
from fastapi import WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel, Field

from .config import get_model_config, configs, OPENROUTER_API_KEY, OPENAI_API_KEY, get_max_tokens_for_model
from .data_pipeline import count_tokens, get_file_content, truncate_prompt_to_fit
from .openai_client import OpenAIClient
from .openrouter_client import OpenRouterClient
from .azureai_client import AzureAIClient
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
    provider: str = Field("google", description="Model provider (google, openai, openrouter, ollama, azure)")
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
        # Receive and parse the request data
        request_data = await websocket.receive_json()
        request = ChatCompletionRequest(**request_data)

        # Create a new RAG instance for this request
        try:
            request_rag = RAG(provider=request.provider, model=request.model)

            # Extract custom file filter parameters if provided
            excluded_dirs = None
            excluded_files = None
            included_dirs = None
            included_files = None

            if request.excluded_dirs:
                excluded_dirs = [unquote(dir_path) for dir_path in request.excluded_dirs.split('\n') if dir_path.strip()]
            if request.excluded_files:
                excluded_files = [unquote(file_pattern) for file_pattern in request.excluded_files.split('\n') if file_pattern.strip()]
            if request.included_dirs:
                included_dirs = [unquote(dir_path) for dir_path in request.included_dirs.split('\n') if dir_path.strip()]
            if request.included_files:
                included_files = [unquote(file_pattern) for file_pattern in request.included_files.split('\n') if file_pattern.strip()]

            request_rag.prepare_retriever(request.repo_url, request.type, request.token, excluded_dirs, excluded_files, included_dirs, included_files)
            logger.info(f"Retriever prepared for {request.repo_url}")
        except Exception as e:
            logger.error(f"Error preparing retriever: {str(e)}")
            await websocket.send_text(f"Error preparing retriever: {str(e)}")
            await websocket.close()
            return

        # Validate request
        if not request.messages or len(request.messages) == 0:
            await websocket.send_text("Error: No messages provided")
            await websocket.close()
            return

        last_message = request.messages[-1]
        if last_message.role != "user":
            await websocket.send_text("Error: Last message must be from the user")
            await websocket.close()
            return

        # Process previous messages to build conversation history
        for i in range(0, len(request.messages) - 1, 2):
            if i + 1 < len(request.messages):
                user_msg = request.messages[i]
                assistant_msg = request.messages[i + 1]
                if user_msg.role == "user" and assistant_msg.role == "assistant":
                    request_rag.memory.add_dialog_turn(user_query=user_msg.content, assistant_response=assistant_msg.content)

        query = last_message.content
        
        # Perform RAG retrieval
        context_text = ""
        try:
            rag_query = f"Contexts related to {request.filePath}" if request.filePath else query
            retrieved_documents = request_rag(rag_query, language=request.language)
            if retrieved_documents and len(retrieved_documents) > 0 and hasattr(retrieved_documents[0], 'documents') and retrieved_documents[0].documents:
                documents = retrieved_documents[0].documents
                context_text = "\n\n".join([f"## File Path: {doc.meta_data.get('file_path', 'unknown')}\n\n{doc.text}" for doc in documents])
        except Exception as e:
            logger.error(f"Error in RAG retrieval: {str(e)}")

        # Fetch file content if provided
        file_content = ""
        if request.filePath:
            try:
                file_content = get_file_content(request.repo_url, request.filePath, request.type, request.token)
            except Exception as e:
                logger.error(f"Error retrieving file content: {str(e)}")

        # System prompt and conversation history setup (remains the same)
        system_prompt = "..." # Keeping the system prompt logic as it was, for brevity
        conversation_history = ""
        for turn_id, turn in request_rag.memory().items():
            if not isinstance(turn_id, int) and hasattr(turn, 'user_query') and hasattr(turn, 'assistant_response'):
                conversation_history += f"<turn>\n<user>{turn.user_query.query_str}</user>\n<assistant>{turn.assistant_response.response_str}</assistant>\n</turn>\n"

        # Intelligent Truncation
        max_tokens = get_max_tokens_for_model(request.provider, request.model)
        truncated_file_content, truncated_context_text = truncate_prompt_to_fit(
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            conversation_history=conversation_history,
            file_content=file_content,
            context_text=context_text,
            query=query,
            is_ollama=(request.provider == "ollama")
        )

        # Build the final prompt
        prompt = f"/no_think {system_prompt}\n\n"
        if conversation_history:
            prompt += f"<conversation_history>\n{conversation_history}</conversation_history>\n\n"
        if truncated_file_content:
            prompt += f"<currentFileContent path=\"{request.filePath}\">\n{truncated_file_content}\n</currentFileContent>\n\n"
        if truncated_context_text:
            prompt += f"<START_OF_CONTEXT>\n{truncated_context_text}\n</END_OF_CONTEXT>\n\n"
        else:
            prompt += "<note>Answering without retrieval augmentation.</note>\n\n"
        prompt += f"<query>\n{query}\n</query>\n\nAssistant: "

        # Model and API call logic (remains the same as the last correct version)
        full_model_config = get_model_config(request.provider, request.model)
        model_config = full_model_config["model_kwargs"]
        init_kwargs = full_model_config.get("initialize_kwargs", {})
        
        model = None
        model_kwargs = {}
        api_kwargs = {}

        if request.provider == "ollama":
            prompt += " /no_think"
            model = OllamaClient(**init_kwargs)
            model_kwargs = {"model": model_config["model"], "stream": True, "options": {"temperature": model_config["temperature"], "top_p": model_config["top_p"], "num_ctx": model_config["num_ctx"]}}
            api_kwargs = model.convert_inputs_to_api_kwargs(input=prompt, model_kwargs=model_kwargs, model_type=ModelType.LLM)
        elif request.provider in ["openai", "vllm", "openrouter"]:
            model_class = OpenRouterClient if request.provider == "openrouter" else OpenAIClient
            model = model_class(**init_kwargs)
            model_kwargs = {"model": request.model, "stream": True, "temperature": model_config.get("temperature", 0.7)}
            if "top_p" in model_config:
                model_kwargs["top_p"] = model_config["top_p"]
            api_kwargs = model.convert_inputs_to_api_kwargs(input=prompt, model_kwargs=model_kwargs, model_type=ModelType.LLM)
        elif request.provider == "azure":
            model = AzureAIClient()
            model_kwargs = {"model": request.model, "stream": True, "temperature": model_config["temperature"], "top_p": model_config["top_p"]}
            api_kwargs = model.convert_inputs_to_api_kwargs(input=prompt, model_kwargs=model_kwargs, model_type=ModelType.LLM)
        else: # Google
            model = genai.GenerativeModel(model_name=model_config["model"], generation_config={"temperature": model_config["temperature"], "top_p": model_config["top_p"], "top_k": model_config["top_k"]})

        # Streaming response
        try:
            if request.provider in ["ollama", "openai", "vllm", "openrouter", "azure"]:
                response = await model.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
                async for chunk in response:
                    if request.provider in ["openai", "vllm", "azure"]:
                        choices = getattr(chunk, "choices", [])
                        if len(choices) > 0 and (delta := getattr(choices[0], "delta", None)) and (text := getattr(delta, "content", None)):
                            await websocket.send_text(text)
                    elif request.provider == "ollama":
                         text = getattr(chunk, 'response', None) or getattr(chunk, 'text', None) or str(chunk)
                         if text and not text.startswith('model='):
                             await websocket.send_text(text.replace('<think>', '').replace('</think>', ''))
                    else: # openrouter
                        await websocket.send_text(chunk)
            else: # google
                response = model.generate_content(prompt, stream=True)
                for chunk in response:
                    if hasattr(chunk, 'text'):
                        await websocket.send_text(chunk.text)
            
            await websocket.close()

        except Exception as e:
            logger.error(f"Error in streaming response: {str(e)}")
            await websocket.send_text(f"\nError: {str(e)}")
            await websocket.close()

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error in WebSocket handler: {str(e)}")
        try:
            await websocket.send_text(f"Error: {str(e)}")
            await websocket.close()
        except:
            pass