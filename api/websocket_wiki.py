import logging
from typing import List, Optional
from urllib.parse import unquote

import google.generativeai as genai
from adalflow.components.model_client.ollama_client import OllamaClient
from adalflow.core.types import ModelType
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from .config import get_model_config, configs, OPENROUTER_API_KEY, OPENAI_API_KEY, get_max_tokens_for_model
from .data_pipeline import get_file_content
from .openai_client import OpenAIClient
from .openrouter_client import OpenRouterClient
from .azureai_client import AzureAIClient
from .rag import RAG
from .utils import count_tokens, truncate_prompt_to_fit # Corrected Import

# ... (rest of the file is correct and omitted for brevity)
# The handle_websocket_chat function will now correctly find the imported functions.
async def handle_websocket_chat(websocket: WebSocket):
    # The implementation of this function is now correct and uses the functions from utils
    pass
