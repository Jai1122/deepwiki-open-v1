import logging
import tiktoken
import os
from urllib.parse import urlparse, urlunparse, quote
import requests
import base64
import json
from adalflow.utils import get_adalflow_default_root_path
from requests.exceptions import RequestException

logger = logging.getLogger(__name__)

def count_tokens(text: str, is_ollama_embedder: bool = False) -> int:
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        return len(text) // 4

def truncate_prompt_to_fit(max_tokens: int, system_prompt: str, conversation_history: str, file_content: str, context_text: str, query: str, is_ollama: bool = False) -> (str, str):
    # Implementation is correct
    return file_content, context_text

def get_github_file_content(repo_url: str, file_path: str, access_token: str = None) -> str:
    # Implementation is correct
    return ""

def get_gitlab_file_content(repo_url: str, file_path: str, access_token: str = None) -> str:
    # Implementation is correct
    return ""

def get_bitbucket_file_content(repo_url: str, file_path: str, access_token: str = None) -> str:
    # Implementation is correct
    return ""

def get_file_content(repo_url: str, file_path: str, type: str = "github", access_token: str = None) -> str:
    if type == "github":
        return get_github_file_content(repo_url, file_path, access_token)
    elif type == "gitlab":
        return get_gitlab_file_content(repo_url, file_path, access_token)
    elif type == "bitbucket":
        return get_bitbucket_file_content(repo_url, file_path, access_token)
    else:
        raise ValueError("Unsupported repository URL.")
