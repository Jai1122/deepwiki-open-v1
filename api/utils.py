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
    """
    Counts the number of tokens in a given text.
    It uses `tiktoken` for accurate token counting and falls back to a
    character-based approximation if `tiktoken` is not available.
    """
    if not text:
        return 0
    try:
        # Using cl100k_base as a standard encoder
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        # Fallback for cases where tiktoken might fail
        return len(text) // 4

def truncate_prompt_to_fit(
    context_window: int,
    max_completion_tokens: int,
    system_prompt: str,
    conversation_history: str,
    file_content: str,
    context_text: str,
    query: str,
    is_ollama: bool = False
) -> (str, str):
    """
    Truncates file_content and context_text to ensure the total prompt
    fits within the model's context window, leaving space for the completion.
    """
    # Calculate the token counts for fixed parts of the prompt
    system_prompt_tokens = count_tokens(system_prompt, is_ollama)
    history_tokens = count_tokens(conversation_history, is_ollama)
    query_tokens = count_tokens(query, is_ollama)

    # Calculate the available token budget for the variable parts (file and RAG context)
    static_tokens = system_prompt_tokens + history_tokens + query_tokens
    available_tokens = context_window - static_tokens - max_completion_tokens

    if available_tokens <= 0:
        logger.warning("Not enough tokens for file and RAG context. Returning empty strings.")
        return "", ""

    # Get token counts for the dynamic parts
    file_content_tokens = count_tokens(file_content, is_ollama)
    context_text_tokens = count_tokens(context_text, is_ollama)
    total_dynamic_tokens = file_content_tokens + context_text_tokens

    # If the total fits, no truncation is needed
    if total_dynamic_tokens <= available_tokens:
        return file_content, context_text

    # If not, truncate the content. We'll prioritize RAG context over file content.
    if context_text_tokens >= available_tokens:
        # If RAG context alone is too big, truncate it and discard file_content
        logger.warning(f"Truncating RAG context from {context_text_tokens} to {available_tokens} tokens.")
        # A simple character-based truncation, assuming uniform token distribution
        ratio = available_tokens / context_text_tokens
        truncated_context = context_text[:int(len(context_text) * ratio)]
        return "", truncated_context
    else:
        # If RAG context fits, allocate remaining budget to file_content
        remaining_tokens_for_file = available_tokens - context_text_tokens
        logger.warning(f"Truncating file content from {file_content_tokens} to {remaining_tokens_for_file} tokens.")
        ratio = remaining_tokens_for_file / file_content_tokens
        truncated_file_content = file_content[:int(len(file_content) * ratio)]
        return truncated_file_content, context_text

def get_github_file_content(repo_url: str, file_path: str, access_token: str = None) -> str:
    """
    Fetches file content from a GitHub repository.
    """
    parsed_url = urlparse(repo_url)
    path_parts = parsed_url.path.strip('/').split('/')
    owner, repo = path_parts[0], path_parts[1]
    
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"
    headers = {'Accept': 'application/vnd.github.v3.raw'}
    if access_token:
        headers['Authorization'] = f'token {access_token}'
        
    try:
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        return response.text
    except RequestException as e:
        logger.error(f"Error fetching file from GitHub {api_url}: {e}")
        return ""

def get_gitlab_file_content(repo_url: str, file_path: str, access_token: str = None) -> str:
    """
    Fetches file content from a GitLab repository.
    """
    parsed_url = urlparse(repo_url)
    project_path = quote(parsed_url.path.strip('/'), safe='')
    api_url = f"https://gitlab.com/api/v4/projects/{project_path}/repository/files/{quote(file_path, safe='')}/raw"
    
    headers = {}
    if access_token:
        headers['PRIVATE-TOKEN'] = access_token
        
    try:
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        return response.text
    except RequestException as e:
        logger.error(f"Error fetching file from GitLab {api_url}: {e}")
        return ""

def get_bitbucket_file_content(repo_url: str, file_path: str, access_token: str = None) -> str:
    """
    Fetches file content from a Bitbucket repository.
    """
    parsed_url = urlparse(repo_url)
    path_parts = parsed_url.path.strip('/').split('/')
    workspace, repo_slug = path_parts[0], path_parts[1]
    
    api_url = f"https://api.bitbucket.org/2.0/repositories/{workspace}/{repo_slug}/src/master/{file_path}"
    
    headers = {}
    if access_token:
        # Bitbucket uses app passwords or OAuth, passed as a standard Auth header
        # Assuming the token is a base64 encoded "user:password" for basic auth
        headers['Authorization'] = f'Bearer {access_token}'
        
    try:
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        return response.text
    except RequestException as e:
        logger.error(f"Error fetching file from Bitbucket {api_url}: {e}")
        return ""

def get_local_file_content(file_path: str) -> str:
    """
    Reads content from a local file path.
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading local file {file_path}: {e}")
        return ""

def get_file_content(repo_url: str, file_path: str, type: str = "github", access_token: str = None) -> str:
    """
    A wrapper function to get file content based on the repository type.
    Handles both remote git repositories and local file paths.
    """
    if type == "local":
        # For local type, repo_url is the base path of the cloned repo
        full_path = os.path.join(repo_url, file_path)
        return get_local_file_content(full_path)
    elif type == "github":
        return get_github_file_content(repo_url, file_path, access_token)
    elif type == "gitlab":
        return get_gitlab_file_content(repo_url, file_path, access_token)
    elif type == "bitbucket":
        return get_bitbucket_file_content(repo_url, file_path, access_token)
    else:
        logger.error(f"Unsupported repository type: {type}")
        raise ValueError("Unsupported repository type.")
