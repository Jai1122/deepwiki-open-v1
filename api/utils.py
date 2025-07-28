import logging
import tiktoken
import os
from typing import List, Tuple
from urllib.parse import urlparse, urlunparse, quote
import requests
import base64
import json
from adalflow.utils import get_adalflow_default_root_path
from adalflow.core.types import ModelType
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

def smart_chunk_text(text: str, max_tokens: int, overlap_tokens: int = 100) -> List[str]:
    """
    Intelligently chunk text into smaller pieces while preserving semantic structure.
    Prioritizes splitting at natural boundaries like paragraphs, sentences, and functions.
    """
    if count_tokens(text) <= max_tokens:
        return [text]
    
    chunks = []
    lines = text.split('\n')
    current_chunk = []
    current_tokens = 0
    
    for line in lines:
        line_tokens = count_tokens(line + '\n')
        
        # If adding this line would exceed the limit, finalize current chunk
        if current_tokens + line_tokens > max_tokens and current_chunk:
            chunk_text = '\n'.join(current_chunk)
            chunks.append(chunk_text)
            
            # Start new chunk with overlap from previous chunk
            if overlap_tokens > 0:
                overlap_lines = []
                overlap_count = 0
                for prev_line in reversed(current_chunk):
                    prev_tokens = count_tokens(prev_line + '\n')
                    if overlap_count + prev_tokens <= overlap_tokens:
                        overlap_lines.insert(0, prev_line)
                        overlap_count += prev_tokens
                    else:
                        break
                current_chunk = overlap_lines
                current_tokens = overlap_count
            else:
                current_chunk = []
                current_tokens = 0
        
        current_chunk.append(line)
        current_tokens += line_tokens
    
    # Add the final chunk if it has content
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    return chunks

def hierarchical_summarize(content: str, max_summary_tokens: int = 500, provider: str = "vllm", model: str = None) -> str:
    """
    Create a hierarchical summary of content by first summarizing chunks, then combining summaries.
    """
    from .config import get_model_config
    
    if count_tokens(content) <= max_summary_tokens:
        return content
    
    try:
        # Get model configuration
        config = get_model_config(provider, model)
        model_client = config["model_client"]
        model_kwargs = config["model_kwargs"]
        initialize_kwargs = config["initialize_kwargs"]
        
        # Initialize the client
        client = model_client(**initialize_kwargs)
        
        # First, chunk the content
        chunks = smart_chunk_text(content, max_tokens=1000)
        
        # Summarize each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            if not chunk or not chunk.strip():
                continue
                
            summary_prompt = f"""Provide a concise technical summary of this code/text chunk. Focus on:
- Main functions, classes, or concepts
- Key logic and algorithms  
- Important dependencies or imports
- Any notable patterns or architectural decisions

Chunk {i+1}/{len(chunks)}:
{chunk}

Summary:"""
            
            try:
                # Make API call to summarize chunk
                api_kwargs = client.convert_inputs_to_api_kwargs(
                    input=summary_prompt,
                    model_kwargs={**model_kwargs, "max_tokens": 200},
                    model_type=ModelType.LLM
                )
                
                response = client.call(api_kwargs, model_type=ModelType.LLM)
                summary = client.chat_completion_parser(response)
                
                # Validate summary before adding
                if summary and summary.strip() and len(summary.strip()) > 10:
                    chunk_summaries.append(summary.strip())
                else:
                    logger.warning(f"Received empty or invalid summary for chunk {i+1}")
                    
            except Exception as e:
                logger.warning(f"Failed to summarize chunk {i+1}: {e}")
                continue
        
        # Check if we have any valid summaries
        if not chunk_summaries:
            logger.warning("No valid chunk summaries generated, returning truncated original content")
            return content[:max_summary_tokens * 4]  # Rough char estimate
        
        # If we still have too many summaries, combine them
        combined_summaries = "\n\n".join(chunk_summaries)
        if count_tokens(combined_summaries) <= max_summary_tokens:
            return combined_summaries
        
        # Validate combined summaries before final pass
        if not combined_summaries or not combined_summaries.strip():
            logger.warning("Combined summaries are empty, returning original content")
            return content[:max_summary_tokens * 4]
        
        # Final summarization pass
        final_prompt = f"""Create a comprehensive technical summary by consolidating these individual summaries into a coherent overview:

Individual Summaries:
{combined_summaries}

Please provide a consolidated technical summary that combines the key points from all the above summaries:"""
        
        try:
            api_kwargs = client.convert_inputs_to_api_kwargs(
                input=final_prompt,
                model_kwargs={**model_kwargs, "max_tokens": max_summary_tokens},
                model_type=ModelType.LLM
            )
            
            response = client.call(api_kwargs, model_type=ModelType.LLM)
            final_summary = client.chat_completion_parser(response)
            
            # Validate final summary
            if final_summary and final_summary.strip() and len(final_summary.strip()) > 20:
                return final_summary.strip()
            else:
                logger.warning("Final summary is empty or too short, returning combined summaries")
                return combined_summaries
                
        except Exception as e:
            logger.warning(f"Final summarization failed: {e}, returning combined summaries")
            return combined_summaries
        
    except Exception as e:
        logger.error(f"Error in hierarchical summarization: {e}")
        # Fallback to simple truncation
        return content[:max_summary_tokens * 4]  # Rough char estimate

def estimate_processing_priority(file_path: str) -> int:
    """
    Estimate the processing priority of a file based on its path and type.
    Lower numbers = higher priority.
    """
    file_path_lower = file_path.lower()
    
    # High priority: core source files
    if any(pattern in file_path_lower for pattern in [
        '/src/', '/lib/', '/app/', '/core/', '/main', 'index.', '__init__'
    ]):
        return 1
    
    # Medium-high priority: configuration and build files
    if any(pattern in file_path_lower for pattern in [
        'config', 'setup', 'makefile', 'dockerfile', 'requirements', 'package.json'
    ]):
        return 2
    
    # Medium priority: documentation and examples
    if any(pattern in file_path_lower for pattern in [
        'readme', '.md', 'doc/', 'example', 'demo'
    ]):
        return 3
    
    # Lower priority: tests
    if any(pattern in file_path_lower for pattern in [
        'test', 'spec', '__test__', '.test.', '.spec.'
    ]):
        return 4
    
    # Lowest priority: generated, vendor, or cache files
    if any(pattern in file_path_lower for pattern in [
        'generated', 'vendor/', 'node_modules/', '.cache', 'dist/', 'build/'
    ]):
        return 5
    
    return 3  # Default medium priority

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
    Enhanced with smart chunking and summarization.
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

    # Enhanced truncation with smart handling
    target_file_tokens = min(file_content_tokens, available_tokens // 2)
    target_context_tokens = available_tokens - target_file_tokens
    
    # Truncate context text intelligently
    truncated_context = context_text
    if context_text_tokens > target_context_tokens:
        logger.warning(f"Truncating RAG context from {context_text_tokens} to {target_context_tokens} tokens.")
        # Try to keep the most relevant parts (beginning and end)
        context_chunks = smart_chunk_text(context_text, target_context_tokens // 2)
        if len(context_chunks) > 1:
            # Keep first and last chunks if multiple exist
            truncated_context = context_chunks[0] + "\n\n[... content truncated ...]\n\n" + context_chunks[-1]
        else:
            truncated_context = context_chunks[0] if context_chunks else ""
    
    # Truncate file content intelligently
    truncated_file = file_content
    if file_content_tokens > target_file_tokens:
        logger.warning(f"Truncating file content from {file_content_tokens} to {target_file_tokens} tokens.")
        
        # Check if hierarchical summarization is enabled (default: disabled for stability)
        use_summarization = os.environ.get('ENABLE_HIERARCHICAL_SUMMARIZATION', 'false').lower() in ['true', '1', 't']
        
        # Try hierarchical summarization for very large files (only if enabled)
        if use_summarization and file_content_tokens > target_file_tokens * 5:
            try:
                logger.info("Attempting hierarchical summarization for large file...")
                truncated_file = hierarchical_summarize(file_content, target_file_tokens)
            except Exception as e:
                logger.warning(f"Summarization failed, falling back to chunking: {e}")
                file_chunks = smart_chunk_text(file_content, target_file_tokens)
                truncated_file = file_chunks[0] if file_chunks else ""
        else:
            # Use smart chunking for files (more reliable)
            logger.info("Using smart chunking for file truncation")
            file_chunks = smart_chunk_text(file_content, target_file_tokens)
            truncated_file = file_chunks[0] if file_chunks else ""

    return truncated_file, truncated_context

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
