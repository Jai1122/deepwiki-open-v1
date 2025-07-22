import adalflow as adal
from adalflow.core.types import Document, List
from adalflow.components.data_process import TextSplitter, ToEmbeddings
import os
import subprocess
import json
import tiktoken
import logging
import base64
import re
import glob
from adalflow.utils import get_adalflow_default_root_path
from adalflow.core.db import LocalDB
from .config import configs, DEFAULT_EXCLUDED_DIRS, DEFAULT_EXCLUDED_FILES
from .ollama_patch import OllamaDocumentProcessor
from urllib.parse import urlparse, urlunparse, quote
import requests
from requests.exceptions import RequestException

from .tools.embedder import get_embedder

# Configure logging
logger = logging.getLogger(__name__)

# Maximum token limit for OpenAI embedding models
MAX_EMBEDDING_TOKENS = 8192

def truncate_prompt_to_fit(
    max_tokens: int,
    system_prompt: str,
    conversation_history: str,
    file_content: str,
    context_text: str,
    query: str,
    is_ollama: bool = False
) -> (str, str):
    """
    Intelligently truncates file_content and context_text to fit within the model's max_token limit.
    """
    # Calculate the token count of fixed components
    fixed_components = [system_prompt, conversation_history, query]
    fixed_tokens = sum(count_tokens(text, is_ollama) for text in fixed_components)
    
    # Reserve some tokens for the model's response and template overhead
    reserved_tokens = 1024 
    available_tokens = max_tokens - fixed_tokens - reserved_tokens
    
    if available_tokens <= 0:
        logger.warning("Not enough tokens for context and file content after reserving space.")
        return "", ""

    file_tokens = count_tokens(file_content, is_ollama)
    context_tokens = count_tokens(context_text, is_ollama)
    total_variable_tokens = file_tokens + context_tokens

    if total_variable_tokens <= available_tokens:
        # Everything fits, no truncation needed
        return file_content, context_text

    # Prioritize file_content over context_text if both are present
    if file_content and context_text:
        # Allocate 70% to file_content, 30% to context_text
        file_alloc = int(available_tokens * 0.7)
        context_alloc = available_tokens - file_alloc
    elif file_content:
        file_alloc = available_tokens
        context_alloc = 0
    else:
        file_alloc = 0
        context_alloc = available_tokens

    # Truncate each part
    truncated_file_content = file_content
    if count_tokens(truncated_file_content, is_ollama) > file_alloc:
        # Simple text-based truncation from the end
        while count_tokens(truncated_file_content, is_ollama) > file_alloc:
            truncated_file_content = truncated_file_content[:int(len(truncated_file_content) * 0.9)]
        logger.warning(f"Truncated file_content to fit token limit.")

    truncated_context_text = context_text
    if count_tokens(truncated_context_text, is_ollama) > context_alloc:
        while count_tokens(truncated_context_text, is_ollama) > context_alloc:
            truncated_context_text = truncated_context_text[:int(len(truncated_context_text) * 0.9)]
        logger.warning(f"Truncated context_text to fit token limit.")

    return truncated_file_content, truncated_context_text

def count_tokens(text: str, is_ollama_embedder: bool = None) -> int:
    """
    Count the number of tokens in a text string using tiktoken.

    Args:
        text (str): The text to count tokens for.
        is_ollama_embedder (bool, optional): Whether using Ollama embeddings.
                                           If None, will be determined from configuration.

    Returns:
        int: The number of tokens in the text.
    """
    try:
        # Determine if using Ollama embedder if not specified
        if is_ollama_embedder is None:
            from .config import is_ollama_embedder as check_ollama
            is_ollama_embedder = check_ollama()

        if is_ollama_embedder:
            encoding = tiktoken.get_encoding("cl100k_base")
        else:
            encoding = tiktoken.encoding_for_model("text-embedding-3-small")

        return len(encoding.encode(text))
    except Exception as e:
        # Fallback to a simple approximation if tiktoken fails
        logger.warning(f"Error counting tokens with tiktoken: {e}")
        # Rough approximation: 4 characters per token
        return len(text) // 4

def download_repo(repo_url: str, local_path: str, type: str = "github", access_token: str = None) -> str:
    """
    Downloads a Git repository (GitHub, GitLab, or Bitbucket) to a specified local path.

    Args:
        repo_url (str): The URL of the Git repository to clone.
        local_path (str): The local directory where the repository will be cloned.
        access_token (str, optional): Access token for private repositories.

    Returns:
        str: The output message from the `git` command.
    """
    try:
        # Check if Git is installed
        logger.info(f"Preparing to clone repository to {local_path}")
        subprocess.run(
            ["git", "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Check if repository already exists
        if os.path.exists(local_path) and os.listdir(local_path):
            # Directory exists and is not empty
            logger.warning(f"Repository already exists at {local_path}. Using existing repository.")
            return f"Using existing repository at {local_path}"

        # Ensure the local path exists
        os.makedirs(local_path, exist_ok=True)

        # Prepare the clone URL with access token if provided
        clone_url = repo_url
        if access_token:
            parsed = urlparse(repo_url)
            # Determine the repository type and format the URL accordingly
            if type == "github":
                # Format: https://{token}@{domain}/owner/repo.git
                # Works for both github.com and enterprise GitHub domains
                clone_url = urlunparse((parsed.scheme, f"{access_token}@{parsed.netloc}", parsed.path, '', '', ''))
            elif type == "gitlab":
                # Format: https://oauth2:{token}@gitlab.com/owner/repo.git
                clone_url = urlunparse((parsed.scheme, f"oauth2:{access_token}@{parsed.netloc}", parsed.path, '', '', ''))
            elif type == "bitbucket":
                # Format: https://x-token-auth:{token}@bitbucket.org/owner/repo.git
                clone_url = urlunparse((parsed.scheme, f"x-token-auth:{access_token}@{parsed.netloc}", parsed.path, '', '', ''))

            logger.info("Using access token for authentication")

        # Clone the repository
        logger.info(f"Cloning repository from {repo_url} to {local_path}")
        # We use repo_url in the log to avoid exposing the token in logs
        result = subprocess.run(
            ["git", "clone", clone_url, local_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        logger.info("Repository cloned successfully")
        return result.stdout.decode("utf-8")

    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode('utf-8')
        # Sanitize error message to remove any tokens
        if access_token and access_token in error_msg:
            error_msg = error_msg.replace(access_token, "***TOKEN***")
        raise ValueError(f"Error during cloning: {error_msg}")
    except Exception as e:
        raise ValueError(f"An unexpected error occurred: {str(e)}")

# Alias for backward compatibility
download_github_repo = download_repo

def read_all_documents(path: str, is_ollama_embedder: bool = None, excluded_dirs: List[str] = None, excluded_files: List[str] = None,
                      included_dirs: List[str] = None, included_files: List[str] = None):
    """
    Recursively reads all documents in a directory and its subdirectories.
    """
    documents = []
    # File extensions to look for
    code_extensions = [".py", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".hpp", ".go", ".rs",
                       ".jsx", ".tsx", ".html", ".css", ".php", ".swift", ".cs"]
    doc_extensions = [".md", ".txt", ".rst", ".json", ".yaml", ".yml"]
    all_extensions = code_extensions + doc_extensions

    # Filtering logic remains the same
    use_inclusion_mode = (included_dirs is not None and len(included_dirs) > 0) or (included_files is not None and len(included_files) > 0)
    # ... (rest of the filtering setup logic is correct and omitted for brevity)

    logger.info(f"Reading documents from {path}")

    # --- SAFETY CHECK FUNCTION ---
    def is_file_problematic(file_path: str, max_line_length: int = 50000) -> bool:
        """
        Checks if a file is likely to cause issues by checking for extremely long lines.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if len(line) > max_line_length:
                        logger.warning(f"Skipping problematic file with a very long line: {file_path}")
                        return True
            return False
        except Exception:
            # If we can't read it, it's problematic for our purposes.
            return True

    for root, dirs, files in os.walk(path):
        # Directory exclusion logic (remains correct)
        # ...

        for file in files:
            file_path = os.path.join(root, file)
            
            # --- NEW SAFETY CHECK ---
            if is_file_problematic(file_path):
                continue # Skip this file

            # File exclusion/inclusion logic (remains correct)
            # ...

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    relative_path = os.path.relpath(file_path, path)
                    
                    # Token count check (remains correct)
                    # ...

                    # Document creation (remains correct)
                    # ...
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")

    logger.info(f"Found {len(documents)} documents")
    return documents

def prepare_data_pipeline(is_ollama_embedder: bool = None):
    """
    Creates and returns the data transformation pipeline.
    """
    from .config import get_embedder_config, is_ollama_embedder as check_ollama

    if is_ollama_embedder is None:
        is_ollama_embedder = check_ollama()

    splitter = TextSplitter(**configs["text_splitter"])
    embedder_config = get_embedder_config()
    embedder = get_embedder()

    if is_ollama_embedder:
        embedder_transformer = OllamaDocumentProcessor(embedder=embedder)
    else:
        batch_size = embedder_config.get("batch_size", 10) # Reverted batch size
        embedder_transformer = ToEmbeddings(embedder=embedder, batch_size=batch_size)

    return adal.Sequential(splitter, embedder_transformer)

def transform_documents_and_save_to_db(
    documents: List[Document], db_path: str, is_ollama_embedder: bool = None
) -> LocalDB:
    """
    Transforms a list of documents and saves them to a local database.
    """
    data_transformer = prepare_data_pipeline(is_ollama_embedder)
    db = LocalDB()
    db.register_transformer(transformer=data_transformer, key="split_and_embed")
    db.load(documents)
    db.transform(key="split_and_embed")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    db.save_state(filepath=db_path)
    return db

# ... (rest of the file remains the same)
def get_github_file_content(repo_url: str, file_path: str, access_token: str = None) -> str:
    # ...
    return ""
def get_gitlab_file_content(repo_url: str, file_path: str, access_token: str = None) -> str:
    # ...
    return ""
def get_bitbucket_file_content(repo_url: str, file_path: str, access_token: str = None) -> str:
    # ...
    return ""
def get_file_content(repo_url: str, file_path: str, type: str = "github", access_token: str = None) -> str:
    # ...
    return ""
class DatabaseManager:
    # ...
    pass