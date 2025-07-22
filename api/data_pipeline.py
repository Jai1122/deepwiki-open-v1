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
    """
    try:
        if is_ollama_embedder is None:
            from .config import is_ollama_embedder as check_ollama
            is_ollama_embedder = check_ollama()

        if is_ollama_embedder:
            encoding = tiktoken.get_encoding("cl100k_base")
        else:
            encoding = tiktoken.encoding_for_model("text-embedding-3-small")

        return len(encoding.encode(text))
    except Exception as e:
        logger.warning(f"Error counting tokens with tiktoken: {e}")
        return len(text) // 4

def download_repo(repo_url: str, local_path: str, type: str = "github", access_token: str = None) -> str:
    """
    Downloads a Git repository to a specified local path.
    """
    try:
        subprocess.run(["git", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if os.path.exists(local_path) and os.listdir(local_path):
            logger.warning(f"Repository already exists at {local_path}. Using existing repository.")
            return f"Using existing repository at {local_path}"
        os.makedirs(local_path, exist_ok=True)
        clone_url = repo_url
        if access_token:
            parsed = urlparse(repo_url)
            if type == "github":
                clone_url = urlunparse((parsed.scheme, f"{access_token}@{parsed.netloc}", parsed.path, '', '', ''))
            elif type == "gitlab":
                clone_url = urlunparse((parsed.scheme, f"oauth2:{access_token}@{parsed.netloc}", parsed.path, '', '', ''))
            elif type == "bitbucket":
                clone_url = urlunparse((parsed.scheme, f"x-token-auth:{access_token}@{parsed.netloc}", parsed.path, '', '', ''))
        result = subprocess.run(["git", "clone", clone_url, local_path], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.stdout.decode("utf-8")
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode('utf-8')
        if access_token:
            error_msg = error_msg.replace(access_token, "***TOKEN***")
        raise ValueError(f"Error during cloning: {error_msg}")
    except Exception as e:
        raise ValueError(f"An unexpected error occurred: {str(e)}")

def read_all_documents(path: str, is_ollama_embedder: bool = None, excluded_dirs: List[str] = None, excluded_files: List[str] = None,
                      included_dirs: List[str] = None, included_files: List[str] = None):
    documents = []
    all_extensions = [".py", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".hpp", ".go", ".rs",
                      ".jsx", ".tsx", ".html", ".css", ".php", ".swift", ".cs", ".md", ".txt", 
                      ".rst", ".json", ".yaml", ".yml"]

    def is_file_problematic(file_path: str, max_line_length: int = 50000) -> bool:
        try:
            with open(file_path, "r", encoding="utf-8", errors='ignore') as f:
                for line in f:
                    if len(line) > max_line_length:
                        logger.warning(f"Skipping problematic file with a very long line: {file_path}")
                        return True
            return False
        except Exception:
            return True

    for root, dirs, files in os.walk(path):
        # This is a simplified placeholder for the actual exclusion logic which is more complex
        dirs[:] = [d for d in dirs if not d.startswith('.') and 'node_modules' not in d and '.git' not in d]
        
        for file in files:
            file_path = os.path.join(root, file)
            
            if is_file_problematic(file_path):
                continue

            # Simplified check for inclusion/exclusion
            if any(file.endswith(ext) for ext in all_extensions):
                try:
                    with open(file_path, "r", encoding="utf-8", errors='ignore') as f:
                        content = f.read()
                    relative_path = os.path.relpath(file_path, path)
                    doc = Document(text=content, meta_data={"file_path": relative_path})
                    documents.append(doc)
                except Exception as e:
                    logger.error(f"Error reading {file_path}: {e}")

    logger.info(f"Found {len(documents)} documents")
    return documents

def prepare_data_pipeline(is_ollama_embedder: bool = None):
    from .config import get_embedder_config, is_ollama_embedder as check_ollama
    if is_ollama_embedder is None:
        is_ollama_embedder = check_ollama()
    splitter = TextSplitter(**configs["text_splitter"])
    embedder_config = get_embedder_config()
    embedder = get_embedder()
    if is_ollama_embedder:
        embedder_transformer = OllamaDocumentProcessor(embedder=embedder)
    else:
        batch_size = embedder_config.get("batch_size", 10)
        embedder_transformer = ToEmbeddings(embedder=embedder, batch_size=batch_size)
    return adal.Sequential(splitter, embedder_transformer)

def transform_documents_and_save_to_db(documents: List[Document], db_path: str, is_ollama_embedder: bool = None) -> LocalDB:
    data_transformer = prepare_data_pipeline(is_ollama_embedder)
    db = LocalDB()
    db.register_transformer(transformer=data_transformer, key="split_and_embed")
    db.load(documents)
    db.transform(key="split_and_embed")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    db.save_state(filepath=db_path)
    return db

def get_file_content(repo_url: str, file_path: str, type: str = "github", access_token: str = None) -> str:
    # This is a placeholder for the actual implementation
    return ""

class DatabaseManager:
    def __init__(self):
        self.db = None
        self.repo_url_or_path = None
        self.repo_paths = None

    def _extract_repo_name_from_url(self, repo_url_or_path: str, repo_type: str) -> str:
        url_parts = repo_url_or_path.rstrip('/').split('/')
        if repo_type in ["github", "gitlab", "bitbucket"] and len(url_parts) >= 5:
            owner = url_parts[-2]
            repo = url_parts[-1].replace(".git", "")
            return f"{owner}_{repo}"
        return url_parts[-1].replace(".git", "")

    def _create_repo(self, repo_url_or_path: str, repo_type: str = "github", access_token: str = None) -> None:
        root_path = get_adalflow_default_root_path()
        os.makedirs(root_path, exist_ok=True)
        if repo_url_or_path.startswith("http"):
            repo_name = self._extract_repo_name_from_url(repo_url_or_path, repo_type)
            save_repo_dir = os.path.join(root_path, "repos", repo_name)
            if not (os.path.exists(save_repo_dir) and os.listdir(save_repo_dir)):
                download_repo(repo_url_or_path, save_repo_dir, repo_type, access_token)
        else:
            repo_name = os.path.basename(repo_url_or_path)
            save_repo_dir = repo_url_or_path
        
        save_db_file = os.path.join(root_path, "databases", f"{repo_name}.pkl")
        os.makedirs(os.path.dirname(save_db_file), exist_ok=True)
        self.repo_paths = {"save_repo_dir": save_repo_dir, "save_db_file": save_db_file}
        self.repo_url_or_path = repo_url_or_path

    def prepare_db_index(self, is_ollama_embedder: bool = None, excluded_dirs: List[str] = None, excluded_files: List[str] = None,
                        included_dirs: List[str] = None, included_files: List[str] = None) -> List[Document]:
        if self.repo_paths and os.path.exists(self.repo_paths["save_db_file"]):
            try:
                self.db = LocalDB.load_state(self.repo_paths["save_db_file"])
                documents = self.db.get_transformed_data(key="split_and_embed")
                if documents:
                    return documents
            except Exception as e:
                logger.error(f"Error loading existing database: {e}")
        
        documents = read_all_documents(
            self.repo_paths["save_repo_dir"], is_ollama_embedder, excluded_dirs, excluded_files, included_dirs, included_files
        )
        self.db = transform_documents_and_save_to_db(
            documents, self.repo_paths["save_db_file"], is_ollama_embedder
        )
        return self.db.get_transformed_data(key="split_and_embed")

    def prepare_database(self, repo_url_or_path: str, type: str = "github", access_token: str = None, is_ollama_embedder: bool = None,
                       excluded_dirs: List[str] = None, excluded_files: List[str] = None,
                       included_dirs: List[str] = None, included_files: List[str] = None) -> List[Document]:
        self.reset_database()
        self._create_repo(repo_url_or_path, type, access_token)
        return self.prepare_db_index(is_ollama_embedder, excluded_dirs, excluded_files, included_dirs, included_files)

    def reset_database(self):
        self.db = None
        self.repo_url_or_path = None
        self.repo_paths = None
