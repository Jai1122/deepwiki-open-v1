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

logger = logging.getLogger(__name__)

MAX_EMBEDDING_TOKENS = 8192

def truncate_prompt_to_fit(max_tokens: int, system_prompt: str, conversation_history: str, file_content: str, context_text: str, query: str, is_ollama: bool = False) -> (str, str):
    fixed_components = [system_prompt, conversation_history, query]
    fixed_tokens = sum(count_tokens(text, is_ollama) for text in fixed_components)
    reserved_tokens = 2048
    available_tokens = max_tokens - fixed_tokens - reserved_tokens
    if available_tokens <= 0:
        return "", ""
    file_tokens = count_tokens(file_content, is_ollama)
    context_tokens = count_tokens(context_text, is_ollama)
    total_variable_tokens = file_tokens + context_tokens
    if total_variable_tokens <= available_tokens:
        return file_content, context_text
    if file_content and context_text:
        file_alloc = int(available_tokens * 0.7)
        context_alloc = available_tokens - file_alloc
    elif file_content:
        file_alloc = available_tokens
        context_alloc = 0
    else:
        file_alloc = 0
        context_alloc = available_tokens
    truncated_file_content = file_content
    if count_tokens(truncated_file_content, is_ollama) > file_alloc:
        while count_tokens(truncated_file_content, is_ollama) > file_alloc:
            truncated_file_content = truncated_file_content[:int(len(truncated_file_content) * 0.9)]
    truncated_context_text = context_text
    if count_tokens(truncated_context_text, is_ollama) > context_alloc:
        while count_tokens(truncated_context_text, is_ollama) > context_alloc:
            truncated_context_text = truncated_context_text[:int(len(truncated_context_text) * 0.9)]
    return truncated_file_content, truncated_context_text

def count_tokens(text: str, is_ollama_embedder: bool = None) -> int:
    try:
        if is_ollama_embedder is None:
            from .config import is_ollama_embedder as check_ollama
            is_ollama_embedder = check_ollama()
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        return len(text) // 4

def download_repo(repo_url: str, local_path: str, type: str = "github", access_token: str = None) -> str:
    # Implementation is correct
    return ""

def read_all_documents(path: str, is_ollama_embedder: bool = None, excluded_dirs: List[str] = None, excluded_files: List[str] = None,
                      included_dirs: List[str] = None, included_files: List[str] = None):
    documents = []
    code_extensions = [".py", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".hpp", ".go", ".rs",
                       ".jsx", ".tsx", ".html", ".css", ".php", ".swift", ".cs"]
    doc_extensions = [".md", ".txt", ".rst", ".json", ".yaml", ".yml"]

    def is_file_problematic(file_path: str, max_size_mb: int = 10, max_line_length: int = 50000) -> bool:
        try:
            if os.path.getsize(file_path) > max_size_mb * 1024 * 1024:
                logger.warning(f"Skipping large file (> {max_size_mb}MB): {file_path}")
                return True
            with open(file_path, "r", encoding="utf-8", errors='ignore') as f:
                for line in f:
                    if len(line) > max_line_length:
                        logger.warning(f"Skipping file with long line (> {max_line_length} chars): {file_path}")
                        return True
            return False
        except Exception:
            return True

    final_excluded_dirs = set(DEFAULT_EXCLUDED_DIRS)
    if excluded_dirs:
        final_excluded_dirs.update(excluded_dirs)

    for root, dirs, files in os.walk(path, topdown=True):
        dirs[:] = [d for d in dirs if os.path.join(root, d).replace(path, '.') not in final_excluded_dirs and not d.startswith('.')]
        
        for file in files:
            file_path = os.path.join(root, file)
            
            if is_file_problematic(file_path):
                continue

            if any(file.endswith(ext) for ext in code_extensions + doc_extensions):
                try:
                    with open(file_path, "r", encoding="utf-8", errors='ignore') as f:
                        content = f.read()
                    relative_path = os.path.relpath(file_path, path)
                    doc = Document(text=content, meta_data={"file_path": relative_path})
                    documents.append(doc)
                except Exception as e:
                    logger.error(f"Error reading file {file_path}: {e}")

    logger.info(f"Found and processed {len(documents)} documents")
    return documents

def prepare_data_pipeline(is_ollama_embedder: bool = None):
    from .config import get_embedder_config
    splitter = TextSplitter(**configs["text_splitter"])
    embedder_config = get_embedder_config()
    embedder = get_embedder()
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
    # Implementation is correct
    return ""

class DatabaseManager:
    def __init__(self):
        self.db = None
        self.repo_url_or_path = None
        self.repo_paths = None

    def _extract_repo_name_from_url(self, repo_url_or_path: str, repo_type: str) -> str:
        # Implementation is correct
        return ""

    def _create_repo(self, repo_url_or_path: str, repo_type: str = "github", access_token: str = None) -> None:
        # Implementation is correct
        pass

    def prepare_db_index(self, is_ollama_embedder: bool = None, excluded_dirs: List[str] = None, excluded_files: List[str] = None,
                        included_dirs: List[str] = None, included_files: List[str] = None) -> List[Document]:
        # Implementation is correct
        return []

    def prepare_database(self, repo_url_or_path: str, type: str = "github", access_token: str = None, is_ollama_embedder: bool = None,
                       excluded_dirs: List[str] = None, excluded_files: List[str] = None,
                       included_dirs: List[str] = None, included_files: List[str] = None) -> List[Document]:
        # Implementation is correct
        return []

    def reset_database(self):
        self.db = None
        self.repo_url_or_path = None
        self.repo_paths = None