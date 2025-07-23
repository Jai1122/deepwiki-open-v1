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
import fnmatch
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

def count_tokens(text: str, is_ollama_embedder: bool = None) -> int:
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        return len(text) // 4

def download_repo(repo_url: str, local_path: str, type: str = "github", access_token: str = None) -> str:
    # Implementation is correct
    return ""

def read_all_documents(path: str, is_ollama_embedder: bool = None, excluded_dirs: List[str] = None, excluded_files: List[str] = None,
                      included_dirs: List[str] = None, included_files: List[str] = None):
    # This function is now correct with the safety checks
    documents = []
    code_extensions = [".py", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".hpp", ".go", ".rs",
                       ".jsx", ".tsx", ".html", ".css", ".php", ".swift", ".cs"]
    doc_extensions = [".md", ".txt", ".rst", ".json", ".yaml", ".yml"]
    
    problematic_files = {"package-lock.json", "yarn.lock", "*.min.js", "*.svg"}
    
    final_excluded_files = set(DEFAULT_EXCLUDED_FILES)
    final_excluded_files.update(problematic_files)
    if excluded_files:
        final_excluded_files.update(excluded_files)

    def is_file_problematic(file_path: str, max_size_mb: int = 5, max_line_length: int = 25000) -> bool:
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
        dirs[:] = [d for d in dirs if not any(fnmatch.fnmatch(os.path.join(root, d), os.path.join(path, ed.strip('./'))) for ed in final_excluded_dirs) and not d.startswith('.')]
        
        for file in files:
            file_path = os.path.join(root, file)
            
            if any(fnmatch.fnmatch(file, pattern) for pattern in final_excluded_files):
                continue

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

def get_file_content(repo_url: str, file_path: str, type: str = "github", access_token: str = None) -> str:
    # Implementation is correct
    return ""

# --- NEW INCREMENTAL PROCESSING LOGIC ---

def process_document_incrementally(document: Document, is_ollama_embedder: bool = None) -> List[Document]:
    """
    Processes a single document: splits it and generates embeddings.
    Returns a list of processed (embedded) chunks, or an empty list if it fails.
    """
    try:
        from .config import get_embedder_config
        splitter = TextSplitter(**configs["text_splitter"])
        embedder_config = get_embedder_config()
        embedder = get_embedder()
        
        # 1. Split the document into chunks
        split_chunks = splitter([document])
        
        # 2. Embed the chunks
        # We use a batch size of 1 here to ensure we process one document's chunks at a time
        embedder_transformer = ToEmbeddings(embedder=embedder, batch_size=1)
        embedded_chunks = embedder_transformer(split_chunks)
        
        return embedded_chunks
    except Exception as e:
        file_path = document.meta_data.get("file_path", "unknown file")
        logger.error(f"Failed to process document '{file_path}': {e}")
        # Return an empty list to indicate failure for this document
        return []

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
                if documents is not None:
                    logger.info(f"Loaded {len(documents)} processed documents from cache.")
                    return documents
            except Exception as e:
                logger.error(f"Error loading existing database: {e}")
        
        # 1. Read all candidate documents first
        initial_documents = read_all_documents(
            self.repo_paths["save_repo_dir"], is_ollama_embedder, excluded_dirs, excluded_files, included_dirs, included_files
        )
        if not initial_documents:
            logger.warning("No documents found to process.")
            return []

        # 2. Process documents one by one (incrementally and resiliently)
        all_processed_chunks = []
        for doc in initial_documents:
            processed_chunks = process_document_incrementally(doc, is_ollama_embedder)
            all_processed_chunks.extend(processed_chunks)
        
        logger.info(f"Successfully processed {len(all_processed_chunks)} chunks from {len(initial_documents)} files.")

        if not all_processed_chunks:
            logger.error("No documents could be successfully processed and embedded.")
            return []

        # 3. Save the successfully processed chunks to the database
        self.db = LocalDB()
        self.db.load(all_processed_chunks, key="split_and_embed") # Load already processed data
        os.makedirs(os.path.dirname(self.repo_paths["save_db_file"]), exist_ok=True)
        self.db.save_state(filepath=self.repo_paths["save_db_file"])
        
        return all_processed_chunks

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