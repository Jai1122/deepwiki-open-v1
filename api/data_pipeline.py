from adalflow.core.types import Document, List
from adalflow.components.data_process import TextSplitter, ToEmbeddings
import os
import subprocess
import json
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
from .utils import get_local_file_content, count_tokens

logger = logging.getLogger(__name__)

def download_repo(repo_url: str, local_path: str, type: str = "github", access_token: str = None) -> str:
    """
    Clones a repository from GitHub, GitLab, or Bitbucket.
    Handles authentication for private repositories.
    """
    if os.path.exists(local_path):
        logger.info(f"Repository already exists at {local_path}, skipping download.")
        return local_path

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    clone_url = repo_url
    if type in ["github", "gitlab"] and access_token:
        parsed_url = urlparse(repo_url)
        if type == "github":
            clone_url = f"https://{access_token}@{parsed_url.netloc}{parsed_url.path}"
        elif type == "gitlab":
            clone_url = f"https://oauth2:{access_token}@{parsed_url.netloc}{parsed_url.path}"

    logger.info(f"Cloning repository from {repo_url} to {local_path}...")
    
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", clone_url, local_path],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info("Repository cloned successfully.")
        return local_path
    except subprocess.CalledProcessError as e:
        error_message = f"Failed to clone repository: {e.stderr}"
        logger.error(error_message)
        raise RuntimeError(error_message)

def read_all_documents(
    path: str,
    is_ollama_embedder: bool = None,
    excluded_dirs: List[str] = None,
    excluded_files: List[str] = None,
    included_dirs: List[str] = None,
    included_files: List[str] = None
) -> List[Document]:
    """
    Reads all files from a directory, splits them into chunks, and returns a list of Document objects.
    This version processes files one by one and handles directory exclusions correctly.
    """
    splitter_config = configs.get("text_splitter", {})
    text_splitter = TextSplitter(
        split_by=splitter_config.get("split_by", "word"),
        chunk_size=splitter_config.get("chunk_size", 1000),
        chunk_overlap=splitter_config.get("chunk_overlap", 200),
    )

    all_documents = []
    final_excluded_dirs = DEFAULT_EXCLUDED_DIRS + (excluded_dirs or [])
    final_excluded_files = DEFAULT_EXCLUDED_FILES + (excluded_files or [])

    # Normalize exclusion patterns for reliable matching
    # Example: './node_modules/' becomes 'node_modules'
    normalized_excluded_dirs = [p.strip('./').strip('/') for p in final_excluded_dirs]

    for root, dirs, files in os.walk(path, topdown=True):
        # Filter directories in-place to prevent traversing them
        dirs[:] = [d for d in dirs if d not in normalized_excluded_dirs]
        
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, path)

            # Normalize path for matching
            normalized_relative_path = relative_path.replace('\\', '/')

            if any(fnmatch.fnmatch(normalized_relative_path, pattern) for pattern in final_excluded_files):
                continue
            if included_files and not any(fnmatch.fnmatch(normalized_relative_path, pattern) for pattern in included_files):
                continue

            try:
                content = get_local_file_content(file_path)
                if not content.strip():
                    continue

                chunks = text_splitter.split_text(content)
                for chunk_content in chunks:
                    doc = Document(text=chunk_content)
                    doc.metadata = {"source": relative_path}
                    all_documents.append(doc)

            except Exception as e:
                logger.warning(f"Could not process file {file_path}: {e}")

    logger.info(f"Total documents after chunking: {len(all_documents)}")
    return all_documents


def prepare_data_pipeline(is_ollama_embedder: bool = None):
    """
    Prepares the data processing pipeline with an embedder.
    """
    embedder = get_embedder(is_ollama_embedder)
    
    if is_ollama_embedder:
        # Use the patched processor for Ollama
        return OllamaDocumentProcessor(embedder=embedder)
    else:
        # Standard pipeline for other embedders
        return ToEmbeddings(
            embedder=embedder,
            batch_size=configs.get("embedder", {}).get("batch_size", 10)
        )

def transform_documents_and_save_to_db(documents: List[Document], db_path: str, is_ollama_embedder: bool = None) -> LocalDB:
    """
    Transforms documents using the data pipeline and saves them to a local database.
    """
    if not documents:
        logger.warning("No documents to process.")
        return None

    pipeline = prepare_data_pipeline(is_ollama_embedder)
    transformed_docs = pipeline(documents)

    if not transformed_docs:
        logger.error("Transformation pipeline returned no documents.")
        return None

    # Save transformed documents to the local DB
    db = LocalDB(db_path)
    db.add(transformed_docs)
    logger.info(f"Saved {len(transformed_docs)} transformed documents to {db_path}")
    
    return db

class DatabaseManager:
    """
    Manages the lifecycle of creating and loading the vector database for a repository.
    """
    def __init__(self):
        self.db = None

    def prepare_database(self, repo_url_or_path: str, type: str = "github", access_token: str = None, is_ollama_embedder: bool = None,
                       excluded_dirs: List[str] = None, excluded_files: List[str] = None,
                       included_dirs: List[str] = None, included_files: List[str] = None) -> List[Document]:
        """
        Main method to prepare the database. It handles cloning, loading from cache,
        or processing and saving documents.
        """
        repo_name = os.path.basename(repo_url_or_path.rstrip('/'))
        db_path = os.path.join(get_adalflow_default_root_path(), "databases", repo_name)

        if os.path.exists(db_path):
            logger.info(f"Loading database from existing path: {db_path}")
            self.db = LocalDB(db_path)
            return self.db.load()

        # Determine the path for the repository content
        if type == "local":
            repo_path = repo_url_or_path
        else:
            repo_path = os.path.join(get_adalflow_default_root_path(), "repos", repo_name)
            download_repo(repo_url_or_path, repo_path, type, access_token)

        # Read and process documents
        documents = read_all_documents(
            repo_path, is_ollama_embedder, excluded_dirs, excluded_files, included_dirs, included_files
        )
        
        if not documents:
            logger.warning("No documents were read from the repository.")
            return []

        # Transform and save to DB
        self.db = transform_documents_and_save_to_db(documents, db_path, is_ollama_embedder)
        
        return self.db.load() if self.db else []

    def reset_database(self):
        """Resets the current database instance."""
        self.db = None