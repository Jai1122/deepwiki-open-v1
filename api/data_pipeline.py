import pickle
from typing import Optional
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
from .config import configs, DEFAULT_EXCLUDED_DIRS, DEFAULT_EXCLUDED_FILES
from .ollama_patch import OllamaDocumentProcessor
from urllib.parse import urlparse, urlunparse, quote
import requests
from requests.exceptions import RequestException

from .tools.embedder import get_embedder
from .utils import get_local_file_content, count_tokens, estimate_processing_priority, smart_chunk_text

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
    included_files: List[str] = None,
    max_total_tokens: int = 1000000,  # Max tokens to process across all files
    prioritize_files: bool = True      # Whether to prioritize important files
) -> List[Document]:
    """
    Reads all files from a directory, splits them into chunks, and returns a list of Document objects.
    Enhanced with smart processing for large repositories.
    """
    splitter_config = configs.get("text_splitter", {})
    
    # Use smarter chunking for large repos
    chunk_size = min(splitter_config.get("chunk_size", 1000), 2000)  # Cap chunk size
    chunk_overlap = min(splitter_config.get("chunk_overlap", 200), chunk_size // 4)
    
    text_splitter = TextSplitter(
        split_by=splitter_config.get("split_by", "word"),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    all_documents = []
    processed_files_log = []
    rejected_files_log = []
    file_candidates = []  # Store files with their priorities for processing
    
    final_excluded_dirs = DEFAULT_EXCLUDED_DIRS + (excluded_dirs or [])
    final_excluded_files = DEFAULT_EXCLUDED_FILES + (excluded_files or [])

    # Normalize exclusion patterns for reliable matching
    normalized_excluded_dirs = [p.strip('./').strip('/') for p in final_excluded_dirs]

    # Get filename patterns for exclusion from repo config
    file_filters_config = configs.get("file_filters", {})
    excluded_filename_patterns = file_filters_config.get("excluded_filename_patterns", [])

    logger.info(f"Starting to walk directory: {path}")
    
    # First pass: collect all candidate files with priorities
    for root, dirs, files in os.walk(path, topdown=True):
        # Filter directories in-place to prevent traversing them
        dirs[:] = [d for d in dirs if d not in normalized_excluded_dirs]
        current_dir_relative = os.path.relpath(root, path)
        
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, path)
            normalized_relative_path = relative_path.replace('\\', '/')

            # Skip excluded files
            if any(fnmatch.fnmatch(file, pattern) for pattern in excluded_filename_patterns):
                rejected_files_log.append(f"{normalized_relative_path} (matches filename pattern)")
                continue

            if any(fnmatch.fnmatch(normalized_relative_path, pattern) for pattern in final_excluded_files):
                rejected_files_log.append(f"{normalized_relative_path} (matches excluded file pattern)")
                continue
            
            if included_files and not any(fnmatch.fnmatch(normalized_relative_path, pattern) for pattern in included_files):
                rejected_files_log.append(f"{normalized_relative_path} (not in included files)")
                continue

            # Estimate file size and priority
            try:
                file_size = os.path.getsize(file_path)
                priority = estimate_processing_priority(normalized_relative_path)
                
                # Skip very large files early
                if file_size > 10 * 1024 * 1024:  # 10MB
                    rejected_files_log.append(f"{normalized_relative_path} (file too large: {file_size} bytes)")
                    continue
                
                file_candidates.append({
                    'path': file_path,
                    'relative_path': relative_path,
                    'normalized_path': normalized_relative_path,
                    'size': file_size,
                    'priority': priority
                })
                
            except OSError as e:
                rejected_files_log.append(f"{normalized_relative_path} (file access error: {e})")
                continue

    # Sort files by priority if enabled
    if prioritize_files:
        file_candidates.sort(key=lambda x: (x['priority'], x['size']))
        logger.info(f"Processing {len(file_candidates)} files in priority order")
    
    # Second pass: process files with token budget management
    total_tokens_processed = 0
    files_processed = 0
    
    for file_info in file_candidates:
        if total_tokens_processed >= max_total_tokens and files_processed >= 100:
            logger.warning(f"Reached token limit ({max_total_tokens}) or file limit. Stopping processing.")
            break
            
        try:
            content = get_local_file_content(file_info['path'])
            if not content.strip():
                rejected_files_log.append(f"{file_info['normalized_path']} (empty content)")
                continue

            # Estimate content tokens
            content_tokens = count_tokens(content)
            
            # Skip or truncate very large files
            if content_tokens > 50000:  # Very large file
                if file_info['priority'] <= 2:  # Only process if high priority
                    logger.info(f"Large file ({content_tokens} tokens): {file_info['normalized_path']}, using smart chunking")
                    # Use smart chunking for large files
                    smart_chunks = smart_chunk_text(content, max_tokens=2000, overlap_tokens=100)
                    content = '\n\n'.join(smart_chunks[:3])  # Take first 3 chunks only
                    content_tokens = count_tokens(content)
                else:
                    rejected_files_log.append(f"{file_info['normalized_path']} (too large and low priority)")
                    continue
            
            # Check if we can afford to process this file
            if total_tokens_processed + content_tokens > max_total_tokens:
                if files_processed < 50:  # If we haven't processed many files yet, try smaller chunk
                    # Truncate content to fit remaining budget
                    remaining_budget = max_total_tokens - total_tokens_processed
                    if remaining_budget > 1000:  # Only if meaningful budget remains
                        ratio = remaining_budget / content_tokens
                        content = content[:int(len(content) * ratio)]
                        content_tokens = remaining_budget
                    else:
                        break
                else:
                    break

            processed_files_log.append(file_info['normalized_path'])
            
            # Process chunks
            chunks = text_splitter.split_text(content)
            for i, chunk_content in enumerate(chunks):
                doc = Document(text=chunk_content)
                doc.metadata = {
                    "source": file_info['relative_path'],
                    "priority": file_info['priority'],
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
                all_documents.append(doc)
            
            total_tokens_processed += content_tokens
            files_processed += 1
            
            # Progress logging for large repos
            if files_processed % 100 == 0:
                logger.info(f"Processed {files_processed} files, {total_tokens_processed} tokens")

        except Exception as e:
            rejected_files_log.append(f"{file_info['normalized_path']} (error: {e})")
            logger.warning(f"Could not process file {file_info['path']}: {e}")

    logger.info(f"Finished processing repository: {path}")
    logger.info(f"Processed {len(processed_files_log)} files ({total_tokens_processed} tokens).")
    logger.info(f"Rejected {len(rejected_files_log)} files.")
    logger.info(f"Total documents after chunking: {len(all_documents)}")
    
    # Detailed logging for transparency
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Processed files: {json.dumps(processed_files_log, indent=2)}")
        logger.debug(f"Rejected files: {json.dumps(rejected_files_log, indent=2)}")

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

def transform_documents_and_save_to_db(documents: List[Document], db_path: str, is_ollama_embedder: bool = None) -> List[Document]:
    """
    Transforms documents using the data pipeline and saves them to a pickle file.
    """
    if not documents:
        logger.warning("No documents to process.")
        return []

    pipeline = prepare_data_pipeline(is_ollama_embedder)
    transformed_docs = pipeline(documents)

    if not transformed_docs:
        logger.error("Transformation pipeline returned no documents.")
        return []

    # Save transformed documents to a pickle file
    with open(db_path, 'wb') as f:
        pickle.dump(transformed_docs, f)
    logger.info(f"Saved {len(transformed_docs)} transformed documents to {db_path}")
    
    return transformed_docs

class DatabaseManager:
    """
    Manages the lifecycle of creating and loading the vector database for a repository.
    This class is designed to be instantiated fresh for each request to ensure no stale state.
    """
    def __init__(self):
        self.db_docs: Optional[List[Document]] = None

    def prepare_database(self, repo_url_or_path: str, type: str = "github", access_token: str = None, is_ollama_embedder: bool = None,
                       excluded_dirs: List[str] = None, excluded_files: List[str] = None,
                       included_dirs: List[str] = None, included_files: List[str] = None,
                       max_total_tokens: int = 1000000, prioritize_files: bool = True) -> List[Document]:
        """
        Main method to prepare the database. It handles cloning, loading from cache,
        or processing and saving documents.
        """
        repo_name = os.path.basename(repo_url_or_path.rstrip('/'))
        db_path = os.path.join(get_adalflow_default_root_path(), "databases", f"{repo_name}.pkl")

        if os.path.exists(db_path):
            logger.info(f"Loading database from existing path: {db_path}")
            try:
                with open(db_path, 'rb') as f:
                    loaded_docs = pickle.load(f)
                
                # Defensive check for stale cache format
                if isinstance(loaded_docs, list):
                    self.db_docs = loaded_docs
                    logger.info(f"Successfully loaded {len(self.db_docs)} documents from cache.")
                    return self.db_docs
                else:
                    logger.warning(f"Cache file {db_path} contains an outdated format. Regenerating database.")
                    self.db_docs = None # Invalidate cache
            except (pickle.UnpicklingError, EOFError, ModuleNotFoundError) as e:
                logger.warning(f"Could not unpickle cache file {db_path} due to '{e}'. Regenerating database.")
                self.db_docs = None # Invalidate cache

        repo_path = repo_url_or_path if type == "local" else os.path.join(get_adalflow_default_root_path(), "repos", repo_name)
        if type != "local":
            download_repo(repo_url_or_path, repo_path, type, access_token)

        documents = read_all_documents(
            repo_path, is_ollama_embedder, excluded_dirs, excluded_files, included_dirs, included_files,
            max_total_tokens, prioritize_files
        )
        
        if not documents:
            logger.warning("No documents were read from the repository.")
            return []

        self.db_docs = transform_documents_and_save_to_db(documents, db_path, is_ollama_embedder)
        
        return self.db_docs