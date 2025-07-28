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
    
    # Get filename patterns for exclusion from repo config
    file_filters_config = configs.get("file_filters", {})
    excluded_filename_patterns = file_filters_config.get("excluded_filename_patterns", [])
    
    # Also add excluded_dirs from repo config if not already included
    repo_excluded_dirs = file_filters_config.get("excluded_dirs", [])
    final_excluded_dirs.extend(repo_excluded_dirs)

    # Normalize exclusion patterns for reliable matching
    normalized_excluded_dirs = [p.strip('./').strip('/') for p in final_excluded_dirs if p.strip('./').strip('/')]

    logger.info(f"Starting to walk directory: {path}")
    logger.info(f"Excluded directories: {normalized_excluded_dirs}")
    logger.info(f"Excluded filename patterns: {excluded_filename_patterns}")
    logger.info(f"Total excluded file patterns: {len(final_excluded_files)}")
    
    # First pass: collect all candidate files with priorities
    for root, dirs, files in os.walk(path, topdown=True):
        # Get current directory relative to the base path
        current_dir_relative = os.path.relpath(root, path) 
        normalized_current_dir = current_dir_relative.replace('\\', '/')
        
        # Skip the root directory case completely to avoid issues
        if normalized_current_dir == '.':
            normalized_current_dir = ''
        
        # Check if any part of the current path should cause us to skip this directory
        should_skip_dir = False
        if normalized_current_dir:
            # Split path and check each component
            path_components = normalized_current_dir.split('/')
            for component in path_components:
                if component in normalized_excluded_dirs:
                    should_skip_dir = True
                    logger.debug(f"Skipping directory '{normalized_current_dir}' due to excluded component '{component}'")
                    break
        
        if should_skip_dir:
            dirs.clear()  # Prevent recursion into this directory tree
            continue
        
        # Filter immediate subdirectories to prevent walking into them
        dirs[:] = [d for d in dirs if d not in normalized_excluded_dirs]
        
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, path)
            normalized_relative_path = relative_path.replace('\\', '/')

            # Skip excluded files by filename pattern (both filename and full path)
            filename_excluded = False
            matched_pattern = None
            for pattern in excluded_filename_patterns:
                # Check against both just the filename and the full relative path
                if fnmatch.fnmatch(file, pattern) or fnmatch.fnmatch(normalized_relative_path, pattern):
                    filename_excluded = True
                    matched_pattern = pattern
                    break
            
            if filename_excluded:
                rejected_files_log.append(f"{normalized_relative_path} (matches filename pattern: {matched_pattern})")
                continue

            # Skip files matching general excluded file patterns
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
    
    # Show some examples of processed and rejected files for debugging
    if processed_files_log:
        logger.info(f"Example processed files: {processed_files_log[:5]}")
    if rejected_files_log:
        logger.info(f"Example rejected files: {rejected_files_log[:5]}")
    
    # Critical check: if no documents were created, log this as an error
    if len(all_documents) == 0:
        logger.error("❌ NO DOCUMENTS WERE CREATED! This will cause the application to fail.")
        logger.error("This might be due to over-aggressive file filtering.")
        logger.error(f"Repository path: {path}")
        logger.error(f"Files found but rejected: {len(rejected_files_log)}")
        
        # In case of complete filtering failure, let's try to include some basic files
        logger.warning("🔧 Attempting emergency file recovery - including basic source files...")
        emergency_candidates = []
        
        # Walk again but with minimal filtering for emergency recovery
        for root, dirs, files in os.walk(path):
            # Only exclude the most critical directories (.git, node_modules, etc)
            critical_exclude = ['.git', 'node_modules', '__pycache__', '.venv', 'venv', 'dist', 'build']
            dirs[:] = [d for d in dirs if d not in critical_exclude]
            
            for file in files:
                # Only include common source files
                if any(file.endswith(ext) for ext in ['.py', '.js', '.ts', '.go', '.java', '.cpp', '.c', '.h', '.md', '.txt', '.json', '.yaml', '.yml']):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, path)
                    emergency_candidates.append({
                        'path': file_path,
                        'relative_path': relative_path,
                        'normalized_path': relative_path.replace('\\', '/'),
                        'size': os.path.getsize(file_path) if os.path.exists(file_path) else 0,
                        'priority': 2  # Medium priority
                    })
                    
                    if len(emergency_candidates) >= 50:  # Limit to prevent overload
                        break
            if len(emergency_candidates) >= 50:
                break
        
        # Process emergency candidates
        if emergency_candidates:
            logger.warning(f"🚑 Emergency recovery: Processing {len(emergency_candidates)} basic files")
            for file_info in emergency_candidates[:20]:  # Process only first 20
                try:
                    content = get_local_file_content(file_info['path'])
                    if content.strip():
                        chunks = text_splitter.split_text(content)
                        for i, chunk_content in enumerate(chunks):
                            doc = Document(text=chunk_content)
                            doc.metadata = {
                                "source": file_info['relative_path'],
                                "priority": file_info['priority'],
                                "chunk_index": i,
                                "total_chunks": len(chunks),
                                "emergency_recovery": True
                            }
                            all_documents.append(doc)
                except Exception as e:
                    logger.warning(f"Emergency recovery failed for {file_info['path']}: {e}")
            
            logger.warning(f"🚑 Emergency recovery completed: {len(all_documents)} documents created")
    
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
                    # Validate embedding dimensions in cached documents
                    cache_valid = self._validate_cache_dimensions(loaded_docs)
                    if cache_valid:
                        self.db_docs = loaded_docs
                        logger.info(f"Successfully loaded {len(self.db_docs)} documents from cache.")
                        return self.db_docs
                    else:
                        logger.warning(f"Cache contains inconsistent embedding dimensions. Regenerating database.")
                        self._clear_cache(db_path)
                else:
                    logger.warning(f"Cache file {db_path} contains an outdated format. Regenerating database.")
                    self._clear_cache(db_path)
            except (pickle.UnpicklingError, EOFError, ModuleNotFoundError) as e:
                logger.warning(f"Could not unpickle cache file {db_path} due to '{e}'. Regenerating database.")
                self._clear_cache(db_path)

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
    
    def _validate_cache_dimensions(self, docs: List[Document]) -> bool:
        """
        Validate that cached documents have consistent embedding dimensions
        and match the current embedding model configuration.
        """
        if not docs:
            return True
            
        # Get current embedding model dimensions from environment
        expected_dim = int(os.environ.get('EMBEDDING_DIMENSIONS', '1024'))
        
        # Check dimensions in cached documents
        found_dimensions = set()
        valid_docs = 0
        
        for doc in docs:
            if hasattr(doc, 'vector') and doc.vector:
                doc_dim = len(doc.vector)
                found_dimensions.add(doc_dim)
                valid_docs += 1
        
        logger.debug(f"Cache validation: Found {valid_docs} docs with embeddings")
        logger.debug(f"Cache dimensions found: {found_dimensions}")
        logger.debug(f"Expected dimensions: {expected_dim}")
        
        # Cache is valid if:
        # 1. All documents have consistent dimensions
        # 2. The dimensions match current model configuration
        if len(found_dimensions) == 1 and expected_dim in found_dimensions:
            logger.info(f"✅ Cache validation passed: {expected_dim}D embeddings")
            return True
        else:
            logger.warning(f"❌ Cache validation failed:")
            if len(found_dimensions) > 1:
                logger.warning(f"   - Inconsistent dimensions in cache: {found_dimensions}")
            if expected_dim not in found_dimensions:
                logger.warning(f"   - Cache dimensions {found_dimensions} don't match expected {expected_dim}")
            return False
    
    def _clear_cache(self, db_path: str):
        """Clear the cache file safely."""
        try:
            if os.path.exists(db_path):
                os.remove(db_path)
                logger.info(f"🗑️  Cache cleared: {db_path}")
        except Exception as e:
            logger.warning(f"Could not clear cache {db_path}: {e}")
        
        # Reset internal state
        self.db_docs = None