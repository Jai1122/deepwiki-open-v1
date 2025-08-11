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
from urllib.parse import urlparse, urlunparse, quote
import requests
from requests.exceptions import RequestException

from .tools.embedder import get_embedder
from .utils import get_local_file_content, count_tokens, estimate_processing_priority, smart_chunk_text, is_text_file

logger = logging.getLogger(__name__)

def safe_chunk_for_embedding(text: str, max_tokens: int = 4000) -> List[str]:
    """
    Safely chunk text to ensure no chunk exceeds the embedding model's token limit.
    Uses recursive splitting if needed.
    """
    if not text or not text.strip():
        return []
    
    # Quick check - if text is small enough, return as-is
    token_count = count_tokens(text)
    if token_count <= max_tokens:
        return [text]
    
    logger.debug(f"Text has {token_count} tokens, splitting to fit {max_tokens} token limit")
    
    # Use smart chunking to split the text
    chunks = smart_chunk_text(text, max_tokens=max_tokens, overlap_tokens=min(200, max_tokens//10))
    
    # Validate each chunk and recursively split if needed
    safe_chunks = []
    for chunk in chunks:
        chunk_tokens = count_tokens(chunk)
        if chunk_tokens <= max_tokens:
            safe_chunks.append(chunk)
        else:
            # Chunk is still too large, split more aggressively
            logger.warning(f"Chunk still too large ({chunk_tokens} tokens), splitting more aggressively")
            
            # Split by sentences, then by lines if needed
            sentences = chunk.split('. ')
            current_chunk = []
            current_tokens = 0
            
            for sentence in sentences:
                sentence_tokens = count_tokens(sentence + '. ')
                
                if current_tokens + sentence_tokens <= max_tokens:
                    current_chunk.append(sentence)
                    current_tokens += sentence_tokens
                else:
                    # Finalize current chunk
                    if current_chunk:
                        safe_chunks.append('. '.join(current_chunk) + '.')
                    
                    # Start new chunk
                    if sentence_tokens <= max_tokens:
                        current_chunk = [sentence]
                        current_tokens = sentence_tokens
                    else:
                        # Even individual sentence is too large, truncate it
                        logger.warning(f"Individual sentence too large ({sentence_tokens} tokens), truncating")
                        truncated = sentence[:int(len(sentence) * (max_tokens / sentence_tokens))]
                        safe_chunks.append(truncated + '...')
                        current_chunk = []
                        current_tokens = 0
            
            # Add final chunk if any
            if current_chunk:
                safe_chunks.append('. '.join(current_chunk) + '.')
    
    # Final validation
    validated_chunks = []
    for chunk in safe_chunks:
        chunk_tokens = count_tokens(chunk)
        if chunk_tokens <= max_tokens:
            validated_chunks.append(chunk)
        else:
            logger.error(f"Chunk still exceeds token limit after aggressive splitting ({chunk_tokens} tokens), truncating")
            # Last resort: character-based truncation
            ratio = max_tokens / chunk_tokens
            truncated = chunk[:int(len(chunk) * ratio * 0.9)]  # 0.9 for safety margin
            validated_chunks.append(truncated + '...')
    
    logger.debug(f"Split {token_count} tokens into {len(validated_chunks)} safe chunks")
    return validated_chunks

def download_repo(repo_url: str, local_path: str, type: str = "github", access_token: str = None) -> str:
    """
    Clones a repository from GitHub, GitLab, or Bitbucket.
    Handles authentication for private repositories.
    
    Args:
        repo_url: The repository URL to clone
        local_path: Local path where repository will be cloned
        type: Repository type ("github", "gitlab", "bitbucket")
        access_token: Authentication token
            - GitHub: Personal Access Token
            - GitLab: Personal Access Token
            - Bitbucket: HTTP Access Token or username:app_password
    """
    if os.path.exists(local_path):
        logger.info(f"Repository already exists at {local_path}, skipping download.")
        return local_path

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    # Normalize repository URL for git clone
    normalized_repo_url = repo_url
    if type == "bitbucket":
        # Ensure Bitbucket URL ends with .git for git clone operations
        if not repo_url.endswith('.git'):
            normalized_repo_url = repo_url.rstrip('/') + '.git'
    
    # Prepare list of clone URLs to try
    clone_urls = [normalized_repo_url]  # Start with normalized URL for public repos
    
    if type in ["github", "gitlab", "bitbucket"] and access_token:
        parsed_url = urlparse(normalized_repo_url)
        if type == "github":
            # URL encode the token to handle special characters
            encoded_token = quote(access_token, safe='')
            clone_urls = [f"https://{encoded_token}@{parsed_url.netloc}{parsed_url.path}"]
        elif type == "gitlab":
            # URL encode the token to handle special characters
            encoded_token = quote(access_token, safe='')
            clone_urls = [f"https://oauth2:{encoded_token}@{parsed_url.netloc}{parsed_url.path}"]
        elif type == "bitbucket":
            # For Bitbucket, try multiple auth methods
            encoded_token = quote(access_token, safe='')
            clone_urls = []
            
            if ':' in access_token:
                # Assume it's username:app_password format
                username, password = access_token.split(':', 1)
                encoded_username = quote(username, safe='')
                encoded_password = quote(password, safe='')
                clone_urls.append(f"https://{encoded_username}:{encoded_password}@{parsed_url.netloc}{parsed_url.path}")
                
                # Also try with x-token-auth in case it's actually an HTTP token with colon
                clone_urls.append(f"https://x-token-auth:{encoded_token}@{parsed_url.netloc}{parsed_url.path}")
            else:
                # HTTP access token - try multiple username formats
                clone_urls.append(f"https://x-token-auth:{encoded_token}@{parsed_url.netloc}{parsed_url.path}")
                # Some repos may work with the workspace name as username
                path_parts = parsed_url.path.strip('/').split('/')
                if len(path_parts) >= 1:
                    workspace = path_parts[0]
                    clone_urls.append(f"https://{workspace}:{encoded_token}@{parsed_url.netloc}{parsed_url.path}")

    logger.info(f"Cloning repository from {repo_url} to {local_path}...")
    
    last_error = None
    for i, clone_url in enumerate(clone_urls):
        try:
            logger.debug(f"Attempting clone method {i+1}/{len(clone_urls)}")
            subprocess.run(
                ["git", "clone", "--depth", "1", clone_url, local_path],
                check=True,
                capture_output=True,
                text=True
            )
            logger.info("Repository cloned successfully.")
            return local_path
        except subprocess.CalledProcessError as e:
            last_error = e
            error_output = e.stderr or e.stdout or ""
            logger.debug(f"Clone attempt {i+1} failed: {error_output}")
            
            # If it's not an auth failure, don't try other methods
            if not ("Authentication failed" in error_output or 
                   "invalid username or password" in error_output.lower() or
                   "could not read Username" in error_output):
                break
                
            # If this was the last attempt, we'll handle the error below
            if i == len(clone_urls) - 1:
                break
    
    # If we reach here, all attempts failed. Handle the error from the last attempt
    if last_error:
        error_output = last_error.stderr or last_error.stdout or ""
        
        # Provide specific error messages for common issues
        if "Authentication failed" in error_output or "invalid username or password" in error_output.lower():
            if type == "bitbucket":
                error_message = f"""Authentication failed for Bitbucket repository. Please check your credentials:

1. For HTTP Access Token: Use the token directly as provided
2. For App Password: Use format 'username:app_password'

Make sure your token/password has Repository read permissions.
Error details: {error_output}"""
            else:
                error_message = f"Authentication failed for {type} repository. Please check your access token. Error details: {error_output}"
        elif "could not resolve host" in error_output.lower() or "name resolution" in error_output.lower():
            error_message = f"Network error: Could not resolve repository host. Please check your internet connection and repository URL. Error details: {error_output}"
        elif "repository not found" in error_output.lower() or "does not exist" in error_output.lower():
            error_message = f"Repository not found. Please check the repository URL and ensure you have access to it. Error details: {error_output}"
        elif "unable to update url base from redirection" in error_output.lower():
            error_message = f"Redirection error encountered. This may be due to special characters in credentials or repository access issues. Please verify your credentials and repository URL. Error details: {error_output}"
        else:
            error_message = f"Failed to clone repository from {repo_url}. Error details: {error_output}"
        
        logger.error(error_message)
        raise RuntimeError(error_message)

def read_all_documents(
    path: str,
    max_total_tokens: int = 1000000,  # Max tokens to process across all files
    prioritize_files: bool = True      # Whether to prioritize important files
) -> List[Document]:
    """
    Reads all files from a directory, splits them into chunks, and returns a list of Document objects.
    Enhanced with smart processing for large repositories.
    """
    splitter_config = configs.get("text_splitter", {})
    
    # Use token-based chunking to ensure embedding limits are respected
    # Get max tokens from embedding config to ensure chunks fit within limits
    max_embedding_tokens = splitter_config.get("max_tokens_per_chunk", 4000)
    safe_chunk_size = min(max_embedding_tokens // 2, 2000)  # Use half the limit for safety
    chunk_overlap = min(splitter_config.get("chunk_overlap", 200), safe_chunk_size // 4)
    
    text_splitter = TextSplitter(
        split_by="token",  # Use token-based splitting for precise control
        chunk_size=safe_chunk_size,
        chunk_overlap=chunk_overlap,
    )

    all_documents = []
    processed_files_log = []
    rejected_files_log = []
    file_candidates = []  # Store files with their priorities for processing
    
    final_excluded_dirs = DEFAULT_EXCLUDED_DIRS
    final_excluded_files = DEFAULT_EXCLUDED_FILES
    
    # Get filename patterns for exclusion from repo config
    file_filters_config = configs.get("file_filters", {})
    excluded_filename_patterns = file_filters_config.get("excluded_filename_patterns", [])
    
    # Also add excluded_dirs from repo config if not already included
    repo_excluded_dirs = file_filters_config.get("excluded_dirs", [])
    final_excluded_dirs.extend(repo_excluded_dirs)

    # Normalize exclusion patterns for reliable matching
    # Handle different formats: "./dir/", "dir", ".git", etc.
    normalized_excluded_dirs = []
    for p in final_excluded_dirs:
        if not p or not p.strip():
            continue
        # Remove leading ./ and trailing /
        normalized = p.strip()
        if normalized.startswith('./'):
            normalized = normalized[2:]
        normalized = normalized.rstrip('/')
        if normalized and normalized not in normalized_excluded_dirs:
            normalized_excluded_dirs.append(normalized)

    logger.info(f"Starting to walk directory: {path}")
    logger.info(f"Excluded directories ({len(normalized_excluded_dirs)}): {normalized_excluded_dirs}")
    logger.info(f"Excluded filename patterns: {excluded_filename_patterns}")
    logger.info(f"Total excluded file patterns: {len(final_excluded_files)}")
    
    # Debug: Check if critical directories are in the exclusion list
    critical_dirs = ['.git', 'vendor', 'node_modules', '__pycache__']
    for critical_dir in critical_dirs:
        if critical_dir in normalized_excluded_dirs:
            logger.info(f"✅ '{critical_dir}' directory is in exclusion list")
        else:
            logger.warning(f"❌ '{critical_dir}' directory is NOT in exclusion list!")
    
    logger.debug(f"All normalized excluded dirs: {sorted(normalized_excluded_dirs)}")
    logger.debug(f"Repo config dirs: {repo_excluded_dirs}")
    
    # First pass: collect all candidate files with priorities
    for root, dirs, files in os.walk(path, topdown=True):
        # Get current directory relative to the base path
        current_dir_relative = os.path.relpath(root, path) 
        normalized_current_dir = current_dir_relative.replace('\\', '/')
        
        # Handle root directory case
        if normalized_current_dir == '.':
            normalized_current_dir = ''
        
        # CRITICAL: Check if current directory should be entirely skipped
        should_skip_current_dir = False
        excluded_component = None
        
        if normalized_current_dir:
            # Check each component of the current path
            path_components = normalized_current_dir.split('/')
            logger.debug(f"Checking path components for '{normalized_current_dir}': {path_components}")
            
            for component in path_components:
                if component in normalized_excluded_dirs:
                    should_skip_current_dir = True
                    excluded_component = component
                    logger.info(f"🚫 SKIPPING ENTIRE TREE '{normalized_current_dir}' - component '{component}' is excluded")
                    break
        
        # If current directory should be skipped, skip it completely
        if should_skip_current_dir:
            dirs.clear()  # Critical: prevent any recursion into this tree
            continue  # Skip processing files in this directory too
        
        # DOUBLE CHECK: Filter immediate subdirectories as additional safety
        original_subdirs = dirs[:]
        dirs[:] = []
        
        for subdir in original_subdirs:
            if subdir in normalized_excluded_dirs:
                logger.info(f"🚫 Excluding subdirectory '{subdir}' from '{normalized_current_dir or 'root'}'")
            else:
                dirs.append(subdir)
        
        # Log directory processing status
        if original_subdirs != dirs:
            filtered_subdirs = set(original_subdirs) - set(dirs)
            logger.info(f"🚫 Filtered {len(filtered_subdirs)} subdirs from '{normalized_current_dir or 'root'}': {filtered_subdirs}")
        
        if dirs:
            logger.debug(f"📂 Processing '{normalized_current_dir or 'root'}' - will recurse into: {dirs}")
        else:
            logger.debug(f"📂 Processing '{normalized_current_dir or 'root'}' - no subdirs to recurse")
        
        # SAFETY CHECK: Verify we're not in an excluded directory
        if normalized_current_dir:
            path_components = normalized_current_dir.split('/')
            for component in path_components:
                if component in normalized_excluded_dirs:
                    logger.error(f"🚨 ERROR: Processing files in excluded directory '{normalized_current_dir}' - this should not happen!")
                    continue  # Skip processing files in this directory
        
        # Log files being considered in this directory
        if files:
            logger.debug(f"📄 Found {len(files)} files in '{normalized_current_dir or 'root'}': {files[:5]}{'...' if len(files) > 5 else ''}")
        
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, path)
            normalized_relative_path = relative_path.replace('\\', '/')

            # FINAL SAFETY CHECK: Ensure we're not processing files from excluded directories
            file_path_components = normalized_relative_path.split('/')
            file_in_excluded_dir = False
            for component in file_path_components[:-1]:  # Exclude the filename itself
                if component in normalized_excluded_dirs:
                    rejected_files_log.append(f"{normalized_relative_path} (in excluded directory: {component})")
                    file_in_excluded_dir = True
                    break
            
            if file_in_excluded_dir:
                continue
            
            # Check if file is binary/non-text to prevent junk characters in embeddings
            if not is_text_file(file_path):
                rejected_files_log.append(f"{normalized_relative_path} (binary/non-text file)")
                continue
            
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
            
            # Skip included_files filtering - using default filtering only

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
            if not content or not content.strip():
                rejected_files_log.append(f"{file_info['normalized_path']} (empty content)")
                continue
            
            # CRITICAL: Final check for null bytes or binary content in the pipeline
            if '\x00' in content or '\0' in content:
                rejected_files_log.append(f"{file_info['normalized_path']} (contains null bytes - binary content)")
                logger.warning(f"PIPELINE SAFETY: Found null bytes in content from {file_info['path']}")
                continue
            
            # Check for other indicators of binary content
            control_char_count = sum(1 for c in content if ord(c) < 32 and c not in '\n\r\t')
            if len(content) > 0 and control_char_count / len(content) > 0.05:
                rejected_files_log.append(f"{file_info['normalized_path']} (excessive control characters: {control_char_count})")
                logger.debug(f"Rejected {file_info['path']} due to {control_char_count} control characters in {len(content)} total")
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
            
            # Process chunks with token validation
            chunks = text_splitter.split_text(content)
            
            # Get max tokens from config for embedding safety
            max_embedding_tokens = configs.get("text_splitter", {}).get("max_tokens_per_chunk", 4000)
            
            safe_chunks = []
            for chunk in chunks:
                # Validate and potentially re-chunk if too large for embedding
                chunk_tokens = count_tokens(chunk)
                if chunk_tokens > max_embedding_tokens:
                    logger.warning(f"Chunk from {file_info['normalized_path']} has {chunk_tokens} tokens, splitting further")
                    sub_chunks = safe_chunk_for_embedding(chunk, max_embedding_tokens)
                    safe_chunks.extend(sub_chunks)
                else:
                    safe_chunks.append(chunk)
            
            # Create documents from safe chunks
            for i, chunk_content in enumerate(safe_chunks):
                # CRITICAL: Final token validation before creating document
                final_token_count = count_tokens(chunk_content)
                if final_token_count > max_embedding_tokens:
                    logger.error(f"CRITICAL: Chunk from {file_info['normalized_path']} exceeds token limit ({final_token_count} > {max_embedding_tokens} tokens)")
                    logger.error(f"Chunk preview: {repr(chunk_content[:200])}...")
                    logger.error("This chunk will be rejected to prevent embedding API errors")
                    continue
                
                # Additional safety check for very large chunks that might cause issues
                if final_token_count > 8000:  # Well below the 8194 limit but flag for attention
                    logger.warning(f"Large chunk detected ({final_token_count} tokens) from {file_info['normalized_path']}")
                    logger.warning("This is within limits but may cause issues if batched with other chunks")
                
                # ABSOLUTE FINAL CHECK: No null bytes in document content
                if '\x00' in chunk_content or '\0' in chunk_content:
                    logger.error(f"CRITICAL: Null bytes found in chunk content from {file_info['relative_path']}")
                    logger.error(f"Chunk preview: {repr(chunk_content[:100])}")
                    continue
                    
                doc = Document(text=chunk_content)
                doc.metadata = {
                    "source": file_info['relative_path'],
                    "priority": file_info['priority'],
                    "chunk_index": i,
                    "total_chunks": len(safe_chunks),
                    "token_count": final_token_count
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
        
        # Count rejection reasons
        rejection_reasons = {}
        for rejected in rejected_files_log:
            if '(' in rejected:
                reason = rejected.split('(')[-1].rstrip(')')
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
        
        logger.info(f"Rejection reasons summary: {dict(rejection_reasons)}")
    
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
                    
                    # Emergency recovery also needs null byte protection
                    if not content or not content.strip():
                        continue
                    
                    if '\x00' in content or '\0' in content:
                        logger.warning(f"EMERGENCY RECOVERY: Skipping file with null bytes: {file_info['path']}")
                        continue
                    
                    if content.strip():
                        chunks = text_splitter.split_text(content)
                        max_embedding_tokens = configs.get("text_splitter", {}).get("max_tokens_per_chunk", 4000)
                        
                        safe_chunks = []
                        for chunk in chunks:
                            chunk_tokens = count_tokens(chunk)
                            if chunk_tokens > max_embedding_tokens:
                                sub_chunks = safe_chunk_for_embedding(chunk, max_embedding_tokens)
                                safe_chunks.extend(sub_chunks)
                            else:
                                safe_chunks.append(chunk)
                        
                        for i, chunk_content in enumerate(safe_chunks):
                            # Final validation
                            if count_tokens(chunk_content) <= max_embedding_tokens:
                                doc = Document(text=chunk_content)
                                doc.metadata = {
                                    "source": file_info['relative_path'],
                                    "priority": file_info['priority'],
                                    "chunk_index": i,
                                    "total_chunks": len(safe_chunks),
                                    "emergency_recovery": True,
                                    "token_count": count_tokens(chunk_content)
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



def prepare_data_pipeline():
    """
    Prepares the data processing pipeline with an embedder.
    """
    embedder = get_embedder()
    
    # Standard pipeline for vLLM embedder
    return ToEmbeddings(
        embedder=embedder,
        batch_size=configs.get("embedder", {}).get("batch_size", 10)
    )

def transform_documents_and_save_to_db(documents: List[Document], db_path: str) -> List[Document]:
    """
    Transforms documents using the data pipeline and saves them to a pickle file.
    """
    if not documents:
        logger.warning("No documents to process.")
        return []

    pipeline = prepare_data_pipeline()
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

    def prepare_database(self, repo_url_or_path: str, type: str = "github", access_token: str = None,
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
                    
                    # Also validate content diversity - ensure we have actual source code files
                    content_valid = self._validate_cache_content(loaded_docs)
                    
                    if cache_valid and content_valid:
                        self.db_docs = loaded_docs
                        logger.info(f"Successfully loaded {len(self.db_docs)} documents from cache.")
                        return self.db_docs
                    else:
                        if not cache_valid:
                            logger.warning(f"Cache contains inconsistent embedding dimensions. Regenerating database.")
                        if not content_valid:
                            logger.warning(f"Cache appears to contain insufficient source code content. Regenerating database.")
                        self._clear_cache(db_path)
                else:
                    logger.warning(f"Cache file {db_path} contains an outdated format. Regenerating database.")
                    self._clear_cache(db_path)
            except (pickle.UnpicklingError, EOFError, ModuleNotFoundError) as e:
                logger.warning(f"Could not unpickle cache file {db_path} due to '{e}'. Regenerating database.")
                self._clear_cache(db_path)

        # If we reach here, we need to process the repository from scratch
        logger.info(f"Processing repository: {repo_url_or_path}")

        repo_path = repo_url_or_path if type == "local" else os.path.join(get_adalflow_default_root_path(), "repos", repo_name)
        
        try:
            if type != "local":
                download_repo(repo_url_or_path, repo_path, type, access_token)

            documents = read_all_documents(
                repo_path, max_total_tokens, prioritize_files
            )
            
            if not documents:
                logger.warning("No documents were read from the repository.")
                self.db_docs = []
                return self.db_docs

            self.db_docs = transform_documents_and_save_to_db(documents, db_path)
            
            if not self.db_docs:
                logger.error("Failed to transform and save documents.")
                self.db_docs = []
                return self.db_docs
            
            logger.info(f"Successfully processed repository with {len(self.db_docs)} documents.")
            return self.db_docs
            
        except Exception as e:
            logger.error(f"Error processing repository {repo_url_or_path}: {str(e)}")
            self.db_docs = []
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
    
    def _validate_cache_content(self, docs: List[Document]) -> bool:
        """
        Validate that cached documents contain sufficient source code content,
        not just documentation files.
        """
        if not docs:
            return False
            
        # Analyze the source files in the cache
        source_files = set()
        for doc in docs:
            if hasattr(doc, 'metadata') and doc.metadata:
                source = doc.metadata.get('source', '')
                if source:
                    source_files.add(source)
        
        # Count source code vs documentation files
        source_code_files = [f for f in source_files if any(f.endswith(ext) for ext in ['.py', '.js', '.ts', '.go', '.java', '.cpp', '.c', '.h', '.json', '.yaml', '.yml'])]
        doc_files = [f for f in source_files if any(f.lower().endswith(ext) for ext in ['.md', '.txt', '.rst'])]
        
        logger.info(f"📊 Cache content analysis: {len(source_code_files)} source files, {len(doc_files)} doc files, {len(source_files)} total")
        
        # Cache is valid if we have a reasonable number of source code files
        # Minimum requirement: at least 3 source code files OR more source files than doc files
        has_sufficient_source = len(source_code_files) >= 3 or (len(source_code_files) > len(doc_files) and len(source_code_files) > 0)
        
        if has_sufficient_source:
            logger.info(f"✅ Cache content validation passed: {len(source_code_files)} source code files found")
            return True
        else:
            logger.warning(f"❌ Cache content validation failed: Only {len(source_code_files)} source files vs {len(doc_files)} doc files")
            logger.warning(f"📁 Source files in cache: {source_code_files[:5]}{'...' if len(source_code_files) > 5 else ''}")
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
    
    def get_documents(self) -> Optional[List[Document]]:
        """
        Get the current loaded documents.
        
        Returns:
            The list of loaded documents, or None if no documents are loaded.
        """
        return self.db_docs
    
    def clear_documents(self):
        """
        Clear the currently loaded documents from memory.
        """
        self.db_docs = None
        logger.info("Cleared documents from memory.")