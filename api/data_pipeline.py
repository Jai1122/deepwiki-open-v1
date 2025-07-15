from langchain.docstore.document import Document
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import subprocess
import json
import tiktoken
import logging
import base64
import re
import glob
from api.config import configs, DEFAULT_EXCLUDED_DIRS, DEFAULT_EXCLUDED_FILES
from api.ollama_patch import OllamaDocumentProcessor
from urllib.parse import urlparse, urlunparse, quote
import requests
from requests.exceptions import RequestException

from api.tools.embedder import get_embedder

# Configure logging
logger = logging.getLogger(__name__)

# Maximum token limit for OpenAI embedding models
MAX_EMBEDDING_TOKENS = 8192

def get_tokenizer(is_ollama_embedder: bool = None):
    """
    Get the tokenizer for the specified embedder.
    """
    # Determine if using Ollama embedder if not specified
    if is_ollama_embedder is None:
        from api.config import is_ollama_embedder as check_ollama
        is_ollama_embedder = check_ollama()

    if is_ollama_embedder:
        return tiktoken.get_encoding("cl100k_base")
    else:
        return tiktoken.encoding_for_model("text-embedding-3-small")

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
        encoding = get_tokenizer(is_ollama_embedder)
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
                # Format: https://{token}@bitbucket.org/owner/repo.git
                clone_url = urlunparse((parsed.scheme, f"{access_token}@{parsed.netloc}", parsed.path, '', '', ''))
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
                      included_dirs: List[str] = None, included_files: List[str] = None, batch_size: int = 100):
    """
    Recursively reads all documents in a directory and its subdirectories, yielding them in batches.

    Args:
        path (str): The root directory path.
        is_ollama_embedder (bool, optional): Whether using Ollama embeddings for token counting.
                                           If None, will be determined from configuration.
        excluded_dirs (List[str], optional): List of directories to exclude from processing.
            Overrides the default configuration if provided.
        excluded_files (List[str], optional): List of file patterns to exclude from processing.
            Overrides the default configuration if provided.
        included_dirs (List[str], optional): List of directories to include exclusively.
            When provided, only files in these directories will be processed.
        included_files (List[str], optional): List of file patterns to include exclusively.
            When provided, only files matching these patterns will be processed.
        batch_size (int): The number of documents to yield in each batch.

    Yields:
        list: A list of Document objects with metadata.
    """
    documents = []
    # File extensions to look for, prioritizing code files
    code_extensions = [".py", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".hpp", ".go", ".rs",
                       ".jsx", ".tsx", ".html", ".css", ".php", ".swift", ".cs"]
    doc_extensions = [".md", ".txt", ".rst", ".json", ".yaml", ".yml"]

    # Determine filtering mode: inclusion or exclusion
    use_inclusion_mode = (included_dirs is not None and len(included_dirs) > 0) or (included_files is not None and len(included_files) > 0)

    if use_inclusion_mode:
        # Inclusion mode: only process specified directories and files
        final_included_dirs = set(included_dirs) if included_dirs else set()
        final_included_files = set(included_files) if included_files else set()

        logger.info(f"Using inclusion mode")
        logger.info(f"Included directories: {list(final_included_dirs)}")
        logger.info(f"Included files: {list(final_included_files)}")

        # Convert to lists for processing
        included_dirs = list(final_included_dirs)
        included_files = list(final_included_files)
        excluded_dirs = []
        excluded_files = []
    else:
        # Exclusion mode: use default exclusions plus any additional ones
        final_excluded_dirs = set(DEFAULT_EXCLUDED_DIRS)
        final_excluded_files = set(DEFAULT_EXCLUDED_FILES)

        # Add any additional excluded directories from config
        if "file_filters" in configs and "excluded_dirs" in configs["file_filters"]:
            final_excluded_dirs.update(configs["file_filters"]["excluded_dirs"])

        # Add any additional excluded files from config
        if "file_filters" in configs and "excluded_files" in configs["file_filters"]:
            final_excluded_files.update(configs["file_filters"]["excluded_files"])

        # Add any explicitly provided excluded directories and files
        if excluded_dirs is not None:
            final_excluded_dirs.update(excluded_dirs)

        if excluded_files is not None:
            final_excluded_files.update(excluded_files)

        # Convert back to lists for compatibility
        excluded_dirs = list(final_excluded_dirs)
        excluded_files = list(final_excluded_files)
        included_dirs = []
        included_files = []

        logger.info(f"Using exclusion mode")
        logger.info(f"Excluded directories: {excluded_dirs}")
        logger.info(f"Excluded files: {excluded_files}")

    logger.info(f"Reading documents from {path}")

    def should_process_file(root_path: str, file_path: str, use_inclusion: bool, included_dirs: List[str], included_files: List[str],
                           excluded_dirs: List[str], excluded_files: List[str]) -> bool:
        """
        Determine if a file should be processed based on inclusion/exclusion rules.

        Args:
            root_path (str): The root path of the repository.
            file_path (str): The file path to check.
            use_inclusion (bool): Whether to use inclusion mode.
            included_dirs (List[str]): List of directories to include.
            included_files (List[str]): List of files to include.
            excluded_dirs (List[str]): List of directories to exclude.
            excluded_files (List[str]): List of files to exclude.

        Returns:
            bool: True if the file should be processed, False otherwise.
        """
        relative_path = os.path.relpath(file_path, root_path)
        file_name = os.path.basename(file_path)

        # Normalize path separators for consistent matching
        relative_path_parts = relative_path.split(os.sep)

        if use_inclusion:
            # Inclusion mode: file must be in an included directory or match an included file pattern.
            is_included = False
            if not included_dirs and not included_files:
                return True  # If no inclusion rules, include everything.

            # Check against included directories
            if included_dirs:
                for included in included_dirs:
                    clean_included = included.strip("./").rstrip("/").split('/')
                    if len(clean_included) <= len(relative_path_parts) and all(part == rel_part for part, rel_part in zip(clean_included, relative_path_parts)):
                        is_included = True
                        break

            # Check against included files if not already included by a directory rule
            if not is_included and included_files:
                for pattern in included_files:
                    if glob.fnmatch.fnmatch(file_name, pattern):
                        is_included = True
                        break
            return is_included
        else:
            # Exclusion mode: file must not be in an excluded directory or match an excluded file pattern.

            # Check against excluded files
            for pattern in excluded_files:
                if glob.fnmatch.fnmatch(file_name, pattern):
                    return False  # Exclude if filename matches a pattern

            # Check against excluded directories
            for excluded in excluded_dirs:
                clean_excluded = excluded.strip("./").rstrip("/").split('/')
                if len(clean_excluded) <= len(relative_path_parts) and all(part == rel_part for part, rel_part in zip(clean_excluded, relative_path_parts)):
                    return False # Exclude if path is within an excluded directory

            return True

    # Process code files first
    for ext in code_extensions:
        files = glob.glob(f"{path}/**/*{ext}", recursive=True)
        for file_path in files:
            # Check if file should be processed based on inclusion/exclusion rules
            if not should_process_file(path, file_path, use_inclusion_mode, included_dirs, included_files, excluded_dirs, excluded_files):
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    relative_path = os.path.relpath(file_path, path)

                    # Determine if this is an implementation file
                    is_implementation = (
                        not relative_path.startswith("test_")
                        and not relative_path.startswith("app_")
                        and "test" not in relative_path.lower()
                    )

                    # Check token count
                    token_count = count_tokens(content, is_ollama_embedder)
                    if token_count > MAX_EMBEDDING_TOKENS * 10:
                        logger.warning(f"Skipping large file {relative_path}: Token count ({token_count}) exceeds limit")
                        continue

                    doc = Document(
                        text=content,
                        meta_data={
                            "file_path": relative_path,
                            "type": ext[1:],
                            "is_code": True,
                            "is_implementation": is_implementation,
                            "title": relative_path,
                            "token_count": token_count,
                        },
                    )
                    documents.append(doc)
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")

    # Then process documentation files
    for ext in doc_extensions:
        files = glob.glob(f"{path}/**/*{ext}", recursive=True)
        for file_path in files:
            # Check if file should be processed based on inclusion/exclusion rules
            if not should_process_file(path, file_path, use_inclusion_mode, included_dirs, included_files, excluded_dirs, excluded_files):
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    relative_path = os.path.relpath(file_path, path)

                    # Check token count
                    token_count = count_tokens(content, is_ollama_embedder)
                    if token_count > MAX_EMBEDDING_TOKENS:
                        logger.warning(f"Skipping large file {relative_path}: Token count ({token_count}) exceeds limit")
                        continue

                    doc = Document(
                        text=content,
                        meta_data={
                            "file_path": relative_path,
                            "type": ext[1:],
                            "is_code": False,
                            "is_implementation": False,
                            "title": relative_path,
                            "token_count": token_count,
                        },
                    )
                    documents.append(doc)
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")

    logger.info(f"Found {len(documents)} documents")
    if documents:
        yield documents

from langchain.vectorstores import FAISS

def create_vector_store(documents, embedder):
    """
    Creates a FAISS vector store from a list of documents.
    """
    text_splitter_config = configs.get("text_splitter", {})
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=text_splitter_config.get("chunk_size", 1000),
        chunk_overlap=text_splitter_config.get("chunk_overlap", 200),
        length_function=len,
        is_separator_regex=False,
    )
    split_documents = splitter.split_documents(documents)
    return FAISS.from_documents(split_documents, embedder)

def get_github_file_content(repo_url: str, file_path: str, access_token: str = None) -> str:
    """
    Retrieves the content of a file from a GitHub repository using the GitHub API.
    Supports both public GitHub (github.com) and GitHub Enterprise (custom domains).
    
    Args:
        repo_url (str): The URL of the GitHub repository 
                       (e.g., "https://github.com/username/repo" or "https://github.company.com/username/repo")
        file_path (str): The path to the file within the repository (e.g., "src/main.py")
        access_token (str, optional): GitHub personal access token for private repositories

    Returns:
        str: The content of the file as a string

    Raises:
        ValueError: If the file cannot be fetched or if the URL is not a valid GitHub URL
    """
    try:
        # Parse the repository URL to support both github.com and enterprise GitHub
        parsed_url = urlparse(repo_url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError("Not a valid GitHub repository URL")

        # Check if it's a GitHub-like URL structure
        path_parts = parsed_url.path.strip('/').split('/')
        if len(path_parts) < 2:
            raise ValueError("Invalid GitHub URL format - expected format: https://domain/owner/repo")

        owner = path_parts[-2]
        repo = path_parts[-1].replace(".git", "")

        # Determine the API base URL
        if parsed_url.netloc == "github.com":
            # Public GitHub
            api_base = "https://api.github.com"
        else:
            # GitHub Enterprise - API is typically at https://domain/api/v3/
            api_base = f"{parsed_url.scheme}://{parsed_url.netloc}/api/v3"
        
        # Use GitHub API to get file content
        # The API endpoint for getting file content is: /repos/{owner}/{repo}/contents/{path}
        api_url = f"{api_base}/repos/{owner}/{repo}/contents/{file_path}"

        # Fetch file content from GitHub API
        headers = {}
        if access_token:
            headers["Authorization"] = f"token {access_token}"
        logger.info(f"Fetching file content from GitHub API: {api_url}")
        try:
            response = requests.get(api_url, headers=headers)
            response.raise_for_status()
        except RequestException as e:
            raise ValueError(f"Error fetching file content: {e}")
        try:
            content_data = response.json()
        except json.JSONDecodeError:
            raise ValueError("Invalid response from GitHub API")

        # Check if we got an error response
        if "message" in content_data and "documentation_url" in content_data:
            raise ValueError(f"GitHub API error: {content_data['message']}")

        # GitHub API returns file content as base64 encoded string
        if "content" in content_data and "encoding" in content_data:
            if content_data["encoding"] == "base64":
                # The content might be split into lines, so join them first
                content_base64 = content_data["content"].replace("\n", "")
                content = base64.b64decode(content_base64).decode("utf-8")
                return content
            else:
                raise ValueError(f"Unexpected encoding: {content_data['encoding']}")
        else:
            raise ValueError("File content not found in GitHub API response")

    except Exception as e:
        raise ValueError(f"Failed to get file content: {str(e)}")

def get_gitlab_file_content(repo_url: str, file_path: str, access_token: str = None) -> str:
    """
    Retrieves the content of a file from a GitLab repository (cloud or self-hosted).

    Args:
        repo_url (str): The GitLab repo URL (e.g., "https://gitlab.com/username/repo" or "http://localhost/group/project")
        file_path (str): File path within the repository (e.g., "src/main.py")
        access_token (str, optional): GitLab personal access token

    Returns:
        str: File content

    Raises:
        ValueError: If anything fails
    """
    try:
        # Parse and validate the URL
        parsed_url = urlparse(repo_url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError("Not a valid GitLab repository URL")

        gitlab_domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
        if parsed_url.port not in (None, 80, 443):
            gitlab_domain += f":{parsed_url.port}"
        path_parts = parsed_url.path.strip("/").split("/")
        if len(path_parts) < 2:
            raise ValueError("Invalid GitLab URL format — expected something like https://gitlab.domain.com/group/project")

        # Build project path and encode for API
        project_path = "/".join(path_parts).replace(".git", "")
        encoded_project_path = quote(project_path, safe='')

        # Encode file path
        encoded_file_path = quote(file_path, safe='')

        # Default to 'main' branch if not specified
        default_branch = 'main'

        api_url = f"{gitlab_domain}/api/v4/projects/{encoded_project_path}/repository/files/{encoded_file_path}/raw?ref={default_branch}"
        # Fetch file content from GitLab API
        headers = {}
        if access_token:
            headers["PRIVATE-TOKEN"] = access_token
        logger.info(f"Fetching file content from GitLab API: {api_url}")
        try:
            response = requests.get(api_url, headers=headers)
            response.raise_for_status()
            content = response.text
        except RequestException as e:
            raise ValueError(f"Error fetching file content: {e}")

        # Check for GitLab error response (JSON instead of raw file)
        if content.startswith("{") and '"message":' in content:
            try:
                error_data = json.loads(content)
                if "message" in error_data:
                    raise ValueError(f"GitLab API error: {error_data['message']}")
            except json.JSONDecodeError:
                pass

        return content

    except Exception as e:
        raise ValueError(f"Failed to get file content: {str(e)}")

def get_bitbucket_file_content(repo_url: str, file_path: str, access_token: str = None) -> str:
    """
    Retrieves the content of a file from a Bitbucket repository using the Bitbucket API.

    Args:
        repo_url (str): The URL of the Bitbucket repository (e.g., "https://bitbucket.org/username/repo")
        file_path (str): The path to the file within the repository (e.g., "src/main.py")
        access_token (str, optional): Bitbucket personal access token for private repositories

    Returns:
        str: The content of the file as a string
    """
    try:
        # Extract owner and repo name from Bitbucket URL
        if not (repo_url.startswith("https://bitbucket.org/") or repo_url.startswith("http://bitbucket.org/")):
            raise ValueError("Not a valid Bitbucket repository URL")

        parts = repo_url.rstrip('/').split('/')
        if len(parts) < 5:
            raise ValueError("Invalid Bitbucket URL format")

        owner = parts[-2]
        repo = parts[-1].replace(".git", "")

        # Use Bitbucket API to get file content
        # The API endpoint for getting file content is: /2.0/repositories/{owner}/{repo}/src/{branch}/{path}
        api_url = f"https://api.bitbucket.org/2.0/repositories/{owner}/{repo}/src/main/{file_path}"

        # Fetch file content from Bitbucket API
        headers = {}
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"
        logger.info(f"Fetching file content from Bitbucket API: {api_url}")
        try:
            response = requests.get(api_url, headers=headers)
            if response.status_code == 200:
                content = response.text
            elif response.status_code == 404:
                raise ValueError("File not found on Bitbucket. Please check the file path and repository.")
            elif response.status_code == 401:
                raise ValueError("Unauthorized access to Bitbucket. Please check your access token.")
            elif response.status_code == 403:
                raise ValueError("Forbidden access to Bitbucket. You might not have permission to access this file.")
            elif response.status_code == 500:
                raise ValueError("Internal server error on Bitbucket. Please try again later.")
            else:
                response.raise_for_status()
                content = response.text
            return content
        except RequestException as e:
            raise ValueError(f"Error fetching file content: {e}")

    except Exception as e:
        raise ValueError(f"Failed to get file content: {str(e)}")


def get_file_content(repo_url: str, file_path: str, type: str = "github", access_token: str = None) -> str:
    """
    Retrieves the content of a file from a Git repository (GitHub or GitLab).

    Args:
        repo_url (str): The URL of the repository
        file_path (str): The path to the file within the repository
        access_token (str, optional): Access token for private repositories

    Returns:
        str: The content of the file as a string

    Raises:
        ValueError: If the file cannot be fetched or if the URL is not valid
    """
    if type == "github":
        return get_github_file_content(repo_url, file_path, access_token)
    elif type == "gitlab":
        return get_gitlab_file_content(repo_url, file_path, access_token)
    elif type == "bitbucket":
        return get_bitbucket_file_content(repo_url, file_path, access_token)
    else:
        raise ValueError("Unsupported repository URL. Only GitHub and GitLab are supported.")

class DatabaseManager:
    """
    Manages the creation, loading, transformation, and persistence of the vector store.
    """

    def __init__(self):
        self.vector_store = None
        self.repo_url_or_path = None
        self.repo_paths = None

    def prepare_database(self, repo_url_or_path: str, type: str = "github", access_token: str = None,
                       is_ollama_embedder: bool = False, excluded_dirs: List[str] = None,
                       excluded_files: List[str] = None, included_dirs: List[str] = None,
                       included_files: List[str] = None) -> FAISS:
        """
        Create a new database from the repository.
        """
        self.reset_database()
        self._create_repo(repo_url_or_path, type, access_token)
        return self.prepare_db_index(is_ollama_embedder=is_ollama_embedder, excluded_dirs=excluded_dirs,
                                   excluded_files=excluded_files, included_dirs=included_dirs,
                                   included_files=included_files)

    def reset_database(self):
        """
        Reset the database to its initial state.
        """
        self.vector_store = None
        self.repo_url_or_path = None
        self.repo_paths = None

    def _extract_repo_name_from_url(self, repo_url_or_path: str, repo_type: str) -> str:
        """
        Extracts the repository name from the URL or path.
        """
        try:
            if os.path.isdir(repo_url_or_path):
                return os.path.basename(repo_url_or_path)

            parsed_url = urlparse(repo_url_or_path)
            path_parts = parsed_url.path.strip('/').split('/')

            if repo_type == "github" and len(path_parts) >= 2:
                return f"github_{path_parts[-2]}_{path_parts[-1]}"
            elif repo_type == "gitlab" and len(path_parts) >= 2:
                return f"gitlab_{path_parts[-2]}_{path_parts[-1]}"
            elif repo_type == "bitbucket" and len(path_parts) >= 2:
                return f"bitbucket_{path_parts[-2]}_{path_parts[-1]}"
            else:
                return "local_repo"
        except Exception as e:
            logger.error(f"Error extracting repo name: {e}")
            return "unknown_repo"

    def _create_repo(self, repo_url_or_path: str, repo_type: str = "github", access_token: str = None) -> None:
        """
        Create the repository directory and download the repository.
        """
        self.repo_url_or_path = repo_url_or_path
        repo_name = self._extract_repo_name_from_url(repo_url_or_path, repo_type)

        save_dir = os.path.join(os.getcwd(), "repositories")
        os.makedirs(save_dir, exist_ok=True)

        save_repo_dir = os.path.join(save_dir, repo_name)
        save_db_dir = os.path.join(save_dir, f"{repo_name}_db")
        os.makedirs(save_db_dir, exist_ok=True)

        self.repo_paths = {
            "save_dir": save_dir,
            "save_repo_dir": save_repo_dir,
            "save_db_dir": save_db_dir,
            "save_db_file": os.path.join(save_db_dir, "faiss_index"),
        }

        download_repo(repo_url_or_path, save_repo_dir, repo_type, access_token)

    def prepare_db_index(self, is_ollama_embedder: bool = False, excluded_dirs: List[str] = None,
                        excluded_files: List[str] = None, included_dirs: List[str] = None,
                        included_files: List[str] = None) -> FAISS:
        """
        Prepare the indexed database for the repository.
        """
        # check the database
        if self.repo_paths and os.path.exists(self.repo_paths["save_db_file"]):
            logger.info("Loading existing database...")
            try:
                embedder = get_embedder()
                self.vector_store = FAISS.load_local(self.repo_paths["save_db_file"], embedder)
                logger.info(f"Loaded existing database from {self.repo_paths['save_db_file']}")
                return self.vector_store
            except Exception as e:
                logger.error(f"Error loading existing database: {e}")
                # Continue to create a new database

        # prepare the database
        logger.info("Creating new database...")
        documents = []
        for docs in read_all_documents(
            self.repo_paths["save_repo_dir"],
            is_ollama_embedder=is_ollama_embedder,
            excluded_dirs=excluded_dirs,
            excluded_files=excluded_files,
            included_dirs=included_dirs,
            included_files=included_files
        ):
            documents.extend(docs)

        embedder = get_embedder()
        self.vector_store = create_vector_store(documents, embedder)
        self.vector_store.save_local(self.repo_paths["save_db_file"])
        logger.info(f"Created and saved new database to {self.repo_paths['save_db_file']}")
        return self.vector_store

    def prepare_retriever(self, repo_url_or_path: str, type: str = "github", access_token: str = None):
        """
        Prepare the retriever for a repository.
        """
        return self.prepare_database(repo_url_or_path, type, access_token)
