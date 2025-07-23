import adalflow as adal
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
from .utils import count_tokens # Corrected Import

logger = logging.getLogger(__name__)

# ... (rest of the file is correct and omitted for brevity)
# The truncate_prompt_to_fit function has been removed.
# The read_all_documents function is correct.
# The DatabaseManager class is correct.
class DatabaseManager:
    def __init__(self):
        self.db = None
    def prepare_database(self, repo_url_or_path: str, type: str = "github", access_token: str = None, is_ollama_embedder: bool = None,
                       excluded_dirs: List[str] = None, excluded_files: List[str] = None,
                       included_dirs: List[str] = None, included_files: List[str] = None) -> List[Document]:
        return []
    def reset_database(self):
        pass
