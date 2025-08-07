import os
import logging
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from typing import List, Optional, Dict, Any, Literal
import json
from datetime import datetime
from pydantic import BaseModel, Field
import google.generativeai as genai
import asyncio

# Configure logging
from .logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


# Initialize FastAPI app
app = FastAPI(
    title="Streaming API",
    description="API for streaming chat completions"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Helper function to get adalflow root path
def get_adalflow_default_root_path():
    return os.path.expanduser(os.path.join("~", ".adalflow"))

# --- Pydantic Models ---
class WikiPage(BaseModel):
    """
    Model for a wiki page.
    """
    id: str
    title: str
    content: str
    filePaths: List[str]
    importance: str # Should ideally be Literal['high', 'medium', 'low']
    relatedPages: List[str]

class ProcessedProjectEntry(BaseModel):
    id: str  # Filename
    owner: str
    repo: str
    name: str  # owner/repo
    repo_type: str # Renamed from type to repo_type for clarity with existing models
    submittedAt: int # Timestamp

class RepoInfo(BaseModel):
    owner: str
    repo: str
    type: str
    token: Optional[str] = None
    localPath: Optional[str] = None
    repoUrl: Optional[str] = None


class WikiSection(BaseModel):
    """
    Model for the wiki sections.
    """
    id: str
    title: str
    pages: List[str]
    subsections: Optional[List[str]] = None


class WikiStructureModel(BaseModel):
    """
    Model for the overall wiki structure.
    """
    id: str
    title: str
    description: str
    pages: List[WikiPage]
    sections: Optional[List[WikiSection]] = None
    rootSections: Optional[List[str]] = None

class WikiCacheData(BaseModel):
    """
    Model for the data to be stored in the wiki cache.
    """
    wiki_structure: WikiStructureModel
    generated_pages: Dict[str, WikiPage]
    repo_url: Optional[str] = None  #compatible for old cache
    repo: Optional[RepoInfo] = None
    provider: Optional[str] = None
    model: Optional[str] = None

class WikiCacheRequest(BaseModel):
    """
    Model for the request body when saving wiki cache.
    """
    repo: RepoInfo
    wiki_structure: WikiStructureModel
    generated_pages: Dict[str, WikiPage]
    provider: str
    model: str

class WikiExportRequest(BaseModel):
    """
    Model for requesting a wiki export.
    """
    repo_url: str = Field(..., description="URL of the repository")
    pages: List[WikiPage] = Field(..., description="List of wiki pages to export")
    format: Literal["markdown", "json"] = Field(..., description="Export format (markdown or json)")

# --- Model Configuration Models ---
class Model(BaseModel):
    """
    Model for LLM model configuration
    """
    id: str = Field(..., description="Model identifier")
    name: str = Field(..., description="Display name for the model")

class Provider(BaseModel):
    """
    Model for LLM provider configuration
    """
    id: str = Field(..., description="Provider identifier")
    name: str = Field(..., description="Display name for the provider")
    models: List[Model] = Field(..., description="List of available models for this provider")
    supportsCustomModel: Optional[bool] = Field(False, description="Whether this provider supports custom models")

class ModelConfig(BaseModel):
    """
    Model for the entire model configuration
    """
    providers: List[Provider] = Field(..., description="List of available model providers")
    defaultProvider: str = Field(..., description="ID of the default provider")

class AuthorizationConfig(BaseModel):
    code: str = Field(..., description="Authorization code")

from .config import configs, WIKI_AUTH_MODE, WIKI_AUTH_CODE, embedder_config

@app.get("/lang/config")
async def get_lang_config():
    """
    Get language configuration. English-only support.
    """
    return {
        "supported_languages": {
            "en": "English"
        },
        "default": "en"
    }

@app.get("/auth/status")
async def get_auth_status():
    """
    Check if authentication is required for the wiki.
    """
    return {"auth_required": WIKI_AUTH_MODE}

@app.post("/auth/validate")
async def validate_auth_code(request: AuthorizationConfig):
    """
    Check authorization code.
    """
    return {"success": WIKI_AUTH_CODE == request.code}

@app.post("/models/update")
async def update_model_selection(request: dict):
    """
    Update the current model and embedding model selections.
    This syncs the selections with environment variables.
    """
    try:
        provider = request.get("provider")
        model = request.get("model")
        embedding_model = request.get("embeddingModel")
        
        logger.info(f"Updating model selection - Provider: {provider}, Model: {model}, Embedding: {embedding_model}")
        
        # Update provider model selection
        if provider and model:
            from api.config import get_model_config
            # This will sync VLLM_MODEL_NAME if provider is vllm
            get_model_config(provider, model)
        
        # Update embedding model selection  
        if embedding_model:
            from api.config import set_embedding_model
            set_embedding_model(embedding_model)
        
        return {
            "success": True,
            "message": "Model selection updated successfully",
            "provider": provider,
            "model": model,
            "embeddingModel": embedding_model
        }
        
    except Exception as e:
        logger.error(f"Error updating model selection: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/models/embeddings", response_model=dict)
async def get_embedding_models():
    """
    Get available embedding models configuration.
    
    Returns:
        dict: Embedding models configuration with display names and API URLs
    """
    try:
        logger.info("Fetching embedding models configuration")
        
        # Get embedding models from embedder config
        embedding_models = embedder_config.get("embedding_models", {})
        default_model = embedder_config.get("default_embedding_model", "jina-embeddings-v3")
        
        # Format for frontend consumption
        models = []
        for model_id, model_config in embedding_models.items():
            models.append({
                "id": model_id,
                "name": model_config.get("display_name", model_id),
                "dimensions": model_config.get("dimensions", 1024),
                "api_url": model_config.get("api_url", "")
            })
        
        return {
            "models": models,
            "defaultModel": default_model
        }
        
    except Exception as e:
        logger.error(f"Error fetching embedding models: {str(e)}")
        # Return default configuration
        return {
            "models": [
                {
                    "id": "jina-embeddings-v3",
                    "name": "Jina Embeddings v3",
                    "dimensions": 1024,
                    "api_url": "https://vllm.com/jina-embeddings-v3/v1"
                }
            ],
            "defaultModel": "jina-embeddings-v3"
        }

@app.get("/models/config", response_model=ModelConfig)
async def get_model_config():
    """
    Get available model providers and their models.

    This endpoint returns the configuration of available model providers and their
    respective models that can be used throughout the application.

    Returns:
        ModelConfig: A configuration object containing providers and their models
    """
    try:
        logger.info("Fetching model configurations")

        # Create providers from the config file
        providers = []
        default_provider = configs.get("default_provider", "google")

        # Add provider configuration based on config.py
        for provider_id, provider_config in configs["providers"].items():
            models = []
            # Add models from config
            for model_id, model_config in provider_config["models"].items():
                # Get display name or use model_id as fallback
                display_name = model_config.get("display_name", model_id)
                models.append(Model(id=model_id, name=display_name))

            # Add provider with its models
            providers.append(
                Provider(
                    id=provider_id,
                    name=f"{provider_id.capitalize()}",
                    supportsCustomModel=provider_config.get("supportsCustomModel", False),
                    models=models
                )
            )

        # Create and return the full configuration
        config = ModelConfig(
            providers=providers,
            defaultProvider=default_provider
        )
        return config

    except Exception as e:
        logger.error(f"Error creating model configuration: {str(e)}")
        # Return some default configuration in case of error
        return ModelConfig(
            providers=[
                Provider(
                    id="google",
                    name="Google",
                    supportsCustomModel=True,
                    models=[
                        Model(id="gemini-2.0-flash", name="Gemini 2.0 Flash")
                    ]
                )
            ],
            defaultProvider="google"
        )

@app.post("/export/wiki")
async def export_wiki(request: WikiExportRequest):
    """
    Export wiki content as Markdown or JSON.

    Args:
        request: The export request containing wiki pages and format

    Returns:
        A downloadable file in the requested format
    """
    try:
        logger.info(f"Exporting wiki for {request.repo_url} in {request.format} format")

        # Extract repository name from URL for the filename
        repo_parts = request.repo_url.rstrip('/').split('/')
        repo_name = repo_parts[-1] if len(repo_parts) > 0 else "wiki"

        # Get current timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if request.format == "markdown":
            # Generate Markdown content
            content = generate_markdown_export(request.repo_url, request.pages)
            filename = f"{repo_name}_wiki_{timestamp}.md"
            media_type = "text/markdown"
        else:  # JSON format
            # Generate JSON content
            content = generate_json_export(request.repo_url, request.pages)
            filename = f"{repo_name}_wiki_{timestamp}.json"
            media_type = "application/json"

        # Create response with appropriate headers for file download
        response = Response(
            content=content,
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )

        return response

    except Exception as e:
        error_msg = f"Error exporting wiki: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/local_repo/structure")
async def get_local_repo_structure(path: str = Query(None, description="Path to local repository")):
    """Return the file tree and README content for a local repository."""
    if not path:
        return JSONResponse(
            status_code=400,
            content={"error": "No path provided. Please provide a 'path' query parameter."}
        )

    if not os.path.isdir(path):
        return JSONResponse(
            status_code=404,
            content={"error": f"Directory not found: {path}"}
        )

    try:
        logger.info(f"Processing local repository at: {path}")
        
        # Use the same comprehensive exclusion logic as data_pipeline.py
        from .config import configs, DEFAULT_EXCLUDED_DIRS, DEFAULT_EXCLUDED_FILES
        import fnmatch
        
        # Get comprehensive exclusion lists
        final_excluded_dirs = DEFAULT_EXCLUDED_DIRS.copy()
        final_excluded_files = DEFAULT_EXCLUDED_FILES.copy()
        
        # Add exclusions from repo config
        file_filters_config = configs.get("file_filters", {})
        excluded_filename_patterns = file_filters_config.get("excluded_filename_patterns", [])
        repo_excluded_dirs = file_filters_config.get("excluded_dirs", [])
        final_excluded_dirs.extend(repo_excluded_dirs)
        
        # Normalize exclusion patterns (same logic as data_pipeline.py)
        normalized_excluded_dirs = []
        for p in final_excluded_dirs:
            if not p or not p.strip():
                continue
            normalized = p.strip()
            if normalized.startswith('./'):
                normalized = normalized[2:]
            normalized = normalized.rstrip('/')
            if normalized and normalized not in normalized_excluded_dirs:
                normalized_excluded_dirs.append(normalized)
        
        logger.info(f"🌳 File tree will exclude: {sorted(set(normalized_excluded_dirs))}")
        
        file_tree_lines = []
        readme_content = ""

        for root, dirs, files in os.walk(path, topdown=True):
            # Get current directory relative to base path
            current_dir_relative = os.path.relpath(root, path)
            normalized_current_dir = current_dir_relative.replace('\\', '/')
            
            if normalized_current_dir == '.':
                normalized_current_dir = ''
            
            # Check if current directory should be skipped entirely
            should_skip_dir = False
            if normalized_current_dir:
                path_components = normalized_current_dir.split('/')
                for component in path_components:
                    if component in normalized_excluded_dirs:
                        should_skip_dir = True
                        logger.debug(f"🚫 File tree: Skipping {normalized_current_dir} ('{component}' excluded)")
                        break
            
            if should_skip_dir:
                dirs.clear()  # Don't recurse into excluded directories
                continue
            
            # Filter immediate subdirectories to prevent walking into them
            dirs[:] = [d for d in dirs if d not in normalized_excluded_dirs]
            
            # Process files in current directory
            for file in files:
                file_path = os.path.join(root, file)
                rel_dir = os.path.relpath(root, path)
                rel_file = os.path.join(rel_dir, file) if rel_dir != '.' else file
                normalized_rel_file = rel_file.replace('\\', '/')
                
                # Check if file should be excluded by filename pattern
                filename_excluded = False
                for pattern in excluded_filename_patterns:
                    if fnmatch.fnmatch(file, pattern) or fnmatch.fnmatch(normalized_rel_file, pattern):
                        filename_excluded = True
                        break
                
                if filename_excluded:
                    continue
                
                # Check if file matches general excluded file patterns
                if any(fnmatch.fnmatch(normalized_rel_file, pattern) for pattern in final_excluded_files):
                    continue
                
                # Safety check: ensure file is not in excluded directory path
                file_path_components = normalized_rel_file.split('/')
                file_in_excluded_dir = False
                for component in file_path_components[:-1]:
                    if component in normalized_excluded_dirs:
                        file_in_excluded_dir = True
                        break
                
                if file_in_excluded_dir:
                    continue
                
                # Add to file tree (only relevant files for LLM analysis)
                file_tree_lines.append(normalized_rel_file)
                
                # Find README.md (case-insensitive)
                if file.lower() == 'readme.md' and not readme_content:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            readme_content = f.read()
                    except Exception as e:
                        logger.warning(f"Could not read README.md: {str(e)}")
                        readme_content = ""

        file_tree_str = '\n'.join(sorted(file_tree_lines))
        logger.info(f"📊 Generated clean file tree: {len(file_tree_lines)} files for LLM analysis (excluded vendor, .git, node_modules, etc.)")
        
        # Enhanced response with architectural context for comprehensive wiki generation
        response = {
            "file_tree": file_tree_str, 
            "readme": readme_content,
            "analysis_guidance": """
COMPREHENSIVE WIKI GENERATION INSTRUCTIONS:

When analyzing this repository structure, focus on creating a holistic, interconnected wiki that includes:

1. **SYSTEM ARCHITECTURE OVERVIEW**
   - Create a comprehensive architecture diagram using mermaid
   - Show how major components interact
   - Include data flow and system boundaries

2. **LOGICAL GROUPINGS** (instead of fragmented pages):
   - Core Application Logic (business logic, main workflows)
   - Data & Infrastructure Layer (database, APIs, external integrations)
   - User Interface & Presentation (frontend, templates, styling)
   - Development & Operations (build, deploy, testing, configuration)
   - Supporting Utilities (helpers, shared libraries, common tools)

3. **MERMAID DIAGRAMS TO INCLUDE**:
   - System architecture flowchart
   - Sequence diagrams for key workflows
   - Component relationship diagrams
   - Data flow diagrams

4. **INTERCONNECTED CONTENT**:
   - Cross-reference related components
   - Show dependencies and relationships
   - Provide navigation between related concepts
   - Link architectural decisions to implementation details

Focus on creating a cohesive documentation experience rather than isolated component descriptions.
            """
        }
        
        return response
    except Exception as e:
        logger.error(f"Error processing local repository: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing local repository: {str(e)}"}
        )

def generate_markdown_export(repo_url: str, pages: List[WikiPage]) -> str:
    """
    Generate Markdown export of wiki pages.

    Args:
        repo_url: The repository URL
        pages: List of wiki pages

    Returns:
        Markdown content as string
    """
    # Start with metadata
    markdown = f"# Wiki Documentation for {repo_url}\n\n"
    markdown += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    # Add table of contents
    markdown += "## Table of Contents\n\n"
    for page in pages:
        markdown += f"- [{page.title}](#{page.id})\n"
    markdown += "\n"

    # Add each page
    for page in pages:
        markdown += f"<a id='{page.id}'></a>\n\n"
        markdown += f"## {page.title}\n\n"



        # Add related pages
        if page.relatedPages and len(page.relatedPages) > 0:
            markdown += "### Related Pages\n\n"
            related_titles = []
            for related_id in page.relatedPages:
                # Find the title of the related page
                related_page = next((p for p in pages if p.id == related_id), None)
                if related_page:
                    related_titles.append(f"[{related_page.title}](#{related_id})")

            if related_titles:
                markdown += "Related topics: " + ", ".join(related_titles) + "\n\n"

        # Add page content
        markdown += f"{page.content}\n\n"
        markdown += "---\n\n"

    return markdown

def generate_json_export(repo_url: str, pages: List[WikiPage]) -> str:
    """
    Generate JSON export of wiki pages.

    Args:
        repo_url: The repository URL
        pages: List of wiki pages

    Returns:
        JSON content as string
    """
    # Create a dictionary with metadata and pages
    export_data = {
        "metadata": {
            "repository": repo_url,
            "generated_at": datetime.now().isoformat(),
            "page_count": len(pages)
        },
        "pages": [page.model_dump() for page in pages]
    }

    # Convert to JSON string with pretty formatting
    return json.dumps(export_data, indent=2)

# Import the simplified chat implementation
from .simple_chat import chat_completions_stream
from .websocket_wiki import handle_websocket_chat

# Add the chat_completions_stream endpoint to the main app
app.add_api_route("/chat/completions/stream", chat_completions_stream, methods=["POST"])

# Add the WebSocket endpoint
app.add_websocket_route("/ws/chat", handle_websocket_chat)

# --- Wiki Cache Helper Functions ---

WIKI_CACHE_DIR = os.path.join(get_adalflow_default_root_path(), "wikicache")
os.makedirs(WIKI_CACHE_DIR, exist_ok=True)

def get_wiki_cache_path(owner: str, repo: str, repo_type: str) -> str:
    """Generates the file path for a given wiki cache."""
    filename = f"deepwiki_cache_{repo_type}_{owner}_{repo}.json"
    return os.path.join(WIKI_CACHE_DIR, filename)

async def read_wiki_cache(owner: str, repo: str, repo_type: str) -> Optional[WikiCacheData]:
    """Reads wiki cache data from the file system."""
    cache_path = get_wiki_cache_path(owner, repo, repo_type)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return WikiCacheData(**data)
        except Exception as e:
            logger.error(f"Error reading wiki cache from {cache_path}: {e}")
            return None
    return None

async def save_wiki_cache(data: WikiCacheRequest) -> bool:
    """Saves wiki cache data to the file system."""
    cache_path = get_wiki_cache_path(data.repo.owner, data.repo.repo, data.repo.type)
    logger.info(f"Attempting to save wiki cache. Path: {cache_path}")
    try:
        payload = WikiCacheData(
            wiki_structure=data.wiki_structure,
            generated_pages=data.generated_pages,
            repo=data.repo,
            provider=data.provider,
            model=data.model
        )
        # Log size of data to be cached for debugging (avoid logging full content if large)
        try:
            payload_json = payload.model_dump_json()
            payload_size = len(payload_json.encode('utf-8'))
            logger.info(f"Payload prepared for caching. Size: {payload_size} bytes.")
        except Exception as ser_e:
            logger.warning(f"Could not serialize payload for size logging: {ser_e}")


        logger.info(f"Writing cache file to: {cache_path}")
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(payload.model_dump(), f, indent=2)
        logger.info(f"Wiki cache successfully saved to {cache_path}")
        return True
    except IOError as e:
        logger.error(f"IOError saving wiki cache to {cache_path}: {e.strerror} (errno: {e.errno})", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"Unexpected error saving wiki cache to {cache_path}: {e}", exc_info=True)
        return False

# --- Wiki Cache API Endpoints ---

@app.get("/api/wiki_cache", response_model=Optional[WikiCacheData])
async def get_cached_wiki(
    owner: str = Query(..., description="Repository owner"),
    repo: str = Query(..., description="Repository name"),
    repo_type: str = Query(..., description="Repository type (e.g., github, gitlab)")
):
    """
    Retrieves cached wiki data (structure and generated pages) for a repository.
    """
    logger.info(f"Attempting to retrieve wiki cache for {owner}/{repo} ({repo_type})")
    cached_data = await read_wiki_cache(owner, repo, repo_type)
    if cached_data:
        return cached_data
    else:
        # Return 200 with null body if not found, as frontend expects this behavior
        logger.info(f"Wiki cache not found for {owner}/{repo} ({repo_type})")
        return None

@app.post("/api/wiki_cache")
async def store_wiki_cache(request_data: WikiCacheRequest):
    """
    Stores generated wiki data (structure and pages) to the server-side cache.
    """
    logger.info(f"Attempting to save wiki cache for {request_data.repo.owner}/{request_data.repo.repo} ({request_data.repo.type})")
    success = await save_wiki_cache(request_data)
    if success:
        return {"message": "Wiki cache saved successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to save wiki cache")

@app.delete("/api/wiki_cache")
async def delete_wiki_cache(
    owner: str = Query(..., description="Repository owner"),
    repo: str = Query(..., description="Repository name"),
    repo_type: str = Query(..., description="Repository type (e.g., github, gitlab)"),
    authorization_code: Optional[str] = Query(None, description="Authorization code")
):
    """
    Deletes a specific wiki cache from the file system.
    """
    if WIKI_AUTH_MODE:
        logger.info("check the authorization code")
        if WIKI_AUTH_CODE != authorization_code:
            raise HTTPException(status_code=401, detail="Authorization code is invalid")

    logger.info(f"Attempting to delete wiki cache for {owner}/{repo} ({repo_type})")
    cache_path = get_wiki_cache_path(owner, repo, repo_type)

    if os.path.exists(cache_path):
        try:
            os.remove(cache_path)
            logger.info(f"Successfully deleted wiki cache: {cache_path}")
            return {"message": f"Wiki cache for {owner}/{repo} deleted successfully"}
        except Exception as e:
            logger.error(f"Error deleting wiki cache {cache_path}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to delete wiki cache: {str(e)}")
    else:
        logger.warning(f"Wiki cache not found, cannot delete: {cache_path}")
        raise HTTPException(status_code=404, detail="Wiki cache not found")

@app.get("/health")
async def health_check():
    """Health check endpoint for Docker and monitoring"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "deepwiki-api"
    }

@app.get("/")
async def root():
    """Root endpoint to check if the API is running and list available endpoints dynamically."""
    # Collect routes dynamically from the FastAPI app
    endpoints = {}
    for route in app.routes:
        if hasattr(route, "methods") and hasattr(route, "path"):
            # Skip docs and static routes
            if route.path in ["/openapi.json", "/docs", "/redoc", "/favicon.ico"]:
                continue
            # Group endpoints by first path segment
            path_parts = route.path.strip("/").split("/")
            group = path_parts[0].capitalize() if path_parts[0] else "Root"
            method_list = list(route.methods - {"HEAD", "OPTIONS"})
            for method in method_list:
                endpoints.setdefault(group, []).append(f"{method} {route.path}")

    # Optionally, sort endpoints for readability
    for group in endpoints:
        endpoints[group].sort()

    return {
        "message": "Welcome to Streaming API",
        "version": "1.0.0",
        "endpoints": endpoints
    }

# --- Processed Projects Endpoint --- (New Endpoint)
@app.get("/api/processed_projects", response_model=List[ProcessedProjectEntry])
async def get_processed_projects():
    """
    Lists all processed projects found in the wiki cache directory.
    Projects are identified by files named like: deepwiki_cache_{repo_type}_{owner}_{repo}.json
    """
    project_entries: List[ProcessedProjectEntry] = []
    # WIKI_CACHE_DIR is already defined globally in the file

    try:
        if not os.path.exists(WIKI_CACHE_DIR):
            logger.info(f"Cache directory {WIKI_CACHE_DIR} not found. Returning empty list.")
            return []

        logger.info(f"Scanning for project cache files in: {WIKI_CACHE_DIR}")
        filenames = await asyncio.to_thread(os.listdir, WIKI_CACHE_DIR) # Use asyncio.to_thread for os.listdir

        for filename in filenames:
            if filename.startswith("deepwiki_cache_") and filename.endswith(".json"):
                file_path = os.path.join(WIKI_CACHE_DIR, filename)
                try:
                    stats = await asyncio.to_thread(os.stat, file_path) # Use asyncio.to_thread for os.stat
                    parts = filename.replace("deepwiki_cache_", "").replace(".json", "").split('_')

                    # Expecting repo_type_owner_repo
                    # Example: deepwiki_cache_github_AsyncFuncAI_deepwiki-open.json
                    # parts = [github, AsyncFuncAI, deepwiki-open]
                    if len(parts) >= 3:
                        repo_type = parts[0]
                        owner = parts[1]
                        repo = "_".join(parts[2:]) # repo can contain underscores

                        project_entries.append(
                            ProcessedProjectEntry(
                                id=filename,
                                owner=owner,
                                repo=repo,
                                name=f"{owner}/{repo}",
                                repo_type=repo_type,
                                submittedAt=int(stats.st_mtime * 1000) # Convert to milliseconds
                            )
                        )
                    else:
                        logger.warning(f"Could not parse project details from filename: {filename}")
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
                    continue # Skip this file on error

        # Sort by most recent first
        project_entries.sort(key=lambda p: p.submittedAt, reverse=True)
        logger.info(f"Found {len(project_entries)} processed project entries.")
        return project_entries

    except Exception as e:
        logger.error(f"Error listing processed projects from {WIKI_CACHE_DIR}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to list processed projects from server cache.")
