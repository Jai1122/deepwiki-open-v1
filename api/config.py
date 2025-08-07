import os
import json
import logging
import re
from pathlib import Path
from typing import List, Union, Dict, Any

logger = logging.getLogger(__name__)

from api.openai_client import OpenAIClient
from api.openrouter_client import OpenRouterClient
from api.bedrock_client import BedrockClient
from api.azureai_client import AzureAIClient
from adalflow import GoogleGenAIClient, OllamaClient

# Get API keys from environment variables
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY')
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.environ.get('AWS_REGION')
AWS_ROLE_ARN = os.environ.get('AWS_ROLE_ARN')

# vLLM Configuration
VLLM_API_KEY = os.environ.get('VLLM_API_KEY')
VLLM_API_BASE_URL = os.environ.get('VLLM_API_BASE_URL')
VLLM_MODEL_NAME = os.environ.get('VLLM_MODEL_NAME')
OPENAI_API_BASE_URL = os.environ.get('OPENAI_API_BASE_URL')

# Embedding Configuration
EMBEDDING_MODEL_NAME = os.environ.get('EMBEDDING_MODEL_NAME', '/app/models/jina-embeddings-v3')
EMBEDDING_DIMENSIONS = os.environ.get('EMBEDDING_DIMENSIONS', '1024')

# Set keys in environment (in case they're needed elsewhere in the code)
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
if OPENROUTER_API_KEY:
    os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY
if AWS_ACCESS_KEY_ID:
    os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
if AWS_SECRET_ACCESS_KEY:
    os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY
if AWS_REGION:
    os.environ["AWS_REGION"] = AWS_REGION
if AWS_ROLE_ARN:
    os.environ["AWS_ROLE_ARN"] = AWS_ROLE_ARN
if VLLM_API_KEY:
    os.environ["VLLM_API_KEY"] = VLLM_API_KEY
if VLLM_API_BASE_URL:
    os.environ["VLLM_API_BASE_URL"] = VLLM_API_BASE_URL
if VLLM_MODEL_NAME:
    os.environ["VLLM_MODEL_NAME"] = VLLM_MODEL_NAME
if OPENAI_API_BASE_URL:
    os.environ["OPENAI_API_BASE_URL"] = OPENAI_API_BASE_URL
if EMBEDDING_MODEL_NAME:
    os.environ["EMBEDDING_MODEL_NAME"] = EMBEDDING_MODEL_NAME
if EMBEDDING_DIMENSIONS:
    os.environ["EMBEDDING_DIMENSIONS"] = EMBEDDING_DIMENSIONS

# Wiki authentication settings
raw_auth_mode = os.environ.get('DEEPWIKI_AUTH_MODE', 'False')
WIKI_AUTH_MODE = raw_auth_mode.lower() in ['true', '1', 't']
WIKI_AUTH_CODE = os.environ.get('DEEPWIKI_AUTH_CODE', '')

# Get configuration directory from environment variable, or use default if not set
CONFIG_DIR = os.environ.get('DEEPWIKI_CONFIG_DIR', None)

# Client class mapping
CLIENT_CLASSES = {
    "GoogleGenAIClient": GoogleGenAIClient,
    "OpenAIClient": OpenAIClient,
    "OpenRouterClient": OpenRouterClient,
    "OllamaClient": OllamaClient,
    "BedrockClient": BedrockClient,
    "AzureAIClient": AzureAIClient
}

def replace_env_placeholders(config: Union[Dict[str, Any], List[Any], str, Any]) -> Union[Dict[str, Any], List[Any], str, Any]:
    """
    Recursively replace placeholders like "${ENV_VAR}" in string values
    within a nested configuration structure (dicts, lists, strings)
    with environment variable values. Logs a warning if a placeholder is not found.
    """
    pattern = re.compile(r"\$\{([A-Z0-9_]+)\}")

    def replacer(match: re.Match[str]) -> str:
        env_var_name = match.group(1)
        original_placeholder = match.group(0)
        env_var_value = os.environ.get(env_var_name)
        if env_var_value is None:
            logger.warning(
                f"Environment variable placeholder '{original_placeholder}' was not found in the environment. "
                f"The placeholder string will be used as is."
            )
            return original_placeholder
        return env_var_value

    if isinstance(config, dict):
        return {k: replace_env_placeholders(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [replace_env_placeholders(item) for item in config]
    elif isinstance(config, str):
        return pattern.sub(replacer, config)
    else:
        # Handles numbers, booleans, None, etc.
        return config

# Load JSON configuration file
def load_json_config(filename):
    try:
        # If environment variable is set, use the directory specified by it
        if CONFIG_DIR:
            config_path = Path(CONFIG_DIR) / filename
        else:
            # Otherwise use default directory
            config_path = Path(__file__).parent / "config" / filename

        logger.info(f"Loading configuration from {config_path}")

        if not config_path.exists():
            logger.warning(f"Configuration file {config_path} does not exist")
            return {}

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            config = replace_env_placeholders(config)
            return config
    except Exception as e:
        logger.error(f"Error loading configuration file {filename}: {str(e)}")
        return {}

# Load generator model configuration
def load_generator_config():
    generator_config = load_json_config("generator.json")

    # Add client classes to each provider
    if "providers" in generator_config:
        for provider_id, provider_config in generator_config["providers"].items():
            # Try to set client class from client_class
            if provider_config.get("client_class") in CLIENT_CLASSES:
                provider_config["model_client"] = CLIENT_CLASSES[provider_config["client_class"]]
            # Fall back to default mapping based on provider_id
            elif provider_id in ["google", "openai", "openrouter", "ollama", "bedrock", "azure", "vllm"]:
                default_map = {
                    "google": GoogleGenAIClient,
                    "openai": OpenAIClient,
                    "openrouter": OpenRouterClient,
                    "ollama": OllamaClient,
                    "bedrock": BedrockClient,
                    "azure": AzureAIClient,
                    "vllm": OpenAIClient # VLLM uses an OpenAI-compatible client
                }
                provider_config["model_client"] = default_map[provider_id]
            else:
                logger.warning(f"Unknown provider or client class: {provider_id}")

    return generator_config

# Load embedder configuration
def load_embedder_config():
    embedder_config = load_json_config("embedder.json")

    # Process client classes
    for key in ["embedder", "embedder_ollama"]:
        if key in embedder_config and "client_class" in embedder_config[key]:
            class_name = embedder_config[key]["client_class"]
            if class_name in CLIENT_CLASSES:
                embedder_config[key]["model_client"] = CLIENT_CLASSES[class_name]

    return embedder_config

def get_embedder_config():
    """
    Get the current embedder configuration.

    Returns:
        dict: The embedder configuration with model_client resolved
    """
    return configs.get("embedder", {})

def is_ollama_embedder():
    """
    Check if the current embedder configuration uses OllamaClient.

    Returns:
        bool: True if using OllamaClient, False otherwise
    """
    embedder_config = get_embedder_config()
    if not embedder_config:
        return False

    # Check if model_client is OllamaClient
    model_client = embedder_config.get("model_client")
    if model_client:
        return model_client.__name__ == "OllamaClient"

    # Fallback: check client_class string
    client_class = embedder_config.get("client_class", "")
    return client_class == "OllamaClient"

# Load repository and file filters configuration
def load_repo_config():
    return load_json_config("repo.json")

# Language configuration removed - English only support

# Default excluded directories and files
DEFAULT_EXCLUDED_DIRS: List[str] = [
    # Virtual environments and package managers
    "./.venv/", "./venv/", "./env/", "./virtualenv/",
    "./node_modules/", "./bower_components/", "./jspm_packages/",
    # Version control
    "./.git/", "./.svn/", "./.hg/", "./.bzr/",
    # Cache and compiled files
    "./__pycache__/", "./.pytest_cache/", "./.mypy_cache/", "./.ruff_cache/", "./.coverage/",
    # Build and distribution
    "./dist/", "./build/", "./out/", "./target/", "./bin/", "./obj/",
    # Documentation
    "./docs/", "./_docs/", "./site-docs/", "./_site/",
    # IDE specific
    "./.idea/", "./.vscode/", "./.vs/", "./.eclipse/", "./.settings/",
    # Logs and temporary files
    "./logs/", "./log/", "./tmp/", "./temp/",
]

DEFAULT_EXCLUDED_FILES: List[str] = [
    "yarn.lock", "pnpm-lock.yaml", "npm-shrinkwrap.json", "poetry.lock",
    "Pipfile.lock", "requirements.txt.lock", "Cargo.lock", "composer.lock",
    ".lock", ".DS_Store", "Thumbs.db", "desktop.ini", "*.lnk", ".env",
    ".env.*", "*.env", "*.cfg", "*.ini", ".flaskenv", ".gitignore",
    ".gitattributes", ".gitmodules", ".github", ".gitlab-ci.yml",
    ".prettierrc", ".eslintrc", ".eslintignore", ".stylelintrc",
    ".editorconfig", ".jshintrc", ".pylintrc", ".flake8", "mypy.ini",
    "pyproject.toml", "tsconfig.json", "webpack.config.js", "babel.config.js",
    "rollup.config.js", "jest.config.js", "karma.conf.js", "vite.config.js",
    "next.config.js", "*.min.js", "*.min.css", "*.bundle.js", "*.bundle.css",
    "*.map", "*.gz", "*.zip", "*.tar", "*.tgz", "*.rar", "*.7z", "*.iso",
    "*.dmg", "*.img", "*.msix", "*.appx", "*.appxbundle", "*.xap", "*.ipa",
    "*.deb", "*.rpm", "*.msi", "*.exe", "*.dll", "*.so", "*.dylib", "*.o",
    "*.obj", "*.jar", "*.war", "*.ear", "*.jsm", "*.class", "*.pyc", "*.pyd",
    "*.pyo", "__pycache__", "*.a", "*.lib", "*.lo", "*.la", "*.slo", "*.dSYM",
    "*.egg", "*.egg-info", "*.dist-info", "*.eggs", "node_modules",
    "bower_components", "jspm_packages", "lib-cov", "coverage", "htmlcov",
    ".nyc_output", ".tox", "dist", "build", "bld", "out", "bin", "target",
    "packages/*/dist", "packages/*/build", ".output"
]

# Initialize empty configuration
configs = {}

# Load all configuration files
generator_config = load_generator_config()
embedder_config = load_embedder_config()
repo_config = load_repo_config()

# Update configuration
if generator_config:
    configs["default_provider"] = generator_config.get("default_provider", "google")
    configs["providers"] = generator_config.get("providers", {})

# Update embedder configuration
if embedder_config:
    for key in ["embedder", "embedder_ollama", "retriever", "text_splitter"]:
        if key in embedder_config:
            configs[key] = embedder_config[key]

# Update repository configuration
if repo_config:
    for key in ["file_filters", "repository"]:
        if key in repo_config:
            configs[key] = repo_config[key]

# Language configuration removed - English only support

def resolve_dynamic_url(provider, model, config_key):
    """
    Resolve dynamic URLs for VLLM models based on model configuration
    
    Args:
        provider (str): Provider name (e.g., 'vllm')
        model (str): Model name (e.g., '/app/models/Qwen3-32B')
        config_key (str): Configuration key to resolve ('base_url', 'model', 'dimensions')
    
    Returns:
        str: Resolved value or original value if not dynamic
    """
    if provider != "vllm":
        return None
        
    provider_config = configs.get("providers", {}).get(provider, {})
    model_config = provider_config.get("models", {}).get(model, {})
    
    if config_key == "base_url":
        return model_config.get("api_url")
    elif config_key == "model":
        return model
    
    return None

def resolve_embedding_config(embedding_model):
    """
    Resolve embedding model configuration from embedder.json
    
    Args:
        embedding_model (str): Embedding model name
        
    Returns:
        dict: Configuration with api_url, dimensions, model name
    """
    embedding_models = embedder_config.get("embedding_models", {})
    model_config = embedding_models.get(embedding_model, {})
    
    return {
        "api_url": model_config.get("api_url"),
        "dimensions": model_config.get("dimensions", 1024),
        "model": embedding_model,
        "display_name": model_config.get("display_name", embedding_model)
    }

def set_embedding_model(embedding_model):
    """
    Set the current embedding model and sync environment variables
    
    Args:
        embedding_model (str): Embedding model name
    """
    embedding_config = resolve_embedding_config(embedding_model)
    
    # Update environment variables
    if embedding_config["api_url"]:
        os.environ["OPENAI_API_BASE_URL"] = embedding_config["api_url"]
        logger.info(f"Updated OPENAI_API_BASE_URL to: {embedding_config['api_url']}")
    
    os.environ["EMBEDDING_MODEL_NAME"] = embedding_model
    os.environ["EMBEDDING_DIMENSIONS"] = str(embedding_config["dimensions"])
    
    logger.info(f"Updated embedding model configuration to: {embedding_model} ({embedding_config['dimensions']}D)")

def get_model_config(provider="google", model=None):
    """
    Get configuration for the specified provider and model

    Parameters:
        provider (str): Model provider ('google', 'openai', 'openrouter', 'ollama', 'bedrock', 'vllm')
        model (str): Model name, or None to use default model

    Returns:
        dict: Configuration containing model_client, model and other parameters
    """
    # Sync environment variable for VLLM
    if provider == "vllm" and model:
        os.environ["VLLM_MODEL_NAME"] = model
        logger.info(f"Updated VLLM_MODEL_NAME environment variable to: {model}")
    
    # Sync OPENAI_BASE_URL for embedding model configuration - will be handled separately
    # when embedding model is selected
    if "providers" not in configs:
        raise ValueError("Provider configuration not loaded")

    provider_config = configs["providers"].get(provider)
    if not provider_config:
        raise ValueError(f"Configuration for provider '{provider}' not found")

    model_client = provider_config.get("model_client")
    if not model_client:
        raise ValueError(f"Model client not specified for provider '{provider}'")

    if not model:
        model = provider_config.get("default_model")
        if not model:
            raise ValueError(f"No default model specified for provider '{provider}'")

    # Get model parameters, falling back to default if specific model not found
    model_params = provider_config.get("models", {}).get(model)
    if model_params is None:
        logger.warning(f"Model '{model}' not found for provider '{provider}'. Falling back to default model config.")
        default_model_key = provider_config.get("default_model")
        model_params = provider_config.get("models", {}).get(default_model_key, {})

    # Get initialize_kwargs and resolve dynamic URLs for VLLM
    initialize_kwargs = provider_config.get("initialize_kwargs", {}).copy()
    
    # Resolve dynamic base_url for VLLM
    if provider == "vllm":
        dynamic_url = resolve_dynamic_url(provider, model, "base_url")
        if dynamic_url:
            initialize_kwargs["base_url"] = dynamic_url
        else:
            # Fallback to environment variable if no specific URL mapping
            initialize_kwargs["base_url"] = os.environ.get('VLLM_API_BASE_URL', 'http://localhost:8000/v1')
    
    result = {
        "model_client": model_client,
        "initialize_kwargs": initialize_kwargs
    }
    
    # Standardize model_kwargs, preparing for the API call
    model_kwargs = {"model": model}
    
    if provider == "ollama":
        # For Ollama, parameters are nested under 'options'
        ollama_options = model_params.get("options", {})
        model_kwargs.update(ollama_options)
    else:
        # For other providers, copy parameters directly
        # We are especially interested in 'max_completion_tokens' which will be renamed to 'max_tokens' for the API call
        params_to_copy = ["temperature", "top_p", "top_k", "max_completion_tokens"]
        for param in params_to_copy:
            if param in model_params:
                # Rename 'max_completion_tokens' to 'max_tokens' for the actual API call
                api_param = "max_tokens" if param == "max_completion_tokens" else param
                model_kwargs[api_param] = model_params[param]

    result["model_kwargs"] = model_kwargs
    return result

def get_context_window_for_model(provider: str, model: str) -> int:
    """
    Get the maximum context window size for a given model from the configuration.
    """
    try:
        provider_config = configs.get("providers", {}).get(provider, {})
        model_config = provider_config.get("models", {}).get(model)

        # If the exact model isn't found, try the default model for the provider
        if model_config is None:
            default_model_key = provider_config.get("default_model")
            model_config = provider_config.get("models", {}).get(default_model_key, {})

        # Check for 'context_window' or ollama's 'num_ctx'
        if 'context_window' in model_config:
            return model_config['context_window']
        if 'options' in model_config and 'num_ctx' in model_config['options']:
            return model_config['options']['num_ctx']
            
        # Fallback to a default value if not specified in any config
        default_context_window = 8192 # A more conservative default
        logger.warning(f"context_window not configured for {provider}/{model}. Falling back to default of {default_context_window}.")
        return default_context_window
    except Exception:
        logger.exception(f"Error getting context_window for {provider}/{model}. Falling back to default of 8192.")
        return 8192

def validate_provider_config(provider: str) -> bool:
    """
    Validate that a provider has the necessary configuration and credentials.
    """
    if provider not in configs.get("providers", {}):
        logger.error(f"Provider '{provider}' not found in configuration")
        return False
    
    provider_config = configs["providers"][provider]
    
    # Check if model_client is set
    if "model_client" not in provider_config:
        logger.error(f"Provider '{provider}' missing model_client configuration")
        return False
    
    # Validate vLLM specific configuration
    if provider == "vllm":
        init_kwargs = provider_config.get("initialize_kwargs", {})
        api_key = init_kwargs.get("api_key", "").replace("${VLLM_API_KEY}", os.environ.get("VLLM_API_KEY", ""))
        base_url = init_kwargs.get("base_url", "").replace("${VLLM_API_BASE_URL}", os.environ.get("VLLM_API_BASE_URL", ""))
        
        if not api_key:
            logger.error("vLLM API key not configured. Set VLLM_API_KEY environment variable.")
            return False
        if not base_url:
            logger.error("vLLM base URL not configured. Set VLLM_API_BASE_URL environment variable.")
            return False
    
    # Validate embedder for OpenAI base URL
    embedder_config = configs.get("embedder", {})
    if embedder_config:
        init_kwargs = embedder_config.get("initialize_kwargs", {})
        openai_base_url = init_kwargs.get("base_url", "").replace("${OPENAI_API_BASE_URL}", os.environ.get("OPENAI_API_BASE_URL", ""))
        openai_api_key = init_kwargs.get("api_key", "").replace("${OPENAI_API_KEY}", os.environ.get("OPENAI_API_KEY", ""))
        
        if not openai_base_url:
            logger.error("Embedder base URL not configured. Set OPENAI_API_BASE_URL environment variable.")
            return False
        if not openai_api_key:
            logger.error("Embedder API key not configured. Set OPENAI_API_KEY environment variable.")
            return False
    
    return True

def get_max_tokens_for_model(provider: str, model: str) -> int:
    """
    Get the maximum completion tokens for a given model from the configuration.
    """
    try:
        provider_config = configs.get("providers", {}).get(provider, {})
        model_config = provider_config.get("models", {}).get(model)
        
        if model_config is None:
            default_model_key = provider_config.get("default_model")
            model_config = provider_config.get("models", {}).get(default_model_key, {})
        
        if 'max_completion_tokens' in model_config:
            return model_config['max_completion_tokens']
        
        # Fallback to a default value
        default_max_tokens = 4096
        logger.warning(f"max_completion_tokens not configured for {provider}/{model}. Falling back to default of {default_max_tokens}.")
        return default_max_tokens
    except Exception:
        logger.exception(f"Error getting max_completion_tokens for {provider}/{model}. Falling back to default of 4096.")
        return 4096
