import adalflow as adal
import logging
import os
from ..config import configs, resolve_embedding_config

logger = logging.getLogger(__name__)

def get_embedder() -> adal.Embedder:
    """
    Initializes and returns the vLLM embedder based on the configuration.

    Returns:
        adal.Embedder: The configured embedder instance.
    """
    from ..config import _ensure_configs_loaded
    
    # Ensure configs are loaded first
    _ensure_configs_loaded()
    
    embedder_key = "embedder"
    logger.info("Using vLLM embedder configuration.")
    
    # Debug: log what configs we have
    logger.debug(f"Available config keys: {list(configs.keys())}")

    embedder_config = configs.get(embedder_key)
    if not embedder_config:
        raise ValueError(f"Missing embedder configuration for '{embedder_key}' in config files. Available keys: {list(configs.keys())}")

    # --- Initialize Embedder ---
    model_client_class = embedder_config.get("model_client")
    logger.debug(f"Raw embedder_config: {embedder_config}")
    logger.debug(f"model_client_class type: {type(model_client_class)}, value: {model_client_class}")
    
    if not model_client_class:
        raise ValueError(f"model_client not specified for embedder '{embedder_key}'. Config: {embedder_config}")

    initialize_kwargs = embedder_config.get("initialize_kwargs", {}).copy()
    model_kwargs = embedder_config.get("model_kwargs", {}).copy()
    
    # Resolve dynamic configuration from current embedding model
    current_embedding_model = os.environ.get("EMBEDDING_MODEL_NAME", "/app/models/jina-embeddings-v3")
    embedding_config = resolve_embedding_config(current_embedding_model)
    
    # Replace dynamic placeholders with actual values
    if initialize_kwargs.get("base_url") == "DYNAMIC_FROM_MODEL_CONFIG":
        initialize_kwargs["base_url"] = embedding_config["api_url"]
        
    if model_kwargs.get("model") == "DYNAMIC_FROM_MODEL_CONFIG":
        model_kwargs["model"] = embedding_config["model"]
        
    if model_kwargs.get("dimensions") == "DYNAMIC_FROM_MODEL_CONFIG":
        model_kwargs["dimensions"] = embedding_config["dimensions"]
        
    # Ensure API key is resolved from environment variable
    if initialize_kwargs.get("api_key") and "${VLLM_API_KEY}" in str(initialize_kwargs.get("api_key")):
        initialize_kwargs["api_key"] = os.environ.get("VLLM_API_KEY", "")
        
    # Additional environment variable resolution for base_url
    if initialize_kwargs.get("base_url") and "${OPENAI_API_BASE_URL}" in str(initialize_kwargs.get("base_url")):
        initialize_kwargs["base_url"] = os.environ.get("OPENAI_API_BASE_URL", "")
    
    # Log resolved configuration for debugging
    logger.info(f"Embedder configuration (resolved):")
    logger.info(f"  Current embedding model: {current_embedding_model}")
    logger.info(f"  Model: {model_kwargs.get('model', 'NOT_SET')}")
    logger.info(f"  Base URL: {initialize_kwargs.get('base_url', 'NOT_SET')}")
    logger.info(f"  Dimensions: {model_kwargs.get('dimensions', 'NOT_SET')}")
    
    try:
        # Create an instance of the model client if it's a class
        if isinstance(model_client_class, type):
            # It's a class, instantiate it
            model_client = model_client_class(**initialize_kwargs)
        else:
            # It's already an instance, use it directly
            model_client = model_client_class
        
        embedder = adal.Embedder(
            model_client=model_client,
            model_kwargs=model_kwargs,
        )
        
        logger.info("✅ Embedder initialized successfully")
        return embedder
        
    except Exception as e:
        logger.error(f"Failed to initialize embedder: {e}")
        raise ValueError(f"Embedder initialization failed: {e}")
