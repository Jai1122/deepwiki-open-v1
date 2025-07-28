import adalflow as adal
import logging
from api.config import configs

logger = logging.getLogger(__name__)

def get_embedder(is_ollama_embedder: bool = False) -> adal.Embedder:
    """
    Initializes and returns the appropriate embedder based on the configuration.

    Args:
        is_ollama_embedder (bool): Flag to determine if the Ollama embedder should be used.

    Returns:
        adal.Embedder: The configured embedder instance.
    """
    if is_ollama_embedder:
        embedder_key = "embedder_ollama"
        logger.info("Using Ollama embedder configuration.")
    else:
        embedder_key = "embedder"
        logger.info("Using default embedder configuration.")

    embedder_config = configs.get(embedder_key)
    if not embedder_config:
        raise ValueError(f"Missing embedder configuration for '{embedder_key}' in config files.")

    # --- Initialize Embedder ---
    model_client_class = embedder_config.get("model_client")
    if not model_client_class:
        raise ValueError(f"model_client not specified for embedder '{embedder_key}'.")

    initialize_kwargs = embedder_config.get("initialize_kwargs", {})
    model_kwargs = embedder_config.get("model_kwargs", {})
    
    # Log configuration for debugging
    logger.info(f"Embedder configuration:")
    logger.info(f"  Model: {model_kwargs.get('model', 'NOT_SET')}")
    logger.info(f"  Base URL: {initialize_kwargs.get('base_url', 'NOT_SET')}")
    logger.info(f"  Dimensions: {model_kwargs.get('dimensions', 'NOT_SET')}")
    
    try:
        model_client = model_client_class(**initialize_kwargs)
        
        embedder = adal.Embedder(
            model_client=model_client,
            model_kwargs=model_kwargs,
        )
        
        logger.info("✅ Embedder initialized successfully")
        return embedder
        
    except Exception as e:
        logger.error(f"Failed to initialize embedder: {e}")
        raise ValueError(f"Embedder initialization failed: {e}")
