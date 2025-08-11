import logging
import os
from ..config import configs, resolve_embedding_config

logger = logging.getLogger(__name__)

def get_embedder():
    """
    Initializes and returns the vLLM embedder based on the configuration.

    Returns:
        Embedder instance: The configured embedder instance.
    """
    from ..config import _ensure_configs_loaded, CLIENT_CLASSES
    
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
    # First check if model_client class is already resolved
    model_client_class = embedder_config.get("model_client")
    
    # If not found, try to resolve it from client_class
    if not model_client_class and "client_class" in embedder_config:
        class_name = embedder_config["client_class"]
        if class_name in CLIENT_CLASSES:
            model_client_class = CLIENT_CLASSES[class_name]
            logger.info(f"Resolved client_class '{class_name}' to {model_client_class}")
    
    logger.debug(f"Raw embedder_config: {embedder_config}")
    logger.debug(f"model_client_class type: {type(model_client_class)}, value: {model_client_class}")
    
    if not model_client_class:
        raise ValueError(f"model_client not specified for embedder '{embedder_key}'. Config: {embedder_config}. Available CLIENT_CLASSES: {list(CLIENT_CLASSES.keys())}")

    initialize_kwargs = embedder_config.get("initialize_kwargs", {}).copy()
    model_kwargs = embedder_config.get("model_kwargs", {}).copy()
    
    # Resolve dynamic configuration from current embedding model
    current_embedding_model = os.environ.get("EMBEDDING_MODEL_NAME", "/app/models/jina-embeddings-v3")
    embedding_config = resolve_embedding_config(current_embedding_model)
    
    # Replace dynamic placeholders with actual values
    if initialize_kwargs.get("base_url") == "DYNAMIC_FROM_MODEL_CONFIG":
        initialize_kwargs["base_url"] = embedding_config["api_url"]
        logger.debug(f"Resolved dynamic base_url to: {embedding_config['api_url']}")
        
    if model_kwargs.get("model") == "DYNAMIC_FROM_MODEL_CONFIG":
        model_kwargs["model"] = embedding_config["model"]
        logger.debug(f"Resolved dynamic model to: {embedding_config['model']}")
        
    if model_kwargs.get("dimensions") == "DYNAMIC_FROM_MODEL_CONFIG":
        model_kwargs["dimensions"] = embedding_config["dimensions"]
        logger.debug(f"Resolved dynamic dimensions to: {embedding_config['dimensions']}")
        
    # Ensure API key is resolved from environment variable
    if initialize_kwargs.get("api_key") and "${VLLM_API_KEY}" in str(initialize_kwargs.get("api_key")):
        vllm_api_key = os.environ.get("VLLM_API_KEY", "dummy")
        initialize_kwargs["api_key"] = vllm_api_key
        logger.debug(f"Resolved VLLM_API_KEY (first 10 chars): {vllm_api_key[:10] if vllm_api_key != 'dummy' else 'dummy'}...")
        
    # Additional environment variable resolution for base_url
    if initialize_kwargs.get("base_url") and "${OPENAI_API_BASE_URL}" in str(initialize_kwargs.get("base_url")):
        openai_base_url = os.environ.get("OPENAI_API_BASE_URL", "")
        initialize_kwargs["base_url"] = openai_base_url
        logger.debug(f"Resolved OPENAI_API_BASE_URL to: {openai_base_url}")
    
    # Validation: ensure required parameters are set
    if not initialize_kwargs.get("base_url"):
        raise ValueError(f"base_url not resolved for embedder. Check OPENAI_API_BASE_URL environment variable or embedding model config.")
    
    if not initialize_kwargs.get("api_key"):
        logger.warning("API key not set, using 'dummy' - this may work for some vLLM setups")
        initialize_kwargs["api_key"] = "dummy"
    
    # Log resolved configuration for debugging
    logger.info(f"Embedder configuration (resolved):")
    logger.info(f"  Current embedding model: {current_embedding_model}")
    logger.info(f"  Model: {model_kwargs.get('model', 'NOT_SET')}")
    logger.info(f"  Base URL: {initialize_kwargs.get('base_url', 'NOT_SET')}")
    logger.info(f"  Dimensions: {model_kwargs.get('dimensions', 'NOT_SET')}")
    logger.info(f"  API Key: {'SET' if initialize_kwargs.get('api_key') else 'NOT_SET'}")
    
    try:
        # Create an instance of the model client if it's a class
        if isinstance(model_client_class, type):
            # It's a class, instantiate it
            logger.debug(f"Instantiating {model_client_class.__name__} with kwargs: {initialize_kwargs}")
            model_client = model_client_class(**initialize_kwargs)
            logger.debug(f"Created model_client instance: {type(model_client)} - {model_client}")
        else:
            # It's already an instance, use it directly
            logger.debug(f"Using existing model_client instance: {type(model_client_class)}")
            model_client = model_client_class
        
        # Final validation - ensure we have an instance, not a class
        if isinstance(model_client, type):
            raise ValueError(f"model_client is still a class after instantiation: {model_client}. Expected an instance.")
        
        # Additional validation - test that the instance works
        try:
            # Test that the client can be used
            test_methods = ['embeddings', 'acall', 'convert_inputs_to_api_kwargs']
            available_methods = [method for method in test_methods if hasattr(model_client, method)]
            logger.debug(f"model_client available methods: {available_methods}")
            
            # Test instantiation worked correctly
            if hasattr(model_client, 'client'):
                logger.debug(f"model_client.client type: {type(model_client.client)}")
            else:
                logger.warning("model_client has no 'client' attribute")
                
        except Exception as validation_error:
            logger.error(f"model_client validation failed: {validation_error}")
        
        logger.info(f"Final model_client validation passed: {type(model_client)}")
        
        # Try importing adalflow for full functionality
        try:
            import adalflow as adal
            logger.debug(f"About to create adalflow Embedder with model_client type: {type(model_client)}")
            # Double-check that model_client has the expected interface
            if not hasattr(model_client, 'embeddings') and not hasattr(model_client, 'acall'):
                logger.warning(f"model_client {type(model_client)} doesn't have expected methods (embeddings or acall)")
            
            # Extra debugging for the specific error
            logger.debug(f"Creating adalflow Embedder with:")
            logger.debug(f"  - model_client: {model_client} (type: {type(model_client)})")
            logger.debug(f"  - model_kwargs: {model_kwargs}")
            logger.debug(f"  - model_client is instance?: {not isinstance(model_client, type)}")
            
            try:
                embedder = adal.Embedder(
                    model_client=model_client,
                    model_kwargs=model_kwargs,
                )
            except Exception as adalflow_error:
                logger.error(f"adalflow Embedder creation failed with: {adalflow_error}")
                logger.error(f"adalflow error type: {type(adalflow_error)}")
                
                # If the error is specifically about ModelClient, try using adalflow's OpenAI client
                if "ModelClient instance" in str(adalflow_error):
                    logger.warning("Trying to use adalflow's built-in OpenAI client instead")
                    try:
                        # Try to use adalflow's OpenAI client
                        from adalflow.components.model_client import OpenAIClient as AdalflowOpenAI
                        
                        adalflow_client = AdalflowOpenAI(**initialize_kwargs)
                        logger.debug(f"Created adalflow OpenAI client: {type(adalflow_client)}")
                        
                        embedder = adal.Embedder(
                            model_client=adalflow_client,
                            model_kwargs=model_kwargs,
                        )
                        logger.info("✅ Embedder initialized successfully with adalflow's OpenAI client")
                        return embedder
                        
                    except Exception as fallback_error:
                        logger.error(f"Adalflow OpenAI client fallback also failed: {fallback_error}")
                
                import traceback
                logger.error(f"Full adalflow traceback:\n{traceback.format_exc()}")
                raise
            logger.info("✅ Embedder initialized successfully with adalflow")
            return embedder
        except ImportError as adal_error:
            # Fallback to a simple wrapper if adalflow is not available
            logger.warning(f"adalflow not available ({adal_error}), creating simple embedder wrapper")
            
            class SimpleEmbedder:
                def __init__(self, model_client, model_kwargs):
                    self.model_client = model_client
                    self.model_kwargs = model_kwargs
                
                def embed(self, text):
                    """Generate embeddings for the given text"""
                    if hasattr(self.model_client, 'embeddings'):
                        return self.model_client.embeddings(input=text, **self.model_kwargs)
                    elif hasattr(self.model_client, 'client') and hasattr(self.model_client.client, 'embeddings'):
                        return self.model_client.client.embeddings.create(input=text, **self.model_kwargs)
                    else:
                        raise ValueError("Model client does not support embeddings")
            
            embedder = SimpleEmbedder(
                model_client=model_client,
                model_kwargs=model_kwargs,
            )
            logger.info("✅ Simple embedder wrapper initialized successfully")
            return embedder
        
    except Exception as e:
        logger.error(f"Failed to initialize embedder: {e}")
        logger.error(f"  model_client_class: {model_client_class}")
        logger.error(f"  initialize_kwargs: {initialize_kwargs}")
        logger.error(f"  model_kwargs: {model_kwargs}")
        import traceback
        logger.error(f"  traceback: {traceback.format_exc()}")
        raise ValueError(f"Embedder initialization failed: {e}")
