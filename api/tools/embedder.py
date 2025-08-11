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
            
            # The key insight: adalflow expects its own ModelClient instances
            # Let's use adalflow's OpenAI client directly instead of our custom one
            logger.info("Using adalflow's native OpenAI client for better compatibility")
            
            try:
                # Try multiple import paths for adalflow's OpenAI client
                AdalflowOpenAI = None
                import_paths = [
                    "adalflow.components.model_client.OpenAIClient",
                    "adalflow.components.model_client.openai_client.OpenAIClient",
                    "adalflow.model_client.OpenAIClient"
                ]
                
                for import_path in import_paths:
                    try:
                        parts = import_path.split('.')
                        module_path = '.'.join(parts[:-1])
                        class_name = parts[-1]
                        module = __import__(module_path, fromlist=[class_name])
                        AdalflowOpenAI = getattr(module, class_name)
                        logger.debug(f"Successfully imported adalflow's OpenAI client from {import_path}")
                        break
                    except (ImportError, AttributeError):
                        continue
                
                if AdalflowOpenAI is None:
                    raise ImportError("Could not find adalflow's OpenAI client in any expected location")
                
                # Create adalflow's client with our config
                logger.debug(f"Creating adalflow OpenAI client with: {initialize_kwargs}")
                adalflow_client = AdalflowOpenAI(**initialize_kwargs)
                logger.debug(f"Created adalflow client: {type(adalflow_client)}")
                
                # Ensure it's an instance, not a class
                if isinstance(adalflow_client, type):
                    raise ValueError(f"adalflow_client is a class, not an instance: {adalflow_client}")
                
                # Create the embedder
                logger.debug(f"Creating adalflow Embedder with model_kwargs: {model_kwargs}")
                embedder = adal.Embedder(
                    model_client=adalflow_client,
                    model_kwargs=model_kwargs,
                )
                logger.info("✅ Embedder created successfully with adalflow's OpenAI client")
                return embedder
                
            except ImportError as e:
                logger.warning(f"Could not import adalflow's OpenAI client: {e}")
                logger.warning("This might be due to adalflow version differences")
                
                # Fallback: try with our custom client and better error handling
                logger.info("Attempting fallback with custom OpenAI client")
                try:
                    # Debug: check what adalflow is expecting
                    logger.debug(f"Our model_client type: {type(model_client)}")
                    logger.debug(f"Our model_client has embeddings method: {hasattr(model_client, 'embeddings')}")
                    logger.debug(f"Our model_client has acall method: {hasattr(model_client, 'acall')}")
                    
                    embedder = adal.Embedder(
                        model_client=model_client,
                        model_kwargs=model_kwargs,
                    )
                    logger.info("✅ Embedder created successfully with custom OpenAI client")
                    return embedder
                    
                except Exception as fallback_error:
                    logger.error(f"Fallback with custom client failed: {fallback_error}")
                    logger.error(f"Error type: {type(fallback_error)}")
                    
                    # If it's the ModelClient error, provide more specific guidance
                    if "ModelClient instance" in str(fallback_error):
                        logger.error("adalflow is rejecting our custom ModelClient")
                        logger.error("This suggests adalflow requires specific ModelClient subclasses")
                        
                    import traceback
                    logger.error(f"Full traceback: {traceback.format_exc()}")
                    raise
                    
            except Exception as e:
                logger.error(f"Failed to create adalflow embedder: {e}")
                logger.error(f"Error type: {type(e)}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                raise
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
