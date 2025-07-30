import uvicorn
import os
import sys
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from api.logging_config import setup_logging

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

# Add the current directory to the path so we can import the api package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Validate configuration and API keys
def validate_configuration():
    """Validate that at least one working provider is configured"""
    from api.config import get_model_config
    
    # Check for working providers
    working_providers = []
    provider_errors = {}
    
    # Test Google provider
    if os.environ.get('GOOGLE_API_KEY') and os.environ.get('GOOGLE_API_KEY') != 'your-google-api-key-here':
        try:
            config = get_model_config("google", "gemini-2.0-flash")
            working_providers.append("google")
            logger.info("✅ Google provider configured")
        except Exception as e:
            provider_errors["google"] = str(e)
            logger.warning(f"❌ Google provider not working: {e}")
    else:
        logger.warning("❌ Google API key not configured")
    
    # Test vLLM provider
    if (os.environ.get('VLLM_API_KEY') and 
        os.environ.get('VLLM_API_KEY') != 'your-actual-api-key' and
        os.environ.get('VLLM_API_BASE_URL') and
        'your-vllm-server-address' not in os.environ.get('VLLM_API_BASE_URL', '')):
        try:
            config = get_model_config("vllm", "/app/models/Qwen3-32B")
            working_providers.append("vllm")
            logger.info("✅ vLLM provider configured")
        except Exception as e:
            provider_errors["vllm"] = str(e)
            logger.warning(f"❌ vLLM provider not working: {e}")
    else:
        logger.warning("❌ vLLM provider not configured properly")
    
    # Test OpenAI provider
    if os.environ.get('OPENAI_API_KEY'):
        try:
            config = get_model_config("openai", "gpt-4o")
            working_providers.append("openai")
            logger.info("✅ OpenAI provider configured")
        except Exception as e:
            provider_errors["openai"] = str(e)
            logger.warning(f"❌ OpenAI provider not working: {e}")
    else:
        logger.warning("❌ OpenAI API key not configured")
    
    if not working_providers:
        logger.error("🚨 CRITICAL: No working LLM providers found!")
        logger.error("   Please configure at least one of: GOOGLE_API_KEY, VLLM_API_KEY+BASE_URL, OPENAI_API_KEY")
        logger.error(f"   Provider errors: {provider_errors}")
        logger.error("   DeepWiki will not function without a working provider")
        return False
    else:
        logger.info(f"✅ {len(working_providers)} working provider(s): {', '.join(working_providers)}")
        return True

# Validate configuration
if not validate_configuration():
    logger.error("🚨 CRITICAL: Cannot start server - no working LLM providers configured!")
    logger.error("   Please configure at least one of: GOOGLE_API_KEY, VLLM_API_KEY+BASE_URL, OPENAI_API_KEY")
    logger.error("   Server will NOT start without proper configuration.")
    sys.exit(1)  # Exit instead of starting with broken configuration

# Configure Google Generative AI
import google.generativeai as genai
from api.config import GOOGLE_API_KEY

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    logger.warning("GOOGLE_API_KEY not configured")

if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 8001))

    # Import the app here to ensure environment variables are set first
    from api.api import app

    logger.info(f"Starting Streaming API on port {port}")

    # Run the FastAPI app with uvicorn
    # Disable reload in production/Docker environment
    is_development = os.environ.get("NODE_ENV") != "production"
    
    if is_development:
        # Prevent infinite logging loop caused by file changes triggering log writes
        logging.getLogger("watchfiles.main").setLevel(logging.WARNING)

    uvicorn.run(
        "api.api:app",
        host="0.0.0.0",
        port=port,
        reload=is_development,
        ws_ping_interval=30,     # Send ping every 30 seconds
        ws_ping_timeout=10,      # Wait 10 seconds for pong response
        ws_max_size=16 * 1024 * 1024,  # 16MB max WebSocket message size
        timeout_keep_alive=60    # Keep connections alive for 60 seconds
    )
