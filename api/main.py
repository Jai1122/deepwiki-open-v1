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
    """Validate that vLLM provider is configured"""
    from api.config import get_model_config
    
    # Check vLLM provider
    vllm_key = os.environ.get('VLLM_API_KEY')
    vllm_url = os.environ.get('VLLM_API_BASE_URL')
    
    if (vllm_key and vllm_key not in ['your-actual-api-key', 'your_vllm_api_key', ''] and
        vllm_url and 'your-vllm-server-address' not in vllm_url):
        try:
            config = get_model_config("vllm", "/app/models/Qwen2.5-VL-7B-Instruct")
            logger.info("✅ vLLM provider configured")
            return True
        except Exception as e:
            logger.error(f"❌ vLLM provider not working: {e}")
            return False
    else:
        if vllm_key in ['your-actual-api-key', 'your_vllm_api_key'] or 'your-vllm-server-address' in (vllm_url or ''):
            logger.error("❌ vLLM credentials are set to placeholder values - please update .env with real values")
        else:
            logger.error("❌ vLLM provider not configured properly")
        return False

# Validate configuration
if not validate_configuration():
    logger.error("🚨 CRITICAL: Cannot start server - vLLM provider not configured!")
    logger.error("   Please configure VLLM_API_KEY and VLLM_API_BASE_URL")
    logger.error("   Server will NOT start without proper configuration.")
    sys.exit(1)  # Exit instead of starting with broken configuration

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
