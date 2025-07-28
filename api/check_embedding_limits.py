#!/usr/bin/env python3
"""
Utility to check and validate embedding model token limits.
This helps identify the actual token limits of your embedding model.
"""

import os
import sys

# Add the api directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import count_tokens
from tools.embedder import get_embedder
from config import is_ollama_embedder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_embedding_token_limits():
    """Test the actual token limits of the current embedding model."""
    
    print("🔍 Testing Embedding Model Token Limits")
    print("=" * 50)
    
    try:
        # Get the current embedder
        embedder = get_embedder(is_ollama_embedder())
        
        # Test with increasingly large text samples
        test_sizes = [1000, 2000, 4000, 6000, 8000, 10000, 12000, 16000]
        
        # Create test text
        base_text = "This is a test sentence for embedding token limit validation. " * 20
        
        max_successful_tokens = 0
        
        for size in test_sizes:
            # Create text of approximately the target token count
            test_text = base_text * (size // count_tokens(base_text) + 1)
            test_text = test_text[:size*4]  # Rough character-based truncation
            
            actual_tokens = count_tokens(test_text)
            
            print(f"\nTesting with ~{actual_tokens} tokens...")
            
            try:
                # Try to embed the text
                embedding = embedder.call(test_text)
                
                if embedding and hasattr(embedding, 'embedding') and embedding.embedding:
                    print(f"  ✅ SUCCESS: {actual_tokens} tokens embedded successfully")
                    max_successful_tokens = actual_tokens
                else:
                    print(f"  ❌ FAILED: No embedding returned for {actual_tokens} tokens")
                    break
                    
            except Exception as e:
                error_msg = str(e)
                print(f"  ❌ FAILED: {actual_tokens} tokens - {error_msg}")
                
                if "maximum context length" in error_msg or "token" in error_msg.lower():
                    print(f"  📊 Token limit error detected at {actual_tokens} tokens")
                    break
                else:
                    print(f"  ⚠️  Other error: {error_msg}")
                    break
        
        print(f"\n📊 Results:")
        print(f"  Maximum successful embedding: {max_successful_tokens} tokens")
        
        if max_successful_tokens > 0:
            recommended_limit = int(max_successful_tokens * 0.8)  # 80% safety margin
            print(f"  Recommended chunk limit: {recommended_limit} tokens")
            
            print(f"\n🔧 Configuration Update:")
            print(f"  Update your api/config/embedder.json:")
            print(f'  "max_tokens_per_chunk": {recommended_limit}')
        else:
            print("  ⚠️  Could not determine token limits - check your configuration")
        
    except Exception as e:
        print(f"❌ Failed to test embedding limits: {e}")
        return None
    
    return max_successful_tokens

def check_current_configuration():
    """Check the current embedding configuration."""
    
    print(f"\n⚙️  Current Configuration:")
    print("-" * 30)
    
    from config import configs
    
    # Check embedder config
    embedder_config = configs.get("embedder", {})
    model_name = embedder_config.get("model_kwargs", {}).get("model", "NOT_SET")
    dimensions = embedder_config.get("model_kwargs", {}).get("dimensions", "NOT_SET")
    
    print(f"  Embedding Model: {model_name}")
    print(f"  Dimensions: {dimensions}")
    
    # Check text splitter config
    splitter_config = configs.get("text_splitter", {})
    chunk_size = splitter_config.get("chunk_size", "NOT_SET")
    max_tokens = splitter_config.get("max_tokens_per_chunk", "NOT_SET")
    
    print(f"  Chunk Size (words): {chunk_size}")
    print(f"  Max Tokens per Chunk: {max_tokens}")
    
    # Environment variables
    print(f"\n🌍 Environment Variables:")
    print(f"  EMBEDDING_MODEL_NAME: {os.environ.get('EMBEDDING_MODEL_NAME', 'NOT_SET')}")
    print(f"  EMBEDDING_DIMENSIONS: {os.environ.get('EMBEDDING_DIMENSIONS', 'NOT_SET')}")
    print(f"  OPENAI_API_BASE_URL: {os.environ.get('OPENAI_API_BASE_URL', 'NOT_SET')}")

if __name__ == "__main__":
    check_current_configuration()
    
    print(f"\n" + "="*60)
    
    # Ask user if they want to run the test
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_embedding_token_limits()
    else:
        print("💡 To test actual token limits, run:")
        print("   python check_embedding_limits.py --test")
        print("\n⚠️  Warning: This will make actual API calls to your embedding service")