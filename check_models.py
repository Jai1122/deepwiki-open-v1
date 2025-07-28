#!/usr/bin/env python3
"""
Quick script to check what embedding models are available on your vLLM endpoint.
This doesn't require additional dependencies.
"""

import os
import json
import urllib.request
import urllib.error

def check_embedding_models():
    # Get configuration from environment
    api_key = os.getenv('OPENAI_API_KEY', 'test')
    base_url = os.getenv('OPENAI_API_BASE_URL', 'https://myvllm.com/jina-embeddings-v3/v1')
    
    print("🔍 Checking available embedding models...")
    print(f"📍 Endpoint: {base_url}")
    print(f"🔑 API Key: {'✅ Set' if api_key else '❌ Missing'}")
    
    # Remove /v1 suffix if present and add /models
    if base_url.endswith('/v1'):
        models_url = base_url[:-3] + '/models'
    else:
        models_url = base_url.rstrip('/') + '/models'
    
    print(f"🌐 Models URL: {models_url}")
    
    try:
        # Create request with headers
        req = urllib.request.Request(models_url)
        req.add_header('Authorization', f'Bearer {api_key}')
        req.add_header('Content-Type', 'application/json')
        
        # Make the request
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            
        # Display results
        if 'data' in data:
            models = [model['id'] for model in data['data']]
            print(f"\n✅ Found {len(models)} models:")
            for i, model in enumerate(models, 1):
                print(f"   {i}. {model}")
            
            print(f"\n💡 To fix your configuration, add this to your .env file:")
            if models:
                print(f"   EMBEDDING_MODEL_NAME={models[0]}")
            
            return models
        else:
            print(f"❌ Unexpected response format: {data}")
            return []
            
    except urllib.error.HTTPError as e:
        print(f"❌ HTTP Error {e.code}: {e.reason}")
        if e.code == 401:
            print("💡 This usually means your API key is incorrect")
        elif e.code == 404:
            print("💡 This usually means the endpoint URL is incorrect")
        return []
    except urllib.error.URLError as e:
        print(f"❌ Connection Error: {e.reason}")
        print("💡 Check that the base URL is accessible from your network")
        return []
    except Exception as e:
        print(f"❌ Error: {e}")
        return []

def test_embedding_model(model_name):
    """Test a specific embedding model"""
    api_key = os.getenv('OPENAI_API_KEY', 'test')
    base_url = os.getenv('OPENAI_API_BASE_URL', 'https://myvllm.com/jina-embeddings-v3/v1')
    
    print(f"\n🧪 Testing model: {model_name}")
    
    # Create embeddings endpoint URL
    if base_url.endswith('/v1'):
        embeddings_url = base_url[:-3] + '/embeddings'
    else:
        embeddings_url = base_url.rstrip('/') + '/embeddings'
    
    # Prepare request data
    data = {
        "model": model_name,
        "input": ["test embedding"]
    }
    
    try:
        req = urllib.request.Request(embeddings_url)
        req.add_header('Authorization', f'Bearer {api_key}')
        req.add_header('Content-Type', 'application/json')
        req.data = json.dumps(data).encode()
        
        with urllib.request.urlopen(req, timeout=10) as response:
            result = json.loads(response.read().decode())
            
        if 'data' in result and len(result['data']) > 0:
            embedding = result['data'][0]['embedding']
            dimensions = len(embedding)
            print(f"   ✅ Success! Embedding dimensions: {dimensions}")
            print(f"   💡 Add to your .env: EMBEDDING_DIMENSIONS={dimensions}")
            return True
        else:
            print(f"   ❌ Unexpected response: {result}")
            return False
            
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 DeepWiki Embedding Model Checker")
    print("=" * 40)
    
    # Load .env file if it exists
    env_file = ".env"
    if os.path.exists(env_file):
        print(f"📄 Loading {env_file}")
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
    
    # Check available models
    models = check_embedding_models()
    
    # Test each model
    if models:
        print(f"\n🧪 Testing models...")
        for model in models:
            test_embedding_model(model)
    
    print(f"\n🎯 Next steps:")
    print(f"   1. Update your .env file with the correct EMBEDDING_MODEL_NAME")
    print(f"   2. Set EMBEDDING_DIMENSIONS to match your model")
    print(f"   3. Restart your DeepWiki application")