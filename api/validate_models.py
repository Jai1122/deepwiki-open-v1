#!/usr/bin/env python3
"""
Model validation utility for DeepWiki.
This script helps validate that your vLLM endpoints and models are correctly configured.
"""

import os
import sys
import asyncio
import json
from typing import Dict, Any, Optional
import logging

# Add the api directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from openai import OpenAI
    import requests
except ImportError as e:
    print(f"Missing required packages: {e}")
    print("Please install: pip install openai requests")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_env_file(env_path: str = ".env") -> Dict[str, str]:
    """Load environment variables from .env file."""
    env_vars = {}
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key] = value
    return env_vars

def test_vllm_llm_endpoint(api_key: str, base_url: str, model_name: str) -> bool:
    """Test the vLLM LLM endpoint."""
    print(f"\n🔍 Testing vLLM LLM Endpoint:")
    print(f"   Base URL: {base_url}")
    print(f"   Model: {model_name}")
    
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        
        # Test with a simple completion
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Hello, respond with just 'OK' to test connectivity."}],
            max_tokens=5,
            temperature=0.1
        )
        
        print(f"   ✅ LLM Test Response: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"   ❌ LLM Test Failed: {e}")
        return False

def test_embedding_endpoint(api_key: str, base_url: str, model_name: str) -> bool:
    """Test the embedding endpoint."""
    print(f"\n🔍 Testing Embedding Endpoint:")
    print(f"   Base URL: {base_url}")
    print(f"   Model: {model_name}")
    
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        
        # Test with a simple embedding
        response = client.embeddings.create(
            model=model_name,
            input=["Hello world test"]
        )
        
        embedding_size = len(response.data[0].embedding)
        print(f"   ✅ Embedding Test Successful")
        print(f"   📐 Embedding Dimensions: {embedding_size}")
        return True
        
    except Exception as e:
        print(f"   ❌ Embedding Test Failed: {e}")
        return False

def list_available_models(api_key: str, base_url: str) -> Optional[list]:
    """List available models on the endpoint."""
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        models = client.models.list()
        return [model.id for model in models.data]
    except Exception as e:
        print(f"   ❌ Failed to list models: {e}")
        return None

def main():
    print("🚀 DeepWiki Model Validation Tool")
    print("=" * 50)
    
    # Load environment variables
    env_vars = load_env_file()
    
    # Get configuration from environment
    vllm_api_key = env_vars.get('VLLM_API_KEY') or os.getenv('VLLM_API_KEY')
    vllm_base_url = env_vars.get('VLLM_API_BASE_URL') or os.getenv('VLLM_API_BASE_URL')
    vllm_model_name = env_vars.get('VLLM_MODEL_NAME') or os.getenv('VLLM_MODEL_NAME')
    
    embedding_api_key = env_vars.get('OPENAI_API_KEY') or os.getenv('OPENAI_API_KEY')
    embedding_base_url = env_vars.get('OPENAI_API_BASE_URL') or os.getenv('OPENAI_API_BASE_URL')
    embedding_model_name = env_vars.get('EMBEDDING_MODEL_NAME') or os.getenv('EMBEDDING_MODEL_NAME', '/app/models/jina-embeddings-v3')
    
    print(f"📋 Configuration Summary:")
    print(f"   vLLM API Key: {'✅ Set' if vllm_api_key else '❌ Missing'}")
    print(f"   vLLM Base URL: {vllm_base_url or '❌ Missing'}")
    print(f"   vLLM Model: {vllm_model_name or '❌ Missing'}")
    print(f"   Embedding API Key: {'✅ Set' if embedding_api_key else '❌ Missing'}")
    print(f"   Embedding Base URL: {embedding_base_url or '❌ Missing'}")
    print(f"   Embedding Model: {embedding_model_name or '❌ Missing'}")
    
    success_count = 0
    total_tests = 0
    
    # Test vLLM LLM endpoint
    if vllm_api_key and vllm_base_url and vllm_model_name:
        total_tests += 1
        if test_vllm_llm_endpoint(vllm_api_key, vllm_base_url, vllm_model_name):
            success_count += 1
        
        # List available LLM models
        print(f"\n📝 Available LLM Models:")
        models = list_available_models(vllm_api_key, vllm_base_url)
        if models:
            for model in models:
                status = "✅" if model == vllm_model_name else "📋"
                print(f"   {status} {model}")
        else:
            print("   ❌ Could not retrieve model list")
    else:
        print(f"\n⚠️  Skipping LLM test - missing configuration")
    
    # Test embedding endpoint
    if embedding_api_key and embedding_base_url and embedding_model_name:
        total_tests += 1
        if test_embedding_endpoint(embedding_api_key, embedding_base_url, embedding_model_name):
            success_count += 1
            
        # List available embedding models
        print(f"\n📝 Available Embedding Models:")
        models = list_available_models(embedding_api_key, embedding_base_url)
        if models:
            for model in models:
                status = "✅" if model == embedding_model_name else "📋"
                print(f"   {status} {model}")
        else:
            print("   ❌ Could not retrieve model list")
    else:
        print(f"\n⚠️  Skipping embedding test - missing configuration")
    
    # Summary
    print("\n" + "=" * 50)
    print(f"🎯 Test Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests and total_tests > 0:
        print("🎉 All tests passed! Your configuration looks good.")
        return True
    else:
        print("❌ Some tests failed. Please check your configuration.")
        print("\n💡 Common fixes:")
        print("   1. Verify your API keys are correct")
        print("   2. Check that model names match what's deployed")
        print("   3. Ensure endpoints are accessible from your network")
        print("   4. Update EMBEDDING_MODEL_NAME in your .env file")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)