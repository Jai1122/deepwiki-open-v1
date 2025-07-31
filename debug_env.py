#!/usr/bin/env python3
"""
Debug script to check environment variable loading
"""
import os
import sys
from pathlib import Path

# Add the api directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'api'))

def test_direct_env():
    """Test direct environment variable access"""
    print("=== Direct Environment Variable Test ===")
    print(f"VLLM_MODEL_NAME: {os.environ.get('VLLM_MODEL_NAME', 'NOT SET')}")
    print(f"VLLM_API_KEY: {os.environ.get('VLLM_API_KEY', 'NOT SET')}")
    print(f"VLLM_API_BASE_URL: {os.environ.get('VLLM_API_BASE_URL', 'NOT SET')}")
    print()

def test_dotenv_loading():
    """Test loading with python-dotenv"""
    try:
        from dotenv import load_dotenv
        print("=== Testing .env file loading ===")
        
        # Check if .env file exists
        env_file = Path('.env')
        if env_file.exists():
            print(f"✅ .env file found at: {env_file.absolute()}")
            # Show first few lines
            with open(env_file, 'r') as f:
                lines = f.readlines()[:10]
            print("First 10 lines of .env file:")
            for i, line in enumerate(lines, 1):
                print(f"  {i}: {line.rstrip()}")
        else:
            print("❌ .env file not found")
            return
        
        # Load environment variables
        load_dotenv()
        print("\nAfter load_dotenv():")
        print(f"VLLM_MODEL_NAME: {os.environ.get('VLLM_MODEL_NAME', 'NOT SET')}")
        print(f"VLLM_API_KEY: {os.environ.get('VLLM_API_KEY', 'NOT SET')}")
        print(f"VLLM_API_BASE_URL: {os.environ.get('VLLM_API_BASE_URL', 'NOT SET')}")
        print()
        
    except ImportError:
        print("❌ python-dotenv not available")
        print()

def test_manual_replacement():
    """Test manual environment variable replacement"""
    print("=== Testing Manual Environment Variable Replacement ===")
    import re
    
    pattern = re.compile(r"\$\{([A-Z0-9_]+)\}")
    
    def replacer(match):
        env_var_name = match.group(1)
        original_placeholder = match.group(0)
        env_var_value = os.environ.get(env_var_name)
        if env_var_value is None:
            print(f"WARNING: Environment variable '{env_var_name}' not found")
            return original_placeholder
        return env_var_value
    
    # Test replacement
    test_string = "${VLLM_MODEL_NAME}"
    result = pattern.sub(replacer, test_string)
    print(f"Replacement test: '{test_string}' -> '{result}'")
    
    # Test with actual generator.json content
    try:
        with open('api/config/generator.json', 'r') as f:
            import json
            config = json.load(f)
            
        vllm_default_model = config.get('providers', {}).get('vllm', {}).get('default_model', 'NOT FOUND')
        print(f"vLLM default_model in generator.json: {vllm_default_model}")
        
        # Try to replace it
        if isinstance(vllm_default_model, str):
            replaced_model = pattern.sub(replacer, vllm_default_model)
            print(f"After replacement: {replaced_model}")
        
    except Exception as e:
        print(f"Error reading generator.json: {e}")

def main():
    print("🔍 DeepWiki Environment Variable Debug Tool")
    print("=" * 50)
    print(f"Working directory: {os.getcwd()}")
    print()
    
    test_direct_env()
    test_dotenv_loading()
    test_manual_replacement()

if __name__ == "__main__":
    main()