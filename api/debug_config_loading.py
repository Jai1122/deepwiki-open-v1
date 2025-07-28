#!/usr/bin/env python3
"""
Debug configuration loading to verify repo.json is being read correctly.
"""

import os
import sys
import json

# Add the api directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_config_loading():
    """Test configuration loading step by step."""
    
    print("🔍 Debugging Configuration Loading")
    print("=" * 50)
    
    # Step 1: Test direct file reading
    repo_config_path = os.path.join(os.path.dirname(__file__), "config", "repo.json")
    print(f"📁 Repo config path: {repo_config_path}")
    print(f"   File exists: {os.path.exists(repo_config_path)}")
    
    if os.path.exists(repo_config_path):
        with open(repo_config_path, 'r') as f:
            raw_config = json.load(f)
        
        print(f"\n📄 Raw repo.json content:")
        print(f"   file_filters keys: {list(raw_config.get('file_filters', {}).keys())}")
        excluded_dirs = raw_config.get('file_filters', {}).get('excluded_dirs', [])
        print(f"   excluded_dirs: {excluded_dirs}")
        print(f"   'vendor' in excluded_dirs: {'vendor' in excluded_dirs}")
    
    # Step 2: Test config module loading
    print(f"\n⚙️  Testing config module loading:")
    try:
        from config import configs, DEFAULT_EXCLUDED_DIRS, DEFAULT_EXCLUDED_FILES
        
        print(f"   configs type: {type(configs)}")
        print(f"   configs keys: {list(configs.keys())}")
        
        file_filters_config = configs.get("file_filters", {})
        print(f"   file_filters_config: {file_filters_config}")
        
        repo_excluded_dirs = file_filters_config.get("excluded_dirs", [])
        print(f"   repo_excluded_dirs: {repo_excluded_dirs}")
        print(f"   'vendor' in repo_excluded_dirs: {'vendor' in repo_excluded_dirs}")
        
        print(f"\n📋 DEFAULT_EXCLUDED_DIRS (first 5): {DEFAULT_EXCLUDED_DIRS[:5]}")
        
        # Step 3: Test the combination logic
        excluded_dirs = None  # Simulate function parameters
        final_excluded_dirs = DEFAULT_EXCLUDED_DIRS + (excluded_dirs or [])
        final_excluded_dirs.extend(repo_excluded_dirs)
        
        normalized_excluded_dirs = [p.strip('./').strip('/') for p in final_excluded_dirs if p.strip('./').strip('/')]
        
        print(f"\n🔄 Processing logic:")
        print(f"   DEFAULT_EXCLUDED_DIRS count: {len(DEFAULT_EXCLUDED_DIRS)}")
        print(f"   repo_excluded_dirs count: {len(repo_excluded_dirs)}")
        print(f"   final_excluded_dirs count: {len(final_excluded_dirs)}")
        print(f"   normalized_excluded_dirs count: {len(normalized_excluded_dirs)}")
        
        print(f"\n🎯 Key checks:")
        print(f"   'vendor' in final_excluded_dirs: {'vendor' in final_excluded_dirs}")
        print(f"   'vendor' in normalized_excluded_dirs: {'vendor' in normalized_excluded_dirs}")
        
        # Show some examples of normalization
        print(f"\n🧪 Normalization examples:")
        examples = ["./vendor/", "vendor", ".venv", "./node_modules/"]
        for example in examples:
            if example in final_excluded_dirs:
                normalized = example.strip('./').strip('/')
                print(f"   '{example}' → '{normalized}' (in final list: True)")
        
        # Check if vendor appears in different forms
        vendor_forms = [d for d in final_excluded_dirs if 'vendor' in d.lower()]
        print(f"\n🔍 All 'vendor' related entries in final_excluded_dirs: {vendor_forms}")
        
        vendor_normalized = [d for d in normalized_excluded_dirs if 'vendor' in d.lower()]
        print(f"   In normalized list: {vendor_normalized}")
        
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        import traceback
        traceback.print_exc()

def test_path_matching():
    """Test path matching logic specifically."""
    
    print(f"\n🧪 Testing Path Matching Logic")
    print("-" * 40)
    
    try:
        from config import configs, DEFAULT_EXCLUDED_DIRS
        
        # Replicate the exact logic
        file_filters_config = configs.get("file_filters", {})
        repo_excluded_dirs = file_filters_config.get("excluded_dirs", [])
        
        final_excluded_dirs = DEFAULT_EXCLUDED_DIRS + repo_excluded_dirs
        normalized_excluded_dirs = [p.strip('./').strip('/') for p in final_excluded_dirs if p.strip('./').strip('/')]
        
        # Test specific paths
        test_paths = [
            "vendor",
            "vendor/package",
            "src/vendor",
            "lib/third_party"
        ]
        
        for test_path in test_paths:
            normalized_path = test_path.replace('\\\\', '/')
            path_components = normalized_path.split('/')
            
            should_skip = False
            matching_component = None
            
            for component in path_components:
                if component in normalized_excluded_dirs:
                    should_skip = True
                    matching_component = component
                    break
            
            result = "SKIP" if should_skip else "PROCESS"
            reason = f" (component '{matching_component}' excluded)" if matching_component else ""
            print(f"   '{test_path}' → {result}{reason}")
            
    except Exception as e:
        print(f"❌ Error in path matching test: {e}")

if __name__ == "__main__":
    test_config_loading()
    test_path_matching()