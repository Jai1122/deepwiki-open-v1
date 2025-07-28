#!/usr/bin/env python3
"""
Test script to specifically debug vendor directory filtering.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add the api directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import configs, DEFAULT_EXCLUDED_DIRS, DEFAULT_EXCLUDED_FILES

def create_test_repo_with_vendor():
    """Create a test repository with vendor directory to test filtering."""
    temp_dir = tempfile.mkdtemp(prefix="vendor_filter_test_")
    
    # Create test directory structure with vendor
    test_structure = {
        "main.go": "package main",
        "src/handler.go": "package src", 
        "vendor/package1/mod.go": "// vendor file 1",
        "vendor/package2/utils.go": "// vendor file 2",
        "vendor/subdir/nested.go": "// nested vendor file",
        "lib/utils.go": "package lib",
        "third_party/external.go": "// third party file"
    }
    
    for file_path, content in test_structure.items():
        full_path = Path(temp_dir) / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)
    
    return temp_dir

def test_vendor_filtering():
    """Test the vendor filtering logic."""
    
    print("🧪 Testing Vendor Directory Filtering")
    print("=" * 60)
    
    # Create test repo
    test_repo = create_test_repo_with_vendor()
    print(f"Created test repo at: {test_repo}")
    
    try:
        # Show the directory structure
        print("\n📁 Test Repository Structure:")
        for root, dirs, files in os.walk(test_repo):
            level = root.replace(test_repo, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")
        
        # Test the configuration loading
        print(f"\n⚙️  Configuration Check:")
        
        # Replicate the exact filtering logic from data_pipeline.py
        excluded_dirs = None
        excluded_files = None
        
        final_excluded_dirs = DEFAULT_EXCLUDED_DIRS + (excluded_dirs or [])
        final_excluded_files = DEFAULT_EXCLUDED_FILES + (excluded_files or [])
        
        # Get filename patterns for exclusion from repo config
        file_filters_config = configs.get("file_filters", {})
        excluded_filename_patterns = file_filters_config.get("excluded_filename_patterns", [])
        
        # Also add excluded_dirs from repo config if not already included
        repo_excluded_dirs = file_filters_config.get("excluded_dirs", [])
        print(f"  Repo config excluded_dirs: {repo_excluded_dirs}")
        
        final_excluded_dirs.extend(repo_excluded_dirs)
        
        # Normalize exclusion patterns for reliable matching
        normalized_excluded_dirs = [p.strip('./').strip('/') for p in final_excluded_dirs if p.strip('./').strip('/')]
        
        print(f"  All excluded directories: {sorted(set(normalized_excluded_dirs))}")
        print(f"  'vendor' in exclusions: {'vendor' in normalized_excluded_dirs}")
        print(f"  'third_party' in exclusions: {'third_party' in normalized_excluded_dirs}")
        
        # Now test the directory walking logic
        print(f"\n🚶 Walking Directory Structure:")
        print("-" * 40)
        
        processed_files = []
        skipped_directories = []
        
        for root, dirs, files in os.walk(test_repo, topdown=True):
            # Get current directory relative to the base path
            current_dir_relative = os.path.relpath(root, test_repo) 
            normalized_current_dir = current_dir_relative.replace('\\\\', '/')
            
            # Skip the root directory case completely to avoid issues
            if normalized_current_dir == '.':
                normalized_current_dir = ''
            
            print(f"\n📂 Current directory: '{normalized_current_dir}' (from '{current_dir_relative}')")
            print(f"   Subdirs found: {dirs}")
            print(f"   Files found: {files}")
            
            # Check if any part of the current path should cause us to skip this directory
            should_skip_dir = False
            excluded_component = None
            if normalized_current_dir:
                # Split path and check each component
                path_components = normalized_current_dir.split('/')
                print(f"   Path components: {path_components}")
                
                for component in path_components:
                    print(f"   Checking component '{component}' against exclusions...")
                    if component in normalized_excluded_dirs:
                        should_skip_dir = True
                        excluded_component = component
                        print(f"   ❌ SKIPPING: '{component}' is in excluded list")
                        break
                    else:
                        print(f"   ✅ OK: '{component}' not in excluded list")
            
            if should_skip_dir:
                skipped_directories.append(normalized_current_dir)
                print(f"   🚫 SKIPPING ENTIRE TREE: {normalized_current_dir}")
                dirs.clear()  # Prevent recursion into this directory tree
                continue
            
            # Filter immediate subdirectories to prevent walking into them
            original_subdirs = dirs[:]
            dirs[:] = [d for d in dirs if d not in normalized_excluded_dirs]
            
            # Log what subdirectories were filtered
            filtered_subdirs = set(original_subdirs) - set(dirs)
            if filtered_subdirs:
                print(f"   🚫 Filtered subdirs: {filtered_subdirs}")
                skipped_directories.extend([f"{normalized_current_dir}/{d}" if normalized_current_dir else d for d in filtered_subdirs])
            
            print(f"   ➡️  Will recurse into: {dirs}")
            
            # Process files (simplified - just count them)
            for file in files:
                relative_path = os.path.relpath(os.path.join(root, file), test_repo)
                normalized_relative_path = relative_path.replace('\\\\', '/')
                processed_files.append(normalized_relative_path)
                print(f"   📄 Processing file: {normalized_relative_path}")
        
        # Summary
        print(f"\n📊 Results:")
        print(f"   Processed files: {len(processed_files)}")
        print(f"   Skipped directories: {len(skipped_directories)}")
        
        print(f"\n✅ Processed files:")
        for f in sorted(processed_files):
            print(f"   - {f}")
        
        print(f"\n🚫 Skipped directories:")
        for d in sorted(skipped_directories):
            print(f"   - {d}")
        
        # Check if vendor files were excluded
        vendor_files = [f for f in processed_files if 'vendor' in f]
        if vendor_files:
            print(f"\n❌ PROBLEM: Found {len(vendor_files)} vendor files in processed list:")
            for f in vendor_files:
                print(f"   - {f}")
            print("   This indicates the vendor filtering is NOT working!")
        else:
            print(f"\n✅ SUCCESS: No vendor files found in processed list!")
            print("   Vendor filtering is working correctly!")
            
    finally:
        # Cleanup
        shutil.rmtree(test_repo)
        print(f"\n🧹 Cleaned up test directory")

if __name__ == "__main__":
    test_vendor_filtering()