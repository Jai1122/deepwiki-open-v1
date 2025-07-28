#!/usr/bin/env python3
"""
Diagnostic script to test the actual file filtering logic
as implemented in data_pipeline.py without running the full application.
"""

import os
import sys
import fnmatch
import tempfile
import shutil

# Add the api directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import configs, DEFAULT_EXCLUDED_DIRS, DEFAULT_EXCLUDED_FILES

def create_test_repo():
    """Create a test repository structure to test filtering."""
    temp_dir = tempfile.mkdtemp(prefix="deepwiki_test_")
    
    # Create test directory structure
    test_files = [
        "README.md",
        "main.go", 
        "src/handler.go",
        "src/service.go",
        "src/handler_mock.go",
        "src/service_test.go",
        "vendor/package.go",
        "vendor/subdir/utils.go",
        "node_modules/package/index.js",
        "build/output.js",
        "dist/bundle.js",
        ".git/config",
        "tests/unit_test.go",
        "app/main_test.go",
        "lib/utils.spec.ts"
    ]
    
    for file_path in test_files:
        full_path = os.path.join(temp_dir, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'w') as f:
            f.write(f"// Content of {file_path}\\n")
    
    return temp_dir

def test_filtering_logic(repo_path):
    """Test the exact filtering logic from data_pipeline.py"""
    
    print(f"🔍 Testing filtering on: {repo_path}")
    print("=" * 60)
    
    # Replicate the exact logic from read_all_documents
    excluded_dirs = None
    excluded_files = None
    included_dirs = None  
    included_files = None
    
    final_excluded_dirs = DEFAULT_EXCLUDED_DIRS + (excluded_dirs or [])
    final_excluded_files = DEFAULT_EXCLUDED_FILES + (excluded_files or [])
    
    # Get filename patterns for exclusion from repo config
    file_filters_config = configs.get("file_filters", {})
    excluded_filename_patterns = file_filters_config.get("excluded_filename_patterns", [])
    
    # Also add excluded_dirs from repo config if not already included
    repo_excluded_dirs = file_filters_config.get("excluded_dirs", [])
    final_excluded_dirs.extend(repo_excluded_dirs)

    # Normalize exclusion patterns for reliable matching
    normalized_excluded_dirs = [p.strip('./').strip('/') for p in final_excluded_dirs if p.strip('./').strip('/')]
    
    print(f"📁 Excluded directories ({len(normalized_excluded_dirs)}):")
    for d in sorted(set(normalized_excluded_dirs))[:15]:
        print(f"   - {d}")
    if len(normalized_excluded_dirs) > 15:
        print(f"   ... and {len(normalized_excluded_dirs) - 15} more")
    
    print(f"\\n📄 Excluded filename patterns: {excluded_filename_patterns}")
    print(f"🗂️ Total excluded file patterns: {len(final_excluded_files)}")
    
    # Replicate the directory walking logic
    processed_files = []
    rejected_files = []
    
    print(f"\\n🚶 Walking directory structure:")
    print("-" * 40)
    
    for root, dirs, files in os.walk(repo_path, topdown=True):
        # Get the current directory relative to the base path
        current_dir_relative = os.path.relpath(root, repo_path)
        normalized_current_dir = current_dir_relative.replace('\\\\', '/')
        
        # Handle root directory case  
        if normalized_current_dir == '.':
            normalized_current_dir = ''
        
        print(f"\\n📂 Processing directory: '{normalized_current_dir}' (original: '{current_dir_relative}')")
        
        # Check if current directory should be excluded (skip entire subtree)
        should_skip_current_dir = False
        if normalized_current_dir:
            # Check if any part of current path matches excluded directories
            path_parts = normalized_current_dir.split('/')
            for part in path_parts:
                if part in normalized_excluded_dirs:
                    should_skip_current_dir = True
                    print(f"   ❌ Directory '{part}' is excluded -> SKIPPING ENTIRE SUBTREE")
                    break
        
        if should_skip_current_dir:
            # Skip this entire directory tree
            dirs.clear()  # Don't recurse into subdirectories
            continue
        
        # Filter immediate subdirectories to prevent traversing excluded ones
        original_dirs = dirs[:]
        dirs[:] = [d for d in dirs if d not in normalized_excluded_dirs]
        
        filtered_dirs = set(original_dirs) - set(dirs)
        if filtered_dirs:
            print(f"   🚫 Filtered subdirs: {filtered_dirs}")
        if dirs:
            print(f"   ✅ Will recurse into: {dirs}")
        
        # Process files in current directory
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, repo_path)
            normalized_relative_path = relative_path.replace('\\\\', '/')
            
            # Skip excluded files by filename pattern (check both filename and full path)
            filename_excluded = False
            matching_pattern = None
            for pattern in excluded_filename_patterns:
                if fnmatch.fnmatch(file, pattern) or fnmatch.fnmatch(normalized_relative_path, pattern):
                    filename_excluded = True
                    matching_pattern = pattern
                    break
            
            if filename_excluded:
                rejected_files.append(f"{normalized_relative_path} (filename pattern: {matching_pattern})")
                print(f"     ❌ {file} -> matches pattern '{matching_pattern}'")
                continue

            # Skip files matching excluded file patterns
            file_pattern_excluded = False
            matching_file_pattern = None
            for pattern in final_excluded_files:
                if fnmatch.fnmatch(normalized_relative_path, pattern):
                    file_pattern_excluded = True
                    matching_file_pattern = pattern
                    break
            
            if file_pattern_excluded:
                rejected_files.append(f"{normalized_relative_path} (file pattern: {matching_file_pattern})")
                print(f"     ❌ {file} -> matches file pattern '{matching_file_pattern}'")
                continue
            
            # File passes all filters
            processed_files.append(normalized_relative_path)
            print(f"     ✅ {file}")
    
    print(f"\\n📊 Final Results:")
    print("=" * 30)
    print(f"✅ Processed files: {len(processed_files)}")
    print(f"❌ Rejected files: {len(rejected_files)}")
    
    if processed_files:
        print(f"\\n📋 Processed files:")
        for f in processed_files:
            print(f"   - {f}")
    
    if rejected_files:
        print(f"\\n🚫 Rejected files:")
        for f in rejected_files[:10]:  # Show first 10
            print(f"   - {f}")
        if len(rejected_files) > 10:
            print(f"   ... and {len(rejected_files) - 10} more")
    
    return len(processed_files) == 0

def main():
    print("🧪 DeepWiki File Filtering Diagnostics")
    print("=" * 50)
    
    # Create test repository
    test_repo = create_test_repo()
    
    try:
        # Test filtering
        no_files_processed = test_filtering_logic(test_repo)
        
        if no_files_processed:
            print("\\n⚠️  WARNING: NO FILES WERE PROCESSED!")
            print("This would cause the application to stop with 'done' message.")
            print("\\n🔧 Possible fixes:")
            print("1. Check if your repo.json exclusions are too aggressive")
            print("2. Verify that your repository contains processable files")
            print("3. Check if directory exclusions are preventing access to source files")
        else:
            print("\\n✅ SUCCESS: Files were processed correctly!")
            print("The filtering logic appears to be working properly.")
            
    finally:
        # Cleanup
        shutil.rmtree(test_repo)
        print(f"\\n🧹 Cleaned up test directory: {test_repo}")

if __name__ == "__main__":
    main()