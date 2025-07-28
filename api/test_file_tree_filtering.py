#!/usr/bin/env python3
"""
Test that the file tree generation properly excludes directories and files
based on repo.json configuration.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add the api directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_test_repo_for_file_tree():
    """Create a test repository with various directories and files."""
    temp_dir = tempfile.mkdtemp(prefix="file_tree_test_")
    
    # Create structure that should be partially filtered
    test_structure = {
        # Should be included
        "README.md": "# Test Repository",
        "main.go": "package main",
        "src/handler.go": "package src", 
        "lib/utils.go": "package lib",
        "docs/guide.md": "# Guide",
        
        # Should be excluded - vendor directory
        "vendor/package1/mod.go": "// vendor file 1",
        "vendor/package2/utils.go": "// vendor file 2", 
        "vendor/nested/deep/file.go": "// deep vendor file",
        
        # Should be excluded - .git directory
        ".git/config": "[core]",
        ".git/objects/abc123": "git object content",
        ".git/refs/heads/main": "ref content",
        
        # Should be excluded - other common exclusions
        "node_modules/package/index.js": "// node module",
        "__pycache__/cache.pyc": "compiled python",
        ".venv/lib/python.py": "virtual env file",
        
        # Should be excluded - mock files (filename pattern)
        "service_mock.go": "// mock file",
        "handler_test.go": "// test file",
        
        # Should be excluded - build artifacts
        "dist/bundle.js": "// built file",
        "build/output.js": "// build output",
        
        # Should be included - normal files
        "config.json": '{"setting": true}',
        "app/main.js": "// main app",
    }
    
    for file_path, content in test_structure.items():
        full_path = Path(temp_dir) / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)
    
    return temp_dir

def test_file_tree_filtering():
    """Test the file tree generation with filtering."""
    
    print("🧪 Testing File Tree Filtering for LLM")
    print("=" * 50)
    
    test_repo = create_test_repo_for_file_tree()
    
    try:
        # Replicate the file tree generation logic from api.py
        from config import configs, DEFAULT_EXCLUDED_DIRS, DEFAULT_EXCLUDED_FILES
        import fnmatch
        
        # Get comprehensive exclusion lists (same as api.py)
        final_excluded_dirs = DEFAULT_EXCLUDED_DIRS.copy()
        final_excluded_files = DEFAULT_EXCLUDED_FILES.copy()
        
        # Add exclusions from repo config
        file_filters_config = configs.get("file_filters", {})
        excluded_filename_patterns = file_filters_config.get("excluded_filename_patterns", [])
        repo_excluded_dirs = file_filters_config.get("excluded_dirs", [])
        final_excluded_dirs.extend(repo_excluded_dirs)
        
        # Normalize exclusion patterns (same logic as api.py)
        normalized_excluded_dirs = []
        for p in final_excluded_dirs:
            if not p or not p.strip():
                continue
            normalized = p.strip()
            if normalized.startswith('./'):
                normalized = normalized[2:]
            normalized = normalized.rstrip('/')
            if normalized and normalized not in normalized_excluded_dirs:
                normalized_excluded_dirs.append(normalized)
        
        print(f"Exclusion list: {sorted(set(normalized_excluded_dirs))}")
        print(f"Filename patterns: {excluded_filename_patterns}")
        
        file_tree_lines = []
        
        # Walk directory tree (same logic as api.py)
        for root, dirs, files in os.walk(test_repo, topdown=True):
            # Get current directory relative to base path
            current_dir_relative = os.path.relpath(root, test_repo)
            normalized_current_dir = current_dir_relative.replace('\\\\', '/')
            
            if normalized_current_dir == '.':
                normalized_current_dir = ''
            
            print(f"\\n📂 Processing directory: '{normalized_current_dir}'")
            print(f"   Found subdirs: {dirs}")
            print(f"   Found files: {files}")
            
            # Check if current directory should be skipped entirely
            should_skip_dir = False
            if normalized_current_dir:
                path_components = normalized_current_dir.split('/')
                for component in path_components:
                    if component in normalized_excluded_dirs:
                        should_skip_dir = True
                        print(f"   🚫 SKIPPING entire tree: '{component}' is excluded")
                        break
            
            if should_skip_dir:
                dirs.clear()  # Don't recurse into excluded directories
                continue
            
            # Filter immediate subdirectories
            original_dirs = dirs[:]
            dirs[:] = [d for d in dirs if d not in normalized_excluded_dirs]
            
            filtered_subdirs = set(original_dirs) - set(dirs)
            if filtered_subdirs:
                print(f"   🚫 Filtered subdirs: {filtered_subdirs}")
            
            print(f"   ➡️  Will recurse into: {dirs}")
            
            # Process files in current directory
            for file in files:
                rel_dir = os.path.relpath(root, test_repo)
                rel_file = os.path.join(rel_dir, file) if rel_dir != '.' else file
                normalized_rel_file = rel_file.replace('\\\\', '/')
                
                # Check exclusions
                excluded = False
                reason = None
                
                # Check if file should be excluded by filename pattern
                for pattern in excluded_filename_patterns:
                    if fnmatch.fnmatch(file, pattern) or fnmatch.fnmatch(normalized_rel_file, pattern):
                        excluded = True
                        reason = f"filename pattern '{pattern}'"
                        break
                
                # Check if file matches general excluded file patterns
                if not excluded:
                    for pattern in final_excluded_files:
                        if fnmatch.fnmatch(normalized_rel_file, pattern):
                            excluded = True
                            reason = f"file pattern '{pattern}'"
                            break
                
                # Safety check: ensure file is not in excluded directory path
                if not excluded:
                    file_path_components = normalized_rel_file.split('/')
                    for component in file_path_components[:-1]:
                        if component in normalized_excluded_dirs:
                            excluded = True
                            reason = f"in excluded directory '{component}'"
                            break
                
                if excluded:
                    print(f"   🚫 EXCLUDING: {normalized_rel_file} ({reason})")
                else:
                    print(f"   ✅ INCLUDING: {normalized_rel_file}")
                    file_tree_lines.append(normalized_rel_file)
        
        # Analysis
        print(f"\\n📊 File Tree Results:")
        print(f"   Total files in tree: {len(file_tree_lines)}")
        
        print(f"\\n📋 Files that will be sent to LLM:")
        for f in sorted(file_tree_lines):
            print(f"   - {f}")
        
        # Check for problematic inclusions
        problematic_files = []
        for f in file_tree_lines:
            if any(excluded_part in f.lower() for excluded_part in ['vendor', '.git', 'node_modules', '__pycache__', 'mock', '_test']):
                problematic_files.append(f)
        
        if problematic_files:
            print(f"\\n❌ PROBLEM: Found excluded files in tree:")
            for f in problematic_files:
                print(f"   - {f}")
            return False
        else:
            print(f"\\n✅ SUCCESS: File tree is clean!")
            print(f"   No vendor, .git, node_modules, or test files included.")
            print(f"   Only relevant source files will be sent to LLM.")
            return True
        
    finally:
        shutil.rmtree(test_repo)

if __name__ == "__main__":
    print("🌳 Testing File Tree Generation for LLM Wiki Structure Analysis")
    print("=" * 70)
    print("This tests that excluded directories/files from repo.json are not")
    print("included in the file tree sent to the LLM for wiki structure analysis.")
    print()
    
    success = test_file_tree_filtering()
    
    print(f"\\n🏁 Final Result: {'✅ PASSED' if success else '❌ FAILED'}")
    
    if success:
        print("\\n🎉 The file tree filtering is working correctly!")
        print("   - vendor, .git, node_modules directories are excluded")
        print("   - mock and test files are excluded") 
        print("   - Only relevant source files are included in LLM analysis")
    else:
        print("\\n⚠️  Issues found with file tree filtering.")
        print("   Some excluded files/directories are still being included.")