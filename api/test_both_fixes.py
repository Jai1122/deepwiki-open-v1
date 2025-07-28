#!/usr/bin/env python3
"""
Test both fixes: .git directory exclusion and binary file detection.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add the api directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import is_text_file
from config import configs, DEFAULT_EXCLUDED_DIRS

def create_test_repo_with_git_and_binary():
    """Create test repo with .git directory and binary files."""
    temp_dir = tempfile.mkdtemp(prefix="git_binary_test_")
    
    # Create structure with .git and binary files
    test_files = {
        "main.go": "package main\\nfunc main() {}",
        "src/handler.go": "package src",
        ".git/config": "[core]\\nrepositoryformatversion = 0",
        ".git/objects/abc123": b"\\x00\\x01\\x02binary_git_object",  # Binary
        "vendor/pkg/mod.go": "// vendor file",
        "image.png": b"\\x89PNG\\r\\n\\x1a\\n\\x00\\x00\\x00\\rIHDR",  # Binary PNG
        "doc.pdf": b"%PDF-1.4\\n1 0 obj",  # Binary PDF
        "README.md": "# Project\\nThis is a readme file."
    }
    
    for file_path, content in test_files.items():
        full_path = Path(temp_dir) / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(content, bytes):
            full_path.write_bytes(content)
        else:
            full_path.write_text(content)
    
    return temp_dir

def test_git_directory_exclusion():
    """Test that .git directory is properly excluded."""
    print("🧪 Testing .git Directory Exclusion")
    print("-" * 40)
    
    # Check configuration
    file_filters_config = configs.get("file_filters", {})
    repo_excluded_dirs = file_filters_config.get("excluded_dirs", [])
    
    final_excluded_dirs = DEFAULT_EXCLUDED_DIRS + repo_excluded_dirs
    
    # Test normalization
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
    
    print(f"Excluded directories: {sorted(set(normalized_excluded_dirs))}")
    
    git_in_exclusions = '.git' in normalized_excluded_dirs
    print(f"'.git' in exclusions: {git_in_exclusions}")
    
    if git_in_exclusions:
        print("✅ .git directory exclusion configuration is correct")
    else:
        print("❌ .git directory is NOT in exclusions!")
        
    return git_in_exclusions

def test_binary_file_detection():
    """Test binary file detection."""
    print("\\n🧪 Testing Binary File Detection")
    print("-" * 40)
    
    test_repo = create_test_repo_with_git_and_binary()
    
    try:
        test_cases = [
            ("main.go", True, "Go source file"),
            ("README.md", True, "Markdown file"),
            ("image.png", False, "PNG image"),
            ("doc.pdf", False, "PDF document"),
            (".git/config", True, "Git config (text)"),
            (".git/objects/abc123", False, "Git object (binary)")
        ]
        
        all_correct = True
        
        for file_path, expected_is_text, description in test_cases:
            full_path = os.path.join(test_repo, file_path)
            if os.path.exists(full_path):
                is_text = is_text_file(full_path)
                result = "✅ CORRECT" if is_text == expected_is_text else "❌ WRONG"
                expected_str = "TEXT" if expected_is_text else "BINARY"
                actual_str = "TEXT" if is_text else "BINARY"
                
                print(f"  {result}: {file_path}")
                print(f"    {description}")
                print(f"    Expected: {expected_str}, Got: {actual_str}")
                
                if is_text != expected_is_text:
                    all_correct = False
            else:
                print(f"  ⚠️  MISSING: {file_path}")
        
        return all_correct
        
    finally:
        shutil.rmtree(test_repo)

def test_directory_walking_logic():
    """Test the actual directory walking with exclusions."""
    print("\\n🧪 Testing Directory Walking Logic")
    print("-" * 40)
    
    test_repo = create_test_repo_with_git_and_binary()
    
    try:
        # Replicate the exclusion logic
        file_filters_config = configs.get("file_filters", {})
        repo_excluded_dirs = file_filters_config.get("excluded_dirs", [])
        final_excluded_dirs = DEFAULT_EXCLUDED_DIRS + repo_excluded_dirs
        
        # Normalize
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
        
        print(f"Normalized exclusions: {sorted(set(normalized_excluded_dirs))}")
        
        processed_files = []
        skipped_dirs = []
        
        for root, dirs, files in os.walk(test_repo, topdown=True):
            current_dir_relative = os.path.relpath(root, test_repo)
            normalized_current_dir = current_dir_relative.replace('\\\\', '/')
            
            if normalized_current_dir == '.':
                normalized_current_dir = ''
            
            # Check if current directory should be skipped
            should_skip = False
            if normalized_current_dir:
                path_components = normalized_current_dir.split('/')
                for component in path_components:
                    if component in normalized_excluded_dirs:
                        should_skip = True
                        skipped_dirs.append(normalized_current_dir)
                        print(f"  🚫 SKIPPING: {normalized_current_dir} (component '{component}' excluded)")
                        break
            
            if should_skip:
                dirs.clear()
                continue
            
            # Filter subdirectories
            original_dirs = dirs[:]
            dirs[:] = [d for d in dirs if d not in normalized_excluded_dirs]
            
            filtered = set(original_dirs) - set(dirs)
            if filtered:
                print(f"  🚫 Filtered subdirs in '{normalized_current_dir or 'root'}': {filtered}")
                skipped_dirs.extend([f"{normalized_current_dir}/{d}" if normalized_current_dir else d for d in filtered])
            
            # Process files
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, test_repo)
                
                if is_text_file(file_path):
                    processed_files.append(relative_path)
                    print(f"  ✅ PROCESSING: {relative_path}")
                else:
                    print(f"  🚫 SKIPPING BINARY: {relative_path}")
        
        print(f"\\n📊 Summary:")
        print(f"  Processed files: {len(processed_files)}")
        print(f"  Skipped directories: {len(skipped_dirs)}")
        
        # Check for issues
        git_files = [f for f in processed_files if '.git' in f]
        binary_files = [f for f in processed_files if f.endswith(('.png', '.pdf'))]
        
        success = len(git_files) == 0 and len(binary_files) == 0
        
        if git_files:
            print(f"  ❌ PROBLEM: Found .git files: {git_files}")
        if binary_files:
            print(f"  ❌ PROBLEM: Found binary files: {binary_files}")
        
        if success:
            print(f"  ✅ SUCCESS: No .git or binary files processed!")
        
        return success
        
    finally:
        shutil.rmtree(test_repo)

if __name__ == "__main__":
    print("🔧 Testing Both Fixes: .git Exclusion + Binary Detection")
    print("=" * 60)
    
    test1 = test_git_directory_exclusion()
    test2 = test_binary_file_detection()
    test3 = test_directory_walking_logic()
    
    print(f"\\n📊 Overall Results:")
    print(f"  .git exclusion config: {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"  Binary file detection: {'✅ PASS' if test2 else '❌ FAIL'}")
    print(f"  Directory walking logic: {'✅ PASS' if test3 else '❌ FAIL'}")
    
    if all([test1, test2, test3]):
        print(f"\\n🎉 All tests PASSED! Both issues should be fixed.")
    else:
        print(f"\\n⚠️  Some tests FAILED. Issues may still exist.")