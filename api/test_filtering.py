#!/usr/bin/env python3
"""
Test script to verify file filtering functionality.
Usage: python test_filtering.py [path_to_test]
"""

import os
import sys
import fnmatch
import json
from pathlib import Path

# Add the api directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import configs

def test_filtering_logic():
    """Test the filtering logic with some sample paths."""
    
    # Get filtering configuration
    file_filters_config = configs.get("file_filters", {})
    excluded_dirs = file_filters_config.get("excluded_dirs", [])
    excluded_filename_patterns = file_filters_config.get("excluded_filename_patterns", [])
    excluded_files = file_filters_config.get("excluded_files", [])
    
    print("🔍 File Filtering Test")
    print("=" * 50)
    
    print(f"📁 Excluded directories: {excluded_dirs}")
    print(f"📄 Excluded filename patterns: {excluded_filename_patterns}")
    print(f"📋 Total excluded files: {len(excluded_files)}")
    print()
    
    # Test cases
    test_cases = [
        # Directory-based tests
        "vendor/package.go",
        "vendor/subdir/file.go", 
        "src/vendor/file.go",  # vendor as subdirectory
        "node_modules/package/index.js",
        "build/output.js",
        
        # Filename pattern tests
        "main_test.go",
        "handler_mock.go", 
        "service_test.go",
        "utils.test.js",
        "component.spec.ts",
        
        # Normal files that should NOT be excluded
        "src/main.go",
        "lib/handler.go",
        "app/service.js",
        "README.md",
    ]
    
    print("📋 Testing file paths:")
    print("-" * 30)
    
    for test_path in test_cases:
        should_exclude = False
        reason = None
        
        # Test directory exclusion
        path_parts = test_path.split('/')
        for part in path_parts[:-1]:  # Exclude the filename itself
            if part in excluded_dirs:
                should_exclude = True
                reason = f"directory '{part}' is excluded"
                break
        
        # Test filename pattern exclusion
        if not should_exclude:
            filename = os.path.basename(test_path)
            for pattern in excluded_filename_patterns:
                if fnmatch.fnmatch(filename, pattern) or fnmatch.fnmatch(test_path, pattern):
                    should_exclude = True
                    reason = f"matches filename pattern '{pattern}'"
                    break
        
        # Test specific file exclusion
        if not should_exclude:
            for pattern in excluded_files:
                if fnmatch.fnmatch(test_path, pattern):
                    should_exclude = True
                    reason = f"matches excluded file pattern '{pattern}'"
                    break
        
        status = "❌ EXCLUDE" if should_exclude else "✅ INCLUDE"
        reason_text = f" ({reason})" if reason else ""
        print(f"  {status}: {test_path}{reason_text}")
    
    print()
    print("💡 To add custom exclusions, edit api/config/repo.json")
    print("   - Add directories to 'excluded_dirs'")
    print("   - Add filename patterns to 'excluded_filename_patterns'")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_path = sys.argv[1]
        print(f"Testing with directory: {test_path}")
        # Could add actual directory walking test here
    else:
        test_filtering_logic()