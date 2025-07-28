#!/usr/bin/env python3
"""
Quick test to verify directory filtering logic works correctly.
"""

import os
import sys
import fnmatch

# Add the api directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import configs, DEFAULT_EXCLUDED_DIRS, DEFAULT_EXCLUDED_FILES

def test_directory_filtering():
    """Test directory filtering with realistic scenarios."""
    
    print("🧪 Directory Filtering Test")
    print("=" * 50)
    
    # Simulate the filtering logic from data_pipeline.py
    excluded_dirs = []  # From function parameters
    excluded_files = []  # From function parameters
    
    # Get configuration
    file_filters_config = configs.get("file_filters", {})
    excluded_filename_patterns = file_filters_config.get("excluded_filename_patterns", [])
    repo_excluded_dirs = file_filters_config.get("excluded_dirs", [])
    
    # Combine all exclusions
    final_excluded_dirs = DEFAULT_EXCLUDED_DIRS + (excluded_dirs or [])
    final_excluded_files = DEFAULT_EXCLUDED_FILES + (excluded_files or [])
    final_excluded_dirs.extend(repo_excluded_dirs)
    
    # Normalize exclusion patterns for reliable matching
    normalized_excluded_dirs = [p.strip('./').strip('/') for p in final_excluded_dirs if p.strip('./').strip('/')]
    
    print(f"📁 Normalized excluded directories ({len(normalized_excluded_dirs)}):") 
    for d in sorted(set(normalized_excluded_dirs))[:10]:  # Show first 10
        print(f"   - {d}")
    if len(normalized_excluded_dirs) > 10:
        print(f"   ... and {len(normalized_excluded_dirs) - 10} more")
    
    print(f"\\n📄 Excluded filename patterns: {excluded_filename_patterns}")
    print(f"\\n🗂️ Total excluded file patterns: {len(final_excluded_files)}")
    
    # Test cases for directories that should be filtered
    test_paths = [
        ("vendor", "vendor/package.go"),
        ("node_modules", "node_modules/package/index.js"),
        ("build", "build/output.js"),
        ("src", "src/main.go"),  # Should NOT be excluded
        ("app", "app/service.js"),  # Should NOT be excluded
        ("lib", "lib/utils.py"),  # Should NOT be excluded
    ]
    
    print(f"\\n🧪 Testing directory exclusion logic:")
    print("-" * 40)
    
    for dir_name, test_file_path in test_paths:
        # Test if directory should be excluded
        should_exclude_dir = dir_name in normalized_excluded_dirs
        
        # Test if file should be excluded by filename pattern
        filename = os.path.basename(test_file_path)
        should_exclude_file_pattern = False
        for pattern in excluded_filename_patterns:
            if fnmatch.fnmatch(filename, pattern) or fnmatch.fnmatch(test_file_path, pattern):
                should_exclude_file_pattern = True
                break
        
        # Test if file should be excluded by file pattern
        should_exclude_file = False
        for pattern in final_excluded_files:
            if fnmatch.fnmatch(test_file_path, pattern):
                should_exclude_file = True
                break
        
        result = "❌ EXCLUDE" if (should_exclude_dir or should_exclude_file_pattern or should_exclude_file) else "✅ INCLUDE"\n        reasons = []\n        if should_exclude_dir:\n            reasons.append(f"directory '{dir_name}' excluded")\n        if should_exclude_file_pattern:\n            reasons.append("filename pattern match")\n        if should_exclude_file:\n            reasons.append("file pattern match")\n        \n        reason_text = f" ({', '.join(reasons)})" if reasons else ""\n        print(f"  {result}: {test_file_path}{reason_text}")\n    \n    # Test mock file patterns specifically\n    print(f"\\n🎭 Testing mock file patterns:")  \n    print("-" * 30)\n    \n    mock_files = [\n        "handler_mock.go",\n        "service_mock.go", \n        "utils_test.go",\n        "component.test.js",\n        "api.spec.ts"\n    ]\n    \n    for mock_file in mock_files:\n        should_exclude = False\n        matching_pattern = None\n        \n        for pattern in excluded_filename_patterns:\n            if fnmatch.fnmatch(mock_file, pattern):\n                should_exclude = True\n                matching_pattern = pattern\n                break\n        \n        result = "❌ EXCLUDE" if should_exclude else "✅ INCLUDE"\n        reason_text = f" (matches '{matching_pattern}')" if matching_pattern else ""\n        print(f"  {result}: {mock_file}{reason_text}")\n\nif __name__ == "__main__":\n    test_directory_filtering()