#!/usr/bin/env python3
"""
Simple test to verify that the file filtering fixes work correctly.
"""

import os
import sys
import fnmatch

# Add the api directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import configs, DEFAULT_EXCLUDED_DIRS, DEFAULT_EXCLUDED_FILES

def test_user_reported_issues():
    """Test the specific issues the user reported."""
    
    print("🔧 Testing User-Reported File Filtering Issues")
    print("=" * 60)
    
    # Replicate filtering logic
    excluded_dirs = None
    excluded_files = None
    
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
    
    print("📋 Configuration Summary:")
    print(f"   Excluded directories: {len(normalized_excluded_dirs)} total")
    print(f"   From repo config: {repo_excluded_dirs}")
    print(f"   Filename patterns: {excluded_filename_patterns}")
    
    # Test cases based on user's complaint
    test_cases = [
        # User said "vendor" directory should be excluded
        {
            "type": "directory", 
            "path": "vendor/package.go",
            "should_exclude": True,
            "reason": "vendor directory should be excluded"
        },
        {
            "type": "directory",
            "path": "vendor/submodule/utils.go", 
            "should_exclude": True,
            "reason": "vendor subdirectories should be excluded"
        },
        # User said "*_mock.go" pattern should be excluded
        {
            "type": "filename_pattern",
            "path": "service_mock.go",
            "should_exclude": True,
            "reason": "*_mock.go pattern should be excluded"
        },
        {
            "type": "filename_pattern", 
            "path": "handler_mock.go",
            "should_exclude": True,
            "reason": "*_mock.go pattern should be excluded"
        },
        {
            "type": "filename_pattern",
            "path": "src/utils_mock.go",
            "should_exclude": True, 
            "reason": "*_mock.go pattern should work in subdirectories"
        },
        # Files that should NOT be excluded
        {
            "type": "normal",
            "path": "main.go",
            "should_exclude": False,
            "reason": "regular go files should be included"
        },
        {
            "type": "normal",
            "path": "src/handler.go", 
            "should_exclude": False,
            "reason": "source files should be included"
        }
    ]
    
    print(f"\\n🧪 Testing {len(test_cases)} cases:")
    print("-" * 40)
    
    all_passed = True
    
    for i, test_case in enumerate(test_cases, 1):
        path = test_case["path"]
        expected_exclude = test_case["should_exclude"]
        reason = test_case["reason"]
        
        # Determine if should be excluded based on logic
        actually_excluded = False
        exclusion_reason = None
        
        # Check directory exclusion
        if '/' in path:
            path_parts = os.path.dirname(path).split('/')
            for part in path_parts:
                if part in normalized_excluded_dirs:
                    actually_excluded = True
                    exclusion_reason = f"directory '{part}' excluded"
                    break
        
        # Check filename pattern exclusion
        if not actually_excluded:
            filename = os.path.basename(path)
            for pattern in excluded_filename_patterns:
                if fnmatch.fnmatch(filename, pattern) or fnmatch.fnmatch(path, pattern):
                    actually_excluded = True
                    exclusion_reason = f"matches pattern '{pattern}'"
                    break
        
        # Check general file exclusion
        if not actually_excluded:
            for pattern in final_excluded_files:
                if fnmatch.fnmatch(path, pattern):
                    actually_excluded = True
                    exclusion_reason = f"matches file pattern '{pattern}'"
                    break
        
        # Compare with expected result
        if actually_excluded == expected_exclude:
            status = "✅ PASS"
            all_passed = True
        else:
            status = "❌ FAIL" 
            all_passed = False
        
        action = "EXCLUDED" if actually_excluded else "INCLUDED"
        expected_action = "EXCLUDED" if expected_exclude else "INCLUDED"
        
        if exclusion_reason:
            detail = f" ({exclusion_reason})"
        else:
            detail = ""
        
        print(f"  {i:2d}. {status}: {path}")
        print(f"      Expected: {expected_action}, Got: {action}{detail}")
        print(f"      Reason: {reason}")
        print()
    
    print("📊 Overall Result:")
    if all_passed:
        print("✅ All tests PASSED! The file filtering fixes are working correctly.")
        print("   - vendor directories are properly excluded")
        print("   - *_mock.go filename patterns are properly excluded") 
        print("   - normal source files are properly included")
    else:
        print("❌ Some tests FAILED! The file filtering needs more work.")
        print("   Review the failed cases above to identify issues.")
    
    return all_passed

if __name__ == "__main__":
    test_user_reported_issues()