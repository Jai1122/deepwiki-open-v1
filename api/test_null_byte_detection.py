#!/usr/bin/env python3
"""
Test null byte detection and binary file filtering.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add the api directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import is_text_file, get_local_file_content

def create_test_files_with_null_bytes():
    """Create test files with various types of content including null bytes."""
    temp_dir = tempfile.mkdtemp(prefix="null_byte_test_")
    
    test_files = {
        # Normal text files
        "normal.txt": "This is a normal text file with no null bytes.",
        "code.go": "package main\nfunc main() {\n    fmt.Println(\"Hello\")\n}",
        
        # Files with null bytes (binary)
        "binary_with_nulls.dat": "Some text\x00with null\x00bytes",
        "git_object": "\x00blob 25\x00Hello world content",
        "compiled.o": "ELF\x7f\x00\x00\x00\x01\x02\x03binary content",
        
        # Edge cases
        "mostly_text_few_nulls.txt": "Normal text\x00but has one null byte",
        "control_chars.txt": "Text with\x01\x02\x03control chars",
        "high_ascii.txt": "Text with high ASCII: \xc0\xc1\xc2\xc3",
        
        # Empty file
        "empty.txt": "",
    }
    
    for filename, content in test_files.items():
        file_path = Path(temp_dir) / filename
        if isinstance(content, str):
            file_path.write_bytes(content.encode('latin-1'))
        else:
            file_path.write_bytes(content)
    
    return temp_dir

def test_null_byte_detection():
    """Test that null bytes are properly detected and files rejected."""
    
    print("🧪 Testing Null Byte Detection")
    print("=" * 50)
    
    test_dir = create_test_files_with_null_bytes()
    
    try:
        test_results = []
        
        for filename in os.listdir(test_dir):
            file_path = os.path.join(test_dir, filename)
            
            print(f"\\n📄 Testing: {filename}")
            
            # Test is_text_file detection
            is_text = is_text_file(file_path)
            print(f"   is_text_file: {is_text}")
            
            # Test file content reading
            content = get_local_file_content(file_path)
            has_content = len(content) > 0
            has_null_in_content = '\\x00' in content or '\\0' in content
            
            print(f"   Content length: {len(content)}")
            print(f"   Has null bytes in content: {has_null_in_content}")
            
            if content and len(content) > 0:
                print(f"   Content preview: {repr(content[:50])}{'...' if len(content) > 50 else ''}")
            
            # Expected behavior analysis
            with open(file_path, 'rb') as f:\n                raw_content = f.read()\n            raw_has_nulls = b'\\x00' in raw_content\n            \n            print(f"   Raw file has null bytes: {raw_has_nulls}")
            \n            # Determine if this test case passed\n            if raw_has_nulls:\n                # Files with null bytes should be rejected\n                passed = not is_text or not has_content or has_null_in_content\n                expected = \"REJECT (has null bytes)\"\n            else:\n                # Files without null bytes should generally be accepted (unless other issues)\n                passed = is_text and has_content and not has_null_in_content\n                expected = \"ACCEPT (no null bytes)\"\n            \n            actual = \"REJECTED\" if not is_text or not has_content else \"ACCEPTED\"\n            status = \"✅ PASS\" if passed else \"❌ FAIL\"\n            \n            print(f\"   Expected: {expected}\")\n            print(f\"   Actual: {actual}\")\n            print(f\"   Result: {status}\")\n            \n            test_results.append({\n                'filename': filename,\n                'passed': passed,\n                'raw_has_nulls': raw_has_nulls,\n                'is_text': is_text,\n                'has_content': has_content,\n                'content_has_nulls': has_null_in_content\n            })\n        \n        # Summary\n        passed_count = sum(1 for r in test_results if r['passed'])\n        total_count = len(test_results)\n        \n        print(f\"\\n📊 Summary:\")\n        print(f\"   Tests passed: {passed_count}/{total_count}\")\n        \n        if passed_count == total_count:\n            print(f\"\\n🎉 ALL TESTS PASSED! Null byte detection is working correctly.\")\n        else:\n            print(f\"\\n⚠️  SOME TESTS FAILED. Issues may remain:\")\n            for result in test_results:\n                if not result['passed']:\n                    print(f\"   - {result['filename']}: is_text={result['is_text']}, has_content={result['has_content']}, nulls={result['content_has_nulls']}\")\n        \n        return passed_count == total_count\n        \n    finally:\n        shutil.rmtree(test_dir)\n\ndef test_specific_patterns():\n    \"\"\"Test specific patterns that might be causing issues.\"\"\"\n    \n    print(f\"\\n🔍 Testing Specific Problematic Patterns\")\n    print(\"-\" * 45)\n    \n    # Test patterns that might be slipping through\n    problematic_patterns = [\n        (\"git_object_1\", b\"\\x78\\x9c\\x4b\\xca\\xc9\\x4f\\x52\\x30\\x34\\x35\\x07\\x00\"),  # Git object\n        (\"git_object_2\", b\"tree 40\\x00main.go\\x00\\x81\\xa4\"),  # Git tree\n        (\"binary_start\", b\"\\x00\\x00\\x00\\x20This looks like text but starts with nulls\"),\n        (\"binary_end\", b\"This looks like text but ends with nulls\\x00\\x00\\x00\"),\n        (\"embedded_nulls\", b\"Normal text\\x00embedded\\x00nulls\\x00in between\"),\n    ]\n    \n    temp_dir = tempfile.mkdtemp(prefix=\"pattern_test_\")\n    \n    try:\n        all_passed = True\n        \n        for name, content in problematic_patterns:\n            file_path = os.path.join(temp_dir, name)\n            with open(file_path, 'wb') as f:\n                f.write(content)\n            \n            is_text = is_text_file(file_path)\n            file_content = get_local_file_content(file_path)\n            \n            # These should all be rejected due to null bytes\n            should_reject = True\n            actually_rejected = not is_text or not file_content\n            \n            status = \"✅ PASS\" if actually_rejected == should_reject else \"❌ FAIL\"\n            result = \"REJECTED\" if actually_rejected else \"ACCEPTED\"\n            \n            print(f\"   {status}: {name} -> {result}\")\n            print(f\"      Content: {repr(content[:30])}...\")\n            \n            if actually_rejected != should_reject:\n                all_passed = False\n        \n        return all_passed\n        \n    finally:\n        shutil.rmtree(temp_dir)\n\nif __name__ == \"__main__\":\n    print(\"🛡️  Testing Null Byte Detection and Binary File Filtering\")\n    print(\"=\" * 60)\n    \n    test1 = test_null_byte_detection()\n    test2 = test_specific_patterns()\n    \n    print(f\"\\n🏁 Final Results:\")\n    print(f\"   General null byte detection: {'✅ PASS' if test1 else '❌ FAIL'}\")\n    print(f\"   Specific pattern detection: {'✅ PASS' if test2 else '❌ FAIL'}\")\n    \n    if test1 and test2:\n        print(f\"\\n🎉 SUCCESS: Null byte detection is working properly!\")\n        print(f\"   No more \\\\x00 characters should appear in embedding input.\")\n    else:\n        print(f\"\\n⚠️  ISSUES REMAIN: Some null bytes may still get through.\")\n        print(f\"   Check the failing test cases above for more details.\")