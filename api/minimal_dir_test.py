#!/usr/bin/env python3
"""
Minimal test to reproduce the vendor directory filtering issue.
"""

import os
import tempfile
import shutil
from pathlib import Path

def test_os_walk_filtering():
    """Test os.walk filtering behavior with a simple example."""
    
    # Create test directory
    temp_dir = tempfile.mkdtemp(prefix="walk_test_")
    
    # Create structure
    structure = {
        "main.go": "main",
        "src/app.go": "src", 
        "vendor/pkg/mod.go": "vendor file",
        "lib/utils.go": "lib"
    }
    
    for file_path, content in structure.items():
        full_path = Path(temp_dir) / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)
    
    print(f"Created test structure in: {temp_dir}")
    
    # Show initial structure
    print(f"\n📁 Initial structure:")
    for root, dirs, files in os.walk(temp_dir):
        level = root.replace(temp_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            print(f"{subindent}{file}")
    
    # Test filtering
    print(f"\n🧪 Testing filtering:")
    excluded_dirs = ['vendor']
    
    processed_files = []
    
    for root, dirs, files in os.walk(temp_dir, topdown=True):
        current_dir_relative = os.path.relpath(root, temp_dir)
        normalized_current_dir = current_dir_relative.replace('\\\\', '/')
        
        if normalized_current_dir == '.':
            normalized_current_dir = ''
        
        print(f"\n📂 Processing: '{normalized_current_dir}'")
        print(f"   dirs: {dirs}")
        print(f"   files: {files}")
        
        # Check if current directory should be skipped
        should_skip = False
        if normalized_current_dir:
            path_components = normalized_current_dir.split('/')
            for component in path_components:
                if component in excluded_dirs:
                    should_skip = True
                    print(f"   ❌ SKIPPING due to component: {component}")
                    break
        
        if should_skip:
            dirs.clear()
            continue
        
        # Filter subdirectories
        original_dirs = dirs[:]
        dirs[:] = [d for d in dirs if d not in excluded_dirs]
        
        filtered = set(original_dirs) - set(dirs)
        if filtered:
            print(f"   🚫 Filtered subdirs: {filtered}")
        
        print(f"   ➡️  Will recurse into: {dirs}")
        
        # Process files
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, temp_dir)
            processed_files.append(relative_path)
            print(f"   📄 Processing: {relative_path}")
    
    print(f"\n📊 Final results:")
    print(f"   Total processed files: {len(processed_files)}")
    for f in processed_files:
        print(f"   - {f}")
    
    # Check for vendor files
    vendor_files = [f for f in processed_files if 'vendor' in f]
    if vendor_files:
        print(f"\n❌ PROBLEM: Found vendor files: {vendor_files}")
    else:
        print(f"\n✅ SUCCESS: No vendor files processed")
    
    # Cleanup
    shutil.rmtree(temp_dir)
    
    return len(vendor_files) == 0

if __name__ == "__main__":
    success = test_os_walk_filtering()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")