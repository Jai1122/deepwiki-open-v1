#!/usr/bin/env python3
"""
Cache clearing utility for DeepWiki.
Use this when you encounter dimension mismatch errors after changing embedding models.
"""

import os
import sys
import shutil
from pathlib import Path

# Add the api directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from adalflow.utils import get_adalflow_default_root_path
except ImportError:
    print("Error: adalflow not found. Make sure you're in the correct environment.")
    sys.exit(1)

def clear_embedding_cache():
    """Clear all cached embeddings and databases."""
    print("🧹 DeepWiki Cache Cleaner")
    print("=" * 40)
    
    # Get the default root path where caches are stored
    root_path = get_adalflow_default_root_path()
    databases_dir = Path(root_path) / "databases"
    
    print(f"📁 Cache directory: {databases_dir}")
    
    if not databases_dir.exists():
        print("✅ No cache directory found - nothing to clear.")
        return
    
    # List all cache files
    cache_files = list(databases_dir.glob("*.pkl"))
    
    if not cache_files:
        print("✅ No cache files found - nothing to clear.")
        return
    
    print(f"📋 Found {len(cache_files)} cache files:")
    for i, cache_file in enumerate(cache_files, 1):
        file_size = cache_file.stat().st_size
        size_mb = file_size / (1024 * 1024)
        print(f"   {i}. {cache_file.name} ({size_mb:.1f} MB)")
    
    # Ask for confirmation
    response = input(f"\n❓ Clear all {len(cache_files)} cache files? [y/N]: ").strip().lower()
    
    if response in ['y', 'yes']:
        cleared_count = 0
        total_size = 0
        
        for cache_file in cache_files:
            try:
                file_size = cache_file.stat().st_size
                cache_file.unlink()
                cleared_count += 1
                total_size += file_size
                print(f"   ✅ Cleared: {cache_file.name}")
            except Exception as e:
                print(f"   ❌ Failed to clear {cache_file.name}: {e}")
        
        total_size_mb = total_size / (1024 * 1024)
        print(f"\n🎉 Successfully cleared {cleared_count} cache files ({total_size_mb:.1f} MB freed)")
        print("\n💡 Next steps:")
        print("   1. Restart your DeepWiki application")
        print("   2. Regenerate wikis - they will use the new embedding model")
        print("   3. First generation will be slower as it rebuilds the cache")
        
    else:
        print("❌ Cache clearing cancelled.")

def clear_specific_repo_cache(repo_name: str):
    """Clear cache for a specific repository."""
    root_path = get_adalflow_default_root_path()
    databases_dir = Path(root_path) / "databases"
    cache_file = databases_dir / f"{repo_name}.pkl"
    
    if cache_file.exists():
        try:
            file_size = cache_file.stat().st_size
            cache_file.unlink()
            size_mb = file_size / (1024 * 1024)
            print(f"✅ Cleared cache for '{repo_name}' ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"❌ Failed to clear cache for '{repo_name}': {e}")
    else:
        print(f"ℹ️  No cache found for repository '{repo_name}'")

def main():
    if len(sys.argv) > 1:
        repo_name = sys.argv[1]
        print(f"🎯 Clearing cache for specific repository: {repo_name}")
        clear_specific_repo_cache(repo_name)
    else:
        clear_embedding_cache()

if __name__ == "__main__":
    main()