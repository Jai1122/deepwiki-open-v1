#!/usr/bin/env python3
"""
Quick diagnostic for wiki generation issues.
This script helps identify common problems with repository processing and RAG setup.
"""

import os
import sys
import logging
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag import RAG
from adalflow.utils import get_adalflow_default_root_path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def diagnose_wiki_issues(repo_path: str):
    """Diagnose common wiki generation issues."""
    
    print("🔍 Wiki Generation Diagnostic")
    print("=" * 50)
    
    # Check if repo path exists
    if not os.path.isdir(repo_path):
        print(f"❌ Repository path does not exist: {repo_path}")
        return False
    
    print(f"✅ Repository path exists: {repo_path}")
    
    # Check cache directory
    repo_name = os.path.basename(repo_path.rstrip('/'))
    cache_dir = os.path.join(get_adalflow_default_root_path(), "databases")
    cache_file = os.path.join(cache_dir, f"{repo_name}.pkl")
    
    print(f"\n📁 Cache Information:")
    print(f"   Cache directory: {cache_dir}")
    print(f"   Cache file: {cache_file}")
    print(f"   Cache exists: {os.path.exists(cache_file)}")
    
    if os.path.exists(cache_file):
        cache_size = os.path.getsize(cache_file)
        print(f"   Cache size: {cache_size} bytes")
        
        if cache_size < 1000:
            print("   ⚠️  Cache is very small - may be corrupted")
        elif cache_size > 100000:
            print("   ✅ Cache has substantial content")
        else:
            print("   📊 Cache has moderate content")
    
    # Test RAG initialization
    print(f"\n🤖 Testing RAG Initialization:")
    try:
        rag_instance = RAG(provider="google", model="gemini-2.0-flash") 
        print("   ✅ RAG instance created successfully")
        
        # Test retriever preparation
        print(f"\n⚙️  Testing Retriever Preparation:")
        rag_instance.prepare_retriever(repo_path, "local", None)
        
        # Check retriever state
        if hasattr(rag_instance, 'retriever') and rag_instance.retriever is not None:
            print("   ✅ Retriever initialized successfully")
        else:
            print("   ❌ Retriever failed to initialize")
            return False
            
        # Check documents
        if hasattr(rag_instance, 'transformed_docs') and rag_instance.transformed_docs:
            print(f"   ✅ Found {len(rag_instance.transformed_docs)} transformed documents")
            
            # Sample document validation
            valid_embeddings = 0
            for i, doc in enumerate(rag_instance.transformed_docs[:5]):
                if hasattr(doc, 'vector') and doc.vector:
                    valid_embeddings += 1
                    print(f"     Doc {i+1}: {len(doc.vector)}D embedding from {doc.metadata.get('source', 'unknown')}")
                else:
                    print(f"     Doc {i+1}: ❌ No valid embedding")
            
            print(f"   📊 {valid_embeddings}/{min(5, len(rag_instance.transformed_docs))} sample docs have valid embeddings")
            
        else:
            print("   ❌ No transformed documents found")
            return False
            
        # Test a simple retrieval
        print(f"\n🔍 Testing Document Retrieval:")
        try:
            result = rag_instance.call("test query", language="en")
            if result and isinstance(result, tuple) and len(result) >= 2:
                docs, _ = result
                if docs and isinstance(docs, list):
                    print(f"   ✅ Retrieved {len(docs)} documents successfully")
                    for i, doc in enumerate(docs[:3]):
                        source = doc.metadata.get('source', 'unknown') if hasattr(doc, 'metadata') else 'unknown'
                        content_preview = doc.text[:50].replace('\n', ' ') if hasattr(doc, 'text') else 'No content'
                        print(f"     Doc {i+1}: {source} - {content_preview}...")
                else:
                    print("   ❌ Retrieval returned invalid document format")
                    return False
            else:
                print("   ❌ Retrieval returned invalid result format")
                return False
        except Exception as e:
            print(f"   ❌ Retrieval test failed: {e}")
            return False
            
        print(f"\n🎉 All diagnostics passed! Wiki generation should work.")
        return True
        
    except Exception as e:
        print(f"   ❌ RAG initialization failed: {e}")
        logger.error("RAG initialization error", exc_info=True)
        return False

def suggest_fixes():
    """Suggest common fixes for wiki generation issues."""
    print(f"\n💡 Common Fixes:")
    print(f"   1. Clear cache: rm -rf ~/.adalflow/databases/")
    print(f"   2. Check embedding model environment variables:")
    print(f"      - EMBEDDING_MODEL_NAME")
    print(f"      - EMBEDDING_DIMENSIONS") 
    print(f"      - OPENAI_API_BASE_URL")
    print(f"   3. Regenerate repository embeddings")
    print(f"   4. Check repository contains actual source files (not just README)")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python diagnose_wiki_generation.py <repo_path>")
        print("Example: python diagnose_wiki_generation.py /path/to/your/repository")
        sys.exit(1)
    
    repo_path = sys.argv[1]
    success = diagnose_wiki_issues(repo_path)
    
    if not success:
        suggest_fixes()
        sys.exit(1)
    else:
        print(f"\n✨ Wiki generation should now work with this repository!")