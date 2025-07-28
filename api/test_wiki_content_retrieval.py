#!/usr/bin/env python3
"""
Test script to verify that wiki generation is retrieving actual source code content
rather than just README files.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag import RAG
from data_pipeline import DatabaseManager
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_wiki_content_retrieval(repo_path: str):
    """Test what content is actually retrieved for wiki generation."""
    
    print("🧪 Testing Wiki Content Retrieval")
    print("=" * 60)
    
    # Initialize RAG system like the websocket handler does
    rag_instance = RAG(provider="google", model="gemini-2.0-flash")
    
    try:
        # Prepare retriever (simulate repository processing)
        print(f"📁 Processing repository: {repo_path}")
        rag_instance.prepare_retriever(repo_path, "local", None)
        
        # Test the same queries used in websocket_wiki.py
        comprehensive_queries = [
            "Development Context",  # Original user query example
            "main application code functions classes implementation",
            "API endpoints routes handlers controllers",
            "database models data structures schemas",
            "configuration settings environment setup",
            "utility functions helpers common code",
        ]
        
        print(f"\n📊 Testing retrieval with comprehensive queries:")
        
        all_retrieved_docs = []
        seen_sources = set()
        
        for i, query in enumerate(comprehensive_queries, 1):
            print(f"\n🔍 Query {i}: '{query}'")
            
            try:
                docs, _ = rag_instance.call(query, language="en")
                print(f"   Retrieved: {len(docs)} documents")
                
                for doc in docs:
                    source = doc.metadata.get('source', 'unknown') if hasattr(doc, 'metadata') else 'unknown'
                    if source not in seen_sources:
                        all_retrieved_docs.append(doc)
                        seen_sources.add(source)
                        content_preview = doc.text[:100].replace('\n', ' ') if hasattr(doc, 'text') else 'No text'
                        print(f"     📄 {source}: {content_preview}...")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        # Summary analysis
        print(f"\n📈 RETRIEVAL SUMMARY:")
        print(f"   Total unique documents: {len(all_retrieved_docs)}")
        print(f"   Total unique source files: {len(seen_sources)}")
        
        # Analyze content types
        readme_files = [s for s in seen_sources if 'readme' in s.lower()]
        source_files = [s for s in seen_sources if any(ext in s.lower() for ext in ['.py', '.js', '.ts', '.go', '.java', '.cpp', '.c'])]
        config_files = [s for s in seen_sources if any(ext in s.lower() for ext in ['.json', '.yaml', '.yml', '.toml', '.ini'])]
        
        print(f"\n📋 CONTENT TYPE ANALYSIS:")
        print(f"   README files: {len(readme_files)} - {readme_files}")
        print(f"   Source code files: {len(source_files)} - {source_files[:5]}{'...' if len(source_files) > 5 else ''}")
        print(f"   Config files: {len(config_files)} - {config_files}")
        
        # Check content quality
        total_content_length = sum(len(doc.text) for doc in all_retrieved_docs if hasattr(doc, 'text'))
        print(f"\n📝 CONTENT QUALITY:")
        print(f"   Total content length: {total_content_length} characters")
        print(f"   Average per document: {total_content_length // len(all_retrieved_docs) if all_retrieved_docs else 0} characters")
        
        # Diagnose issues
        print(f"\n🔍 DIAGNOSTIC ASSESSMENT:")
        if len(readme_files) > len(source_files):
            print("   ⚠️  WARNING: More README files than source files retrieved")
            print("   💡 Issue: RAG system may be biased toward documentation over code")
        
        if total_content_length < 10000:
            print("   ⚠️  WARNING: Very low total content length")
            print("   💡 Issue: Not enough comprehensive content for quality wiki generation")
        
        if len(source_files) < 3:
            print("   ⚠️  WARNING: Very few actual source code files retrieved")
            print("   💡 Issue: Wiki will lack implementation details")
        
        if len(source_files) >= 5 and total_content_length > 15000:
            print("   ✅ SUCCESS: Good mix of source files with substantial content")
            print("   💡 Wiki generation should produce comprehensive documentation")
        
        # Test direct access fallback
        print(f"\n🔄 Testing direct access fallback:")
        if hasattr(rag_instance, 'transformed_docs') and rag_instance.transformed_docs:
            direct_docs = rag_instance.transformed_docs
            print(f"   📚 Direct access found: {len(direct_docs)} total documents in database")
            
            direct_sources = set()
            for doc in direct_docs[:10]:  # Sample first 10
                source = doc.metadata.get('source', 'unknown') if hasattr(doc, 'metadata') else 'unknown'
                direct_sources.add(source)
            
            print(f"   📁 Sample direct sources: {list(direct_sources)}")
        else:
            print("   ❌ No direct access to transformed documents available")
        
        return len(source_files) >= 3 and total_content_length > 10000
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_wiki_content_retrieval.py <repo_path>")
        print("Example: python test_wiki_content_retrieval.py /path/to/your/repository")
        sys.exit(1)
    
    repo_path = sys.argv[1]
    
    if not os.path.isdir(repo_path):
        print(f"❌ Repository path does not exist: {repo_path}")
        sys.exit(1)
    
    success = test_wiki_content_retrieval(repo_path)
    
    print(f"\n🏁 FINAL RESULT: {'✅ PASS' if success else '❌ FAIL'}")
    
    if success:
        print("✨ Wiki generation should now produce comprehensive, code-based documentation!")
    else:
        print("⚠️  Wiki generation may still produce generic documentation.")
        print("💡 Check the diagnostic messages above for specific issues to address.")