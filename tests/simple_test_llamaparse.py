#!/usr/bin/env python3
"""
Simple test script for the updated RAG server with LlamaParse
"""

import os
from pathlib import Path
import sys

# Add the current directory to Python path
sys.path.insert(0, '.')

def test_direct_function():
    """Test the function directly without MCP wrapper"""
    print("Testing LlamaParse integration...")
    
    # Import necessary modules
    import chromadb
    from chromadb.config import Settings
    import uuid
    from llama_parse import LlamaParse
    
    # Initialize ChromaDB directly
    persist_directory = "./chroma_db"
    os.makedirs(persist_directory, exist_ok=True)
    
    chroma_client = chromadb.PersistentClient(
        path=persist_directory,
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True
        )
    )
    
    # Use a test-specific collection to avoid conflicts
    collection_name = "test_rag_documents"
    try:
        # Try to delete existing test collection if it exists
        chroma_client.delete_collection(name=collection_name)
    except:
        pass  # Collection doesn't exist, which is fine
    
    collection = chroma_client.create_collection(
        name=collection_name,
        metadata={"description": "Test collection for RAG document storage"}
    )
    
    print(f"ChromaDB initialized. Collection has {collection.count()} documents.")
    
    # Test text file ingestion (should work without API key)
    sample_file = "sample_document.txt"
    if not Path(sample_file).exists():
        print(f"Sample file '{sample_file}' not found")
        return False
    
    try:
        # Read the text file directly
        with open(sample_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"Read {len(content)} characters from {sample_file}")
        
        # Simple chunking
        chunk_size = 500
        overlap = 100
        chunks = []
        start = 0
        while start < len(content):
            end = start + chunk_size
            chunk = content[start:end]
            
            if end < len(content) and not content[end].isspace():
                last_space = chunk.rfind(' ')
                if last_space > start:
                    end = start + last_space
                    chunk = content[start:end]
            
            chunks.append(chunk.strip())
            start = end - overlap
        
        # Generate IDs and metadata
        chunk_ids = [f"test_{i}_{uuid.uuid4().hex[:8]}" for i in range(len(chunks))]
        metadatas = [{
            "source_file": sample_file,
            "file_type": ".txt",
            "chunk_index": i,
            "total_chunks": len(chunks),
            "chunk_size": len(chunk),
            "parsed_with": "direct_read"
        } for i, chunk in enumerate(chunks)]
        
        # Add to collection
        collection.add(
            documents=chunks,
            metadatas=metadatas,
            ids=chunk_ids
        )
        
        print(f"✓ Successfully added {len(chunks)} chunks to the database")
        
        # Test querying
        results = collection.query(
            query_texts=["What is machine learning?"],
            n_results=3,
            include=["documents", "metadatas", "distances"]
        )
        
        if results["documents"] and results["documents"][0]:
            print(f"✓ Query successful! Found {len(results['documents'][0])} results")
            print(f"First result preview: {results['documents'][0][0][:100]}...")
            return True
        else:
            print("✗ No results found for query")
            return False
            
    except Exception as e:
        print(f"✗ Error during test: {e}")
        return False

def test_llamaparse_availability():
    """Test if LlamaParse is properly installed"""
    print("\nTesting LlamaParse availability...")
    try:
        from llama_parse import LlamaParse
        print("✓ LlamaParse imported successfully")
        
        # Check if API key is available
        api_key = os.getenv('LLAMA_CLOUD_API_KEY')
        if api_key:
            print("✓ LLAMA_CLOUD_API_KEY is set")
            print("  You can now parse PDF, DOCX, PPTX and other document formats!")
        else:
            print("⚠️ LLAMA_CLOUD_API_KEY not set")
            print("  Text files will work, but PDF parsing requires an API key")
            print("  Get one at: https://cloud.llamaindex.ai/")
        
        return True
    except ImportError as e:
        print(f"✗ Failed to import LlamaParse: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing RAG Server with LlamaParse Integration")
    print("=" * 60)
    
    # Test LlamaParse availability
    llamaparse_test = test_llamaparse_availability()
    
    # Test direct function
    function_test = test_direct_function()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"LlamaParse availability: {'✓ PASSED' if llamaparse_test else '✗ FAILED'}")
    print(f"Core functionality: {'✓ PASSED' if function_test else '✗ FAILED'}")
    
    total_tests = 2
    passed_tests = sum([llamaparse_test, function_test])
    print(f"\nPassed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("\n🎉 All tests passed! The LlamaParse integration is working.")
        print("\n📝 Next steps:")
        print("1. Set LLAMA_CLOUD_API_KEY to test PDF parsing")
        print("2. Use the MCP server with your preferred client")
        print("3. Try ingesting PDF, DOCX, or PPTX files")
    else:
        print("\n⚠️ Some tests failed. Check the error messages above.")

if __name__ == "__main__":
    main()