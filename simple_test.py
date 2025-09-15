#!/usr/bin/env python3
"""
Simple test script for RAG server core functionality.
This bypasses the MCP decorators to test the underlying logic.
"""

import sys
import os
from pathlib import Path

def test_chromadb_setup():
    """Test ChromaDB initialization."""
    print("Testing ChromaDB setup...")
    try:
        import chromadb
        from chromadb.config import Settings
        
        # Initialize ChromaDB client
        client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Create or get collection
        try:
            collection = client.get_collection("rag_documents")
            print(f"✓ Found existing collection with {collection.count()} documents")
        except:
            collection = client.create_collection(
                name="rag_documents",
                metadata={"description": "Collection for RAG document storage"}
            )
            print("✓ Created new collection")
        
        return client, collection
    except Exception as e:
        print(f"✗ ChromaDB setup failed: {e}")
        return None, None

def test_file_ingestion(collection, file_path="sample_document.txt"):
    """Test file ingestion functionality."""
    print(f"\nTesting file ingestion with {file_path}...")
    try:
        # Check if file exists
        path = Path(file_path)
        if not path.exists():
            print(f"✗ File {file_path} does not exist")
            return False
        
        # Read file content
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"✓ Read file: {len(content)} characters")
        
        # Simple chunking
        chunk_size = 1000
        overlap = 200
        chunks = []
        
        for i in range(0, len(content), chunk_size - overlap):
            chunk = content[i:i + chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        
        print(f"✓ Created {len(chunks)} chunks")
        
        # Add to collection
        import uuid
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [{
            "file_path": str(path),
            "chunk_index": i,
            "total_chunks": len(chunks)
        } for i in range(len(chunks))]
        
        collection.add(
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"✓ Added {len(chunks)} chunks to collection")
        print(f"✓ Collection now has {collection.count()} total documents")
        return True
        
    except Exception as e:
        print(f"✗ File ingestion failed: {e}")
        return False

def test_querying(collection, query="What is machine learning?"):
    """Test document querying functionality."""
    print(f"\nTesting querying with: '{query}'")
    try:
        results = collection.query(
            query_texts=[query],
            n_results=3,
            include=["documents", "metadatas", "distances"]
        )
        
        if not results['documents'][0]:
            print("✗ No documents found")
            return False
        
        print(f"✓ Found {len(results['documents'][0])} relevant documents")
        
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            print(f"\nResult {i+1} (distance: {distance:.3f}):")
            print(f"File: {metadata.get('file_path', 'Unknown')}")
            print(f"Chunk: {metadata.get('chunk_index', 'Unknown')}/{metadata.get('total_chunks', 'Unknown')}")
            print(f"Content preview: {doc[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ Querying failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("RAG Server Core Functionality Test")
    print("=" * 60)
    
    # Test ChromaDB setup
    client, collection = test_chromadb_setup()
    if not client or not collection:
        print("\n✗ Cannot proceed without ChromaDB")
        return
    
    # Test file ingestion
    ingestion_success = test_file_ingestion(collection)
    
    # Test querying
    if ingestion_success:
        query_success = test_querying(collection)
    else:
        print("\n⚠️ Skipping query test due to ingestion failure")
        query_success = False
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"ChromaDB Setup: {'✓ PASSED' if client and collection else '✗ FAILED'}")
    print(f"File Ingestion: {'✓ PASSED' if ingestion_success else '✗ FAILED'}")
    print(f"Document Querying: {'✓ PASSED' if query_success else '✗ FAILED'}")
    
    total_passed = sum([bool(client and collection), ingestion_success, query_success])
    print(f"\nTotal: {total_passed}/3 tests passed")
    
    if total_passed == 3:
        print("\n🎉 All core functionality tests passed!")
        print("The RAG server should work correctly with MCP clients.")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()