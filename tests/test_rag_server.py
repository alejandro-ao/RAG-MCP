#!/usr/bin/env python3
"""
Test script for the RAG MCP Server

This script demonstrates how to test the RAG server functionality:
1. Install dependencies
2. Test file ingestion
3. Test document querying
4. Test other server capabilities
"""

import subprocess
import sys
import os
import time
from pathlib import Path

def install_dependencies():
    """Install required dependencies"""
    print("Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install dependencies: {e}")
        return False

def test_server_import():
    """Test if the server can be imported without errors"""
    print("\nTesting server import...")
    try:
        # Test import
        import rag_server
        print("✓ Server imported successfully")
        
        # Test ChromaDB initialization
        if rag_server.collection is not None:
            print("✓ ChromaDB initialized successfully")
            print(f"✓ Collection has {rag_server.collection.count()} documents")
        else:
            print("✗ ChromaDB collection not initialized")
            return False
            
        return True
    except Exception as e:
        print(f"✗ Failed to import server: {e}")
        return False

def test_file_ingestion():
    """Test file ingestion functionality"""
    print("\nTesting file ingestion...")
    try:
        import rag_server
        
        # Test ingesting the sample document
        sample_file = "sample_document.txt"
        if not Path(sample_file).exists():
            print(f"✗ Sample file '{sample_file}' not found")
            return False
        
        # Call the function directly from the imported module
        from rag_server import ingest_file
        result = ingest_file(sample_file, chunk_size=500, overlap=100)
        print(f"Ingestion result: {result}")
        
        if "Successfully ingested" in result:
            print("✓ File ingestion successful")
            
            # Check collection count
            count = rag_server.collection.count()
            print(f"✓ Collection now has {count} documents")
            return True
        else:
            print("✗ File ingestion failed")
            return False
            
    except Exception as e:
        print(f"✗ Error during file ingestion: {e}")
        return False

def test_document_querying():
    """Test document querying functionality"""
    print("\nTesting document querying...")
    try:
        import rag_server
        
        # Import the function directly
        from rag_server import query_documents
        
        # Test various queries
        test_queries = [
            "What is machine learning?",
            "Types of machine learning",
            "supervised learning examples",
            "neural networks",
            "challenges in machine learning"
        ]
        
        for query in test_queries:
            print(f"\nQuerying: '{query}'")
            result = query_documents(query, n_results=3)
            
            if "No relevant documents found" in result:
                print("✗ No results found")
            elif "Error" in result:
                print(f"✗ Query error: {result}")
            else:
                print("✓ Query successful")
                # Print first 200 characters of result
                print(f"Preview: {result[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during querying: {e}")
        return False

def test_additional_tools():
    """Test additional server tools"""
    print("\nTesting additional tools...")
    try:
        import rag_server
        
        # Import functions directly
        from rag_server import list_ingested_files, get_rag_status
        
        # Test list_ingested_files
        print("\nTesting list_ingested_files...")
        files_result = list_ingested_files()
        print(f"Files result: {files_result}")
        
        # Test get_rag_status
        print("\nTesting get_rag_status...")
        status = get_rag_status()
        print(f"Status: {status}")
        
        if status.get("status") == "active":
            print("✓ RAG system is active")
        else:
            print("✗ RAG system status issue")
            
        return True
        
    except Exception as e:
        print(f"✗ Error testing additional tools: {e}")
        return False

def test_mcp_inspector():
    """Instructions for testing with MCP Inspector"""
    print("\n" + "="*60)
    print("MCP INSPECTOR TESTING")
    print("="*60)
    print("""
To test the MCP server with the built-in inspector:

1. Install FastMCP with CLI tools:
   pip install "fastmcp[cli]"

2. Run the server with inspector:
   python rag_server.py
   
   OR
   
   fastmcp dev rag_server.py

3. Open your browser to:
   http://127.0.0.1:6274

4. Test the following tools in the inspector:
   - ingest_file: Use 'sample_document.txt'
   - query_documents: Try queries like 'machine learning types'
   - list_ingested_files: See what files are stored
   - clear_database: Clear all documents (use with caution)

5. Test the resource:
   - rag://status: Get system status

6. Test the prompt:
   - rag_analysis_prompt: Generate analysis prompts
""")

def main():
    """Main test function"""
    print("RAG MCP Server Test Suite")
    print("="*40)
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    tests = [
        ("Installing Dependencies", install_dependencies),
        ("Testing Server Import", test_server_import),
        ("Testing File Ingestion", test_file_ingestion),
        ("Testing Document Querying", test_document_querying),
        ("Testing Additional Tools", test_additional_tools)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ Test '{test_name}' failed with exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = 0
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Your RAG MCP Server is ready to use.")
        test_mcp_inspector()
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)