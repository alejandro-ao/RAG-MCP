#!/usr/bin/env python3
"""
Test script for auto-ingestion functionality
"""

import os
import sys
import logging
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_auto_ingest():
    """Test that the server auto-ingests files from data directory"""
    print("🧪 Testing auto-ingestion functionality...")
    
    try:
        # Import the server (this will trigger initialization and auto-ingestion)
        print("📥 Importing server (this will trigger auto-ingestion)...")
        import rag_server
        
        # Check if collection has documents
        if rag_server.collection is None:
            print("❌ Collection is None - initialization failed")
            return False
            
        doc_count = rag_server.collection.count()
        print(f"📊 Collection has {doc_count} documents after auto-ingestion")
        
        if doc_count > 0:
            print("✅ Auto-ingestion successful!")
            
            # Test querying
            print("\n🔍 Testing document querying...")
            try:
                # Query the collection directly
                results = rag_server.collection.query(
                    query_texts=["attention mechanism"],
                    n_results=3
                )
                
                if results and results['documents'] and len(results['documents'][0]) > 0:
                    print(f"📝 Found {len(results['documents'][0])} relevant documents")
                    print(f"📄 First result preview: {results['documents'][0][0][:200]}...")
                    print("✅ Querying works!")
                    return True
                else:
                    print("❌ No results found")
                    return False
            except Exception as e:
                print(f"❌ Query failed: {e}")
                return False
        else:
            print("⚠️  No documents found - check if files exist in data directory")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_data_directory():
    """Check what files are in the data directory"""
    print("📁 Checking data directory contents...")
    # Import get_data_directory function from rag_server
    from rag_server import get_data_directory
    data_dir = get_data_directory()
    
    if not data_dir.exists():
        print("❌ Data directory doesn't exist")
        return False
        
    files = list(data_dir.glob("*"))
    files = [f for f in files if f.is_file()]
    
    print(f"📄 Found {len(files)} files in data directory:")
    for file in files:
        print(f"   - {file.name} ({file.stat().st_size} bytes)")
    
    return len(files) > 0

if __name__ == "__main__":
    print("🚀 Starting auto-ingestion test...\n")
    
    # Check data directory first
    has_files = check_data_directory()
    
    if not has_files:
        print("\n⚠️  No files in data directory. Auto-ingestion test may not be meaningful.")
    
    print("\n" + "="*50)
    
    # Run the test
    success = test_auto_ingest()
    
    print("\n" + "="*50)
    if success:
        print("🎉 All tests passed!")
    else:
        print("💥 Tests failed!")
    
    sys.exit(0 if success else 1)