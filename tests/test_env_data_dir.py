#!/usr/bin/env python3
"""
Test script to verify the environment variable data directory functionality.
"""

import os
import sys
from pathlib import Path

# Add current directory to path to import rag_server
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_server import get_data_directory

def test_env_variable():
    """Test that LLAMA_RAG_DATA_DIR environment variable is respected"""
    print("🧪 Testing environment variable functionality...\n")
    
    # Test 1: No environment variable (should use workspace or fallback)
    print("Test 1: No environment variable set")
    if 'LLAMA_RAG_DATA_DIR' in os.environ:
        del os.environ['LLAMA_RAG_DATA_DIR']
    
    data_dir = get_data_directory()
    print(f"   Result: {data_dir}")
    print(f"   Resolved: {data_dir.resolve()}\n")
    
    # Test 2: Set environment variable to a custom path
    print("Test 2: Environment variable set to custom path")
    custom_path = "/tmp/custom_rag_data"
    os.environ['LLAMA_RAG_DATA_DIR'] = custom_path
    
    data_dir = get_data_directory()
    print(f"   Result: {data_dir}")
    print(f"   Expected: {Path(custom_path).resolve()}")
    print(f"   Match: {data_dir.resolve() == Path(custom_path).resolve()}\n")
    
    # Test 3: Set environment variable with tilde expansion
    print("Test 3: Environment variable with tilde expansion")
    home_path = "~/Documents/rag_data"
    os.environ['LLAMA_RAG_DATA_DIR'] = home_path
    
    data_dir = get_data_directory()
    expected = Path(home_path).expanduser().resolve()
    print(f"   Result: {data_dir}")
    print(f"   Expected: {expected}")
    print(f"   Match: {data_dir == expected}\n")
    
    # Test 4: Relative path in environment variable
    print("Test 4: Environment variable with relative path")
    rel_path = "../test_data"
    os.environ['LLAMA_RAG_DATA_DIR'] = rel_path
    
    data_dir = get_data_directory()
    expected = Path(rel_path).expanduser().resolve()
    print(f"   Result: {data_dir}")
    print(f"   Expected: {expected}")
    print(f"   Match: {data_dir == expected}\n")
    
    # Clean up
    if 'LLAMA_RAG_DATA_DIR' in os.environ:
        del os.environ['LLAMA_RAG_DATA_DIR']
    
    print("✅ Environment variable tests completed!")
    return True

def show_usage_examples():
    """Show usage examples for the environment variable"""
    print("\n📖 Usage Examples:")
    print("\n1. Set environment variable for current session:")
    print("   export LLAMA_RAG_DATA_DIR=/path/to/your/data")
    print("   python rag_server.py")
    
    print("\n2. Set for single command:")
    print("   LLAMA_RAG_DATA_DIR=/path/to/your/data python rag_server.py")
    
    print("\n3. Use workspace-relative data directory (default):")
    print("   # No environment variable set")
    print("   # Will use ./data in current working directory")
    
    print("\n4. Use home directory:")
    print("   export LLAMA_RAG_DATA_DIR=~/Documents/my_rag_data")
    
    print("\n5. Use absolute path:")
    print("   export LLAMA_RAG_DATA_DIR=/Users/username/projects/data")

if __name__ == "__main__":
    print("🚀 Testing RAG Server Data Directory Configuration\n")
    print("=" * 60)
    
    success = test_env_variable()
    
    show_usage_examples()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests passed!")
    else:
        print("💥 Tests failed!")
    
    sys.exit(0 if success else 1)