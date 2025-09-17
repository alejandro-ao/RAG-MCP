#!/usr/bin/env python3
"""
Test script to verify the environment variable database directory functionality.
"""

import os
import sys
from pathlib import Path

# Add current directory to path to import rag_server
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_server import get_database_directory

def test_env_variable():
    """Test that LLAMA_RAG_DB_DIR environment variable is respected"""
    print("🧪 Testing database directory environment variable functionality...\n")
    
    # Test 1: No environment variable (should use XDG standard)
    print("Test 1: No environment variable set (XDG standard)")
    if 'LLAMA_RAG_DB_DIR' in os.environ:
        del os.environ['LLAMA_RAG_DB_DIR']
    
    db_dir = get_database_directory()
    print(f"   Result: {db_dir}")
    print(f"   Resolved: {db_dir.resolve()}")
    
    # Check if it follows XDG standard
    home_dir = Path.home()
    xdg_data_home = os.getenv('XDG_DATA_HOME', home_dir / '.local' / 'share')
    expected_standard = Path(xdg_data_home) / 'rag-server'
    is_standard = db_dir.resolve() == expected_standard.resolve()
    print(f"   Follows XDG standard: {is_standard}\n")
    
    # Test 2: Set environment variable to a custom path
    print("Test 2: Environment variable set to custom path")
    custom_path = "/tmp/custom_rag_db"
    os.environ['LLAMA_RAG_DB_DIR'] = custom_path
    
    db_dir = get_database_directory()
    print(f"   Result: {db_dir}")
    print(f"   Expected: {Path(custom_path).resolve()}")
    print(f"   Match: {db_dir.resolve() == Path(custom_path).resolve()}\n")
    
    # Test 3: Set environment variable with tilde expansion
    print("Test 3: Environment variable with tilde expansion")
    home_path = "~/Documents/rag_database"
    os.environ['LLAMA_RAG_DB_DIR'] = home_path
    
    db_dir = get_database_directory()
    expected = Path(home_path).expanduser().resolve()
    print(f"   Result: {db_dir}")
    print(f"   Expected: {expected}")
    print(f"   Match: {db_dir == expected}\n")
    
    # Test 4: Relative path in environment variable
    print("Test 4: Environment variable with relative path")
    rel_path = "../test_database"
    os.environ['LLAMA_RAG_DB_DIR'] = rel_path
    
    db_dir = get_database_directory()
    expected = Path(rel_path).expanduser().resolve()
    print(f"   Result: {db_dir}")
    print(f"   Expected: {expected}")
    print(f"   Match: {db_dir == expected}\n")
    
    # Test 5: Test XDG_DATA_HOME override
    print("Test 5: XDG_DATA_HOME environment variable override")
    if 'LLAMA_RAG_DB_DIR' in os.environ:
        del os.environ['LLAMA_RAG_DB_DIR']
    
    custom_xdg = "/tmp/custom_xdg_data"
    os.environ['XDG_DATA_HOME'] = custom_xdg
    
    db_dir = get_database_directory()
    expected_xdg = Path(custom_xdg) / 'rag-server'
    print(f"   Result: {db_dir}")
    print(f"   Expected: {expected_xdg}")
    print(f"   Match: {db_dir.resolve() == expected_xdg.resolve()}\n")
    
    # Clean up
    if 'LLAMA_RAG_DB_DIR' in os.environ:
        del os.environ['LLAMA_RAG_DB_DIR']
    if 'XDG_DATA_HOME' in os.environ:
        del os.environ['XDG_DATA_HOME']
    
    print("✅ Database directory environment variable tests completed!")
    return True

def show_usage_examples():
    """Show usage examples for the database directory environment variable"""
    print("\n📖 Database Directory Usage Examples:")
    print("\n1. Set custom database directory for current session:")
    print("   export LLAMA_RAG_DB_DIR=/path/to/your/database")
    print("   python rag_server.py")
    
    print("\n2. Set for single command:")
    print("   LLAMA_RAG_DB_DIR=/path/to/your/database python rag_server.py")
    
    print("\n3. Use XDG standard directory (default):")
    print("   # No environment variable set")
    print("   # Will use ~/.local/share/rag-server")
    
    print("\n4. Use home directory:")
    print("   export LLAMA_RAG_DB_DIR=~/Documents/my_rag_database")
    
    print("\n5. Use absolute path:")
    print("   export LLAMA_RAG_DB_DIR=/Users/username/databases/rag")
    
    print("\n6. Override XDG data directory:")
    print("   export XDG_DATA_HOME=/custom/data/location")
    print("   # Will use /custom/data/location/rag-server")
    
    print("\n📁 Standard Locations by Platform:")
    print("   • Linux/macOS: ~/.local/share/rag-server")
    print("   • Windows: %LOCALAPPDATA%/rag-server (if implemented)")
    print("   • Fallback: ./chroma (current directory)")

def show_xdg_info():
    """Show information about XDG Base Directory specification"""
    print("\n📋 XDG Base Directory Specification:")
    print("   The XDG Base Directory Specification defines standard")
    print("   locations for user-specific data files on Unix-like systems.")
    print("\n   Data Directory: $XDG_DATA_HOME or ~/.local/share")
    print("   Benefits:")
    print("   • Follows system conventions")
    print("   • Keeps user data organized")
    print("   • Respects user preferences")
    print("   • Easy to backup/restore")

if __name__ == "__main__":
    print("🚀 Testing RAG Server Database Directory Configuration\n")
    print("=" * 70)
    
    success = test_env_variable()
    
    show_usage_examples()
    show_xdg_info()
    
    print("\n" + "=" * 70)
    if success:
        print("🎉 All tests passed!")
    else:
        print("💥 Tests failed!")
    
    sys.exit(0 if success else 1)