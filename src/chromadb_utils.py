#!/usr/bin/env python3
"""
ChromaDB Utilities - Simple and reliable database management

This module provides a simple, reliable way to reset ChromaDB by
deleting the entire directory, which is the safest approach.
"""

import os
import shutil
import chromadb
from chromadb.config import Settings
from typing import Optional
import time


def safe_reset_chromadb(chroma_path: str = "./chroma_db") -> bool:
    """Safely reset ChromaDB by deleting the entire directory.
    
    This is the most reliable way to reset ChromaDB and avoid corruption issues.
    
    Args:
        chroma_path: Path to the ChromaDB directory
        
    Returns:
        True if reset was successful, False otherwise
    """
    try:
        if os.path.exists(chroma_path):
            # Add a small delay to ensure any file handles are released
            time.sleep(0.1)
            shutil.rmtree(chroma_path)
            print(f"Successfully deleted ChromaDB directory: {chroma_path}")
        else:
            print(f"ChromaDB directory does not exist: {chroma_path}")
        
        return True
        
    except Exception as e:
        print(f"Error deleting ChromaDB directory: {e}")
        return False


def create_fresh_client(chroma_path: str = "./chroma_db") -> chromadb.PersistentClient:
    """Create a fresh ChromaDB client.
    
    Args:
        chroma_path: Path to the ChromaDB directory
        
    Returns:
        New ChromaDB PersistentClient instance
    """
    return chromadb.PersistentClient(path=chroma_path)


def get_database_size(chroma_path: str = "./chroma_db") -> int:
    """Get the current size of the ChromaDB directory.
    
    Args:
        chroma_path: Path to the ChromaDB directory
        
    Returns:
        Size in bytes, or 0 if directory doesn't exist
    """
    if not os.path.exists(chroma_path):
        return 0
    
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(chroma_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
    except Exception as e:
        print(f"Error calculating directory size: {e}")
    
    return total_size


def reset_and_recreate(chroma_path: str = "./chroma_db") -> tuple[bool, chromadb.PersistentClient]:
    """Reset ChromaDB and create a new client.
    
    This is the recommended way to reset ChromaDB completely.
    
    Args:
        chroma_path: Path to the ChromaDB directory
        
    Returns:
        Tuple of (success: bool, new_client: PersistentClient or None)
    """
    # Reset by deletion
    reset_success = safe_reset_chromadb(chroma_path)
    
    if not reset_success:
        return False, None
    
    try:
        # Create new client
        new_client = create_fresh_client(chroma_path)
        return True, new_client
        
    except Exception as e:
        print(f"Error creating new client: {e}")
        return False, None


class ChromaDBManager:
    """A simple manager class for ChromaDB operations."""
    
    def __init__(self, chroma_path: str = "./chroma_db"):
        self.chroma_path = chroma_path
        self.client = None
    
    def initialize(self) -> bool:
        """Initialize or reinitialize the ChromaDB client."""
        try:
            self.client = create_fresh_client(self.chroma_path)
            return True
        except Exception as e:
            print(f"Error initializing ChromaDB: {e}")
            return False
    
    def reset(self) -> bool:
        """Reset the database by deleting the directory and reinitializing."""
        # Close current client
        self.client = None
        
        # Reset database
        reset_success = safe_reset_chromadb(self.chroma_path)
        
        if reset_success:
            # Reinitialize
            return self.initialize()
        
        return False
    
    def get_client(self) -> Optional[chromadb.PersistentClient]:
        """Get the current client, initializing if necessary."""
        if self.client is None:
            self.initialize()
        return self.client
    
    def get_size(self) -> int:
        """Get the current database size in bytes."""
        return get_database_size(self.chroma_path)


if __name__ == "__main__":
    # Example usage
    print("ChromaDB Utils - Simple Reset Example")
    
    # Method 1: Direct functions
    print("\n=== Method 1: Direct Functions ===")
    
    # Create initial data
    client = create_fresh_client()
    collection = client.get_or_create_collection(name="test_collection")
    collection.add(documents=["test doc 1", "test doc 2"], ids=["1", "2"])
    
    print(f"Initial size: {get_database_size():,} bytes")
    print(f"Collection count: {collection.count()}")
    
    # Reset
    client = None  # Release client reference
    success, new_client = reset_and_recreate()
    
    if success:
        print(f"Reset successful. New size: {get_database_size():,} bytes")
        new_collection = new_client.get_or_create_collection(name="test_collection")
        print(f"New collection count: {new_collection.count()}")
    
    # Method 2: Manager class
    print("\n=== Method 2: Manager Class ===")
    
    manager = ChromaDBManager()
    manager.initialize()
    
    client = manager.get_client()
    collection = client.get_or_create_collection(name="managed_collection")
    collection.add(documents=["managed doc 1"], ids=["m1"])
    
    print(f"Before reset: {manager.get_size():,} bytes, count: {collection.count()}")
    
    # Reset using manager
    reset_success = manager.reset()
    
    if reset_success:
        client = manager.get_client()
        collection = client.get_or_create_collection(name="managed_collection")
        print(f"After reset: {manager.get_size():,} bytes, count: {collection.count()}")