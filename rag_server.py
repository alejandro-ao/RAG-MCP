#!/usr/bin/env python3
"""
RAG MCP Server - Retrieval Augmented Generation using FastMCP and ChromaDB

This server provides tools for:
1. Ingesting files into a local vector database (ChromaDB)
2. Retrieving relevant information based on queries
"""

import os
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import uuid
from textwrap import dedent


# FastMCP imports
from fastmcp import FastMCP
from fastmcp.prompts.prompt import PromptMessage, TextContent

# ChromaDB imports
import chromadb
from chromadb.config import Settings
from chromadb_utils import safe_reset_chromadb

# LlamaParse and LlamaIndex imports
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastMCP server with RAG capabilities
mcp = FastMCP("RAG Server")

# Global ChromaDB client and collection
chroma_client = None
collection = None

def initialize_chromadb():
    """Initialize ChromaDB client and collection, then auto-ingest files from data directory"""
    global chroma_client, collection
    
    try:
        # Get database directory using flexible resolution
        persist_directory = get_database_directory()
        safe_reset_chromadb(chroma_path=str(persist_directory))
        
        os.makedirs(persist_directory, exist_ok=True)
        
        chroma_client = chromadb.PersistentClient(
            path=str(persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Create fresh collection for RAG documents
        collection = chroma_client.create_collection(
            name="rag_documents",
            metadata={"description": "Collection for RAG document storage"}
        )
        
        logger.info(f"ChromaDB initialized successfully. Vector database has {collection.count()} documents.")
        
        # Auto-ingest files from data directory
        auto_ingest_files()
        
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB: {e}")
        raise

def get_database_directory():
    """Get the ChromaDB persistent directory path with flexible resolution strategy.
    
    Priority order:
    1. LLAMA_RAG_DB_DIR environment variable (user-specified)
    2. ~/.local/share/rag-server (XDG Base Directory standard)
    3. ./chroma relative to current working directory (fallback)
    
    Returns:
        Path: Resolved database directory path
    """
    # 1. Check environment variable first
    env_db_dir = os.getenv('LLAMA_RAG_DB_DIR')
    if env_db_dir:
        db_path = Path(env_db_dir).expanduser().resolve()
        logger.info(f"Using database directory from LLAMA_RAG_DB_DIR: {db_path}")
        return db_path
    
    # 2. Use XDG Base Directory standard (~/.local/share/rag-server)
    try:
        home_dir = Path.home()
        xdg_data_home = os.getenv('XDG_DATA_HOME', home_dir / '.local' / 'share')
        standard_db = Path(xdg_data_home) / 'rag-server'
        logger.info(f"Using standard database directory: {standard_db}")
        return standard_db
    except Exception as e:
        logger.warning(f"Could not access standard database directory: {e}")
    
    # 3. Fallback to server-relative ./chroma
    fallback_db = Path('./chroma')
    logger.info(f"Using fallback database directory: {fallback_db.resolve()}")
    return fallback_db

def get_data_directory():
    """Get the data directory path with flexible resolution strategy.
    
    Priority order:
    1. LLAMA_RAG_DATA_DIR environment variable (user-specified)
    2. ./data in current working directory (workspace-relative)
    
    3. Raise error if no environment variable and no existing data directory
    
    Returns:
        Path: Resolved data directory path
        
    Raises:
        ValueError: If no environment variable is set and no data directory exists
    """
    # 1. Check environment variable first
    env_data_dir = os.getenv('LLAMA_RAG_DATA_DIR')
    if env_data_dir:
        data_path = Path(env_data_dir).expanduser().resolve()
        logger.info(f"Using data directory from LLAMA_RAG_DATA_DIR: {data_path}")
        return data_path
    
    # 2. Try workspace-relative ./data (current working directory)
    cwd_data = Path.cwd() / 'data'
    if cwd_data.exists():
        logger.info(f"Using workspace-relative data directory: {cwd_data}")
        return cwd_data
    
    # 3. No environment variable and no existing data directory - raise error
    error_msg = (
        "No data directory found. Please either:\n"
        "1. Set the LLAMA_RAG_DATA_DIR environment variable to specify a data directory, or\n"
        "2. Create a 'data' directory in the current working directory\n\n"
        "Examples:\n"
        "  export LLAMA_RAG_DATA_DIR=/path/to/your/documents\n"
        "  mkdir data  # Create data directory in current location"
    )
    logger.error(error_msg)
    raise ValueError(error_msg)

def auto_ingest_files():
    """Automatically ingest all files from the data directory"""
    global collection
    
    try:
        # Get data directory using flexible resolution
        data_path = get_data_directory()
        
        # Create data directory if it doesn't exist
        os.makedirs(data_path, exist_ok=True)
        
        # Check if data directory has any files
        files = list(data_path.glob("*"))
        files = [f for f in files if f.is_file()]
        
        if not files:
            logger.info("No files found in data directory. Skipping auto-ingestion.")
            return
        
        logger.info(f"Found {len(files)} files in data directory. Starting auto-ingestion...")
        
        # Initialize LlamaParse for PDF and other document types
        api_key = os.getenv("LLAMA_CLOUD_API_KEY")
        if not api_key:
            logger.warning("LLAMA_CLOUD_API_KEY not set. PDF parsing will be limited.")
            parser = None
        else:
            parser = LlamaParse(api_key=api_key, result_type="text")
        
        # Set up file extractors for different file types
        file_extractor = {}
        if parser:
            file_extractor = {
                ".pdf": parser,
                ".docx": parser,
                ".pptx": parser,
                ".doc": parser,
                ".ppt": parser
            }
        
        # Use SimpleDirectoryReader to load all documents
        documents = SimpleDirectoryReader(
            input_dir=str(data_path),
            file_extractor=file_extractor,
            recursive=True
        ).load_data()
        
        if not documents:
            logger.info("No documents loaded from data directory.")
            return
        
        logger.info(f"Loaded {len(documents)} documents from data directory.")
        
        # Process and add documents to ChromaDB
        for doc in documents:
            try:
                # Get document content and metadata
                content = doc.text
                metadata = doc.metadata or {}
                id = doc.id_
                
                # Add file name to metadata if available
                if hasattr(doc, 'metadata') and 'file_name' in doc.metadata:
                    file_name = doc.metadata['file_name']
                elif hasattr(doc, 'metadata') and 'file_path' in doc.metadata:
                    file_name = Path(doc.metadata['file_path']).name
                else:
                    file_name = "unknown"
                
                metadata.update({
                    "file_name": file_name,
                    "ingestion_method": "auto_ingest",
                    "chunk_size": len(content)
                })
                
                # Add to ChromaDB collection
                collection.add(
                    documents=[content],
                    metadatas=[metadata],
                    ids=[id]
                )
                
                logger.info(f"Successfully ingested: {file_name}")
                
            except Exception as e:
                logger.error(f"Failed to ingest document: {e}")
                continue
        
        final_count = collection.count()
        logger.info(f"Auto-ingestion completed. Collection now has {final_count} documents.")
        
    except ValueError as e:
        # Data directory configuration error - log and skip auto-ingestion
        logger.warning(f"Skipping auto-ingestion: {e}")
        return
    except Exception as e:
        logger.error(f"Failed during auto-ingestion: {e}")
        # Don't raise here as we want the server to continue even if auto-ingestion fails

# Initialize ChromaDB on server startup
initialize_chromadb()

@mcp.tool
def query_documents(query: str, n_results: int = 5, include_metadata: bool = True) -> str:
    """
    Query the vector database to retrieve relevant documents.
    
    Args:
        query: The search query
        n_results: Number of results to return (default: 5)
        include_metadata: Whether to include metadata in results (default: True)
    
    Returns:
        Formatted string with relevant documents and their metadata
    """
    global collection
    try:
        if not query.strip():
            return "Error: Query cannot be empty."
        
        # Validate n_results
        if n_results <= 0:
            n_results = 5
        elif n_results > 20:
            n_results = 20
        
        # Query the collection
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        if not results["documents"] or not results["documents"][0]:
            return "No relevant documents found for your query."
        
        # Format results
        formatted_results = []
        documents = results["documents"][0]
        metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(documents)
        distances = results["distances"][0] if results["distances"] else [0] * len(documents)
        
        for i, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
            result_text = f"\n--- Result {i+1} ---\n"
            result_text += f"Content: {doc}\n"
            
            if include_metadata and metadata:
                result_text += f"Source: {metadata.get('source_file', 'Unknown')}\n"
                result_text += f"Chunk: {metadata.get('chunk_index', 'Unknown')} of {metadata.get('total_chunks', 'Unknown')}\n"
                result_text += f"Similarity Score: {1 - distance:.3f}\n"
            
            formatted_results.append(result_text)
        
        response = f"Found {len(documents)} relevant documents for query: '{query}'\n"
        response += "\n".join(formatted_results)
        
        logger.info(f"Query '{query}' returned {len(documents)} results")
        return response
        
    except Exception as e:
        error_msg = f"Error querying documents: {str(e)}"
        logger.error(error_msg)
        return error_msg

@mcp.tool
def list_ingested_files() -> str:
    """
    List all files that have been ingested into the vector database.
    
    Returns:
        Formatted string with information about ingested files
    """
    global collection
    try:
        # Get all documents with metadata
        all_docs = collection.get(include=["metadatas"])
        
        if not all_docs["metadatas"]:
            return "No files have been ingested yet."
        
        # Group by file name and path
        file_info = {}
        for metadata in all_docs["metadatas"]:
            if metadata and "file_name" in metadata:
                file_name = metadata["file_name"]
                file_path = metadata.get("file_path", "Unknown path")
                file_key = f"{file_name} ({file_path})"
                
                if file_key not in file_info:
                    file_info[file_key] = {
                        "file_name": file_name,
                        "file_path": file_path,
                        "file_type": metadata.get("file_type", "Unknown"),
                        "file_size": metadata.get("file_size", 0),
                        "creation_date": metadata.get("creation_date", "Unknown"),
                        "last_modified_date": metadata.get("last_modified_date", "Unknown"),
                        "ingestion_method": metadata.get("ingestion_method", "Unknown"),
                        "chunks_found": 0,
                        "total_chunk_size": 0
                    }
                file_info[file_key]["chunks_found"] += 1
                file_info[file_key]["total_chunk_size"] += metadata.get("chunk_size", 0)
        
        if not file_info:
            return "No files have been ingested yet."
        
        # Format response
        response = f"Ingested Files ({len(file_info)} total):\n\n"
        for i, (file_key, info) in enumerate(file_info.items(), 1):
            response += f"{i}. {info['file_name']}\n"
            response += f"   Path: {info['file_path']}\n"
            response += f"   Type: {info['file_type']}\n"
            response += f"   Size: {info['file_size']:,} bytes\n"
            response += f"   Created: {info['creation_date']}\n"
            response += f"   Modified: {info['last_modified_date']}\n"
            response += f"   Chunks: {info['chunks_found']}\n"
            response += f"   Total chunk size: {info['total_chunk_size']:,} characters\n"
            response += f"   Ingestion method: {info['ingestion_method']}\n\n"
        
        total_chunks = sum(info["chunks_found"] for info in file_info.values())
        total_chunk_size = sum(info["total_chunk_size"] for info in file_info.values())
        response += f"Total chunks in database: {total_chunks}\n"
        response += f"Total content size: {total_chunk_size:,} characters"
        
        return response
        
    except Exception as e:
        error_msg = f"Error listing ingested files: {str(e)}"
        logger.error(error_msg)
        return error_msg

@mcp.tool
def reingest_data_directory() -> str:
    """
    Reingest all files from the data directory into the vector database.
    
    This tool clears the existing database and reprocesses all files in the data directory,
    which is useful to reindex the contents when new files have been added to the data directory
    or when you want to refresh the entire database with the latest file contents.
    
    Returns:
        Status message indicating success or failure with details about ingested files
    """
    global collection
    try:
        if not collection:
            return "Error: Database is not initialized."
        
        # First, clear the existing database
        logger.info("Clearing existing database before reingestion...")
        chroma_client.delete_collection(name="rag_documents")
        collection = chroma_client.create_collection(
            name="rag_documents",
            metadata={"description": "Collection for RAG document storage"}
        )
        
        # Now reingest all files from data directory
        logger.info("Starting reingestion of data directory...")
        auto_ingest_files()
        
        # Get final count
        final_count = collection.count()
        
        success_msg = f"Successfully reingested data directory. Database now contains {final_count} documents."
        logger.info(success_msg)
        return success_msg
        
    except Exception as e:
        error_msg = f"Error during reingestion: {str(e)}"
        logger.error(error_msg)
        return error_msg

@mcp.resource("rag://status")
def get_rag_status() -> Dict[str, Any]:
    """
    Get the current status of the RAG system.
    
    Returns:
        Dictionary with system status information
    """
    global collection
    try:
        doc_count = collection.count() if collection else 0
        
        return {
            "status": "active",
            "database_type": "ChromaDB",
            "total_documents": doc_count,
            "collection_name": "rag_documents",
            "persist_directory": "./chroma_db"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@mcp.prompt
def rag_analysis_prompt(topic: str) -> PromptMessage:
    """
    Generate a prompt for analyzing documents related to a specific topic.
    
    Args:
        topic: The topic to analyze
    
    Returns:
        PromptMessage for RAG analysis
    """
    text = dedent(f"""
      Please analyze the documents in the RAG database related to '{topic}'. 

      First, query the database for relevant information about this topic, then provide:
      1. A comprehensive summary of the key points
      2. Any important insights or patterns you notice
      3. Potential areas for further investigation
      4. Sources and references from the retrieved documents

      Use the query_documents tool to search for information about '{topic}' and base your analysis on the retrieved content.
      """)
    
    return PromptMessage(
        role="user",
        content=TextContent(type="text", text=text)
    )

if __name__ == "__main__":
    # Run the MCP server
    logger.info("Starting RAG MCP Server...")
    mcp.run("stdio")