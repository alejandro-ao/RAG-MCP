#!/usr/bin/env python3
"""
RAG MCP Server - Retrieval Augmented Generation using FastMCP and ChromaDB

This server provides tools for:
1. Ingesting files into a local vector database (ChromaDB)
2. Retrieving relevant information based on queries
"""

import os
import logging
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastMCP server with RAG capabilities
mcp = FastMCP("RAG Server", dependencies=["chromadb", "sentence-transformers"])

# Global ChromaDB client and collection
chroma_client = None
collection = None

def initialize_chromadb():
    """Initialize ChromaDB client and collection"""
    global chroma_client, collection
    
    try:
        # Create persistent ChromaDB client
        persist_directory = "./chroma_db"
        os.makedirs(persist_directory, exist_ok=True)
        
        chroma_client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection for RAG documents
        collection = chroma_client.get_or_create_collection(
            name="rag_documents",
            metadata={"description": "Collection for RAG document storage"}
        )
        
        logger.info(f"ChromaDB initialized successfully. Collection has {collection.count()} documents.")
        
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB: {e}")
        raise

# Initialize ChromaDB on server startup
initialize_chromadb()

@mcp.tool
def ingest_file(file_path: str, chunk_size: int = 5000, overlap: int = 800) -> str:
    """
    Ingest a text file into the vector database for RAG.
    
    Args:
        file_path: Path to the text file to ingest
        chunk_size: Size of text chunks (default: 5000 characters)
        overlap: Overlap between chunks (default: 800 characters)
    
    Returns:
        Status message indicating success or failure
    """
    global collection
    try:
        # Validate file exists and is readable
        path = Path(file_path)
        if not path.exists():
            return f"Error: File '{file_path}' does not exist."
        
        if not path.is_file():
            return f"Error: '{file_path}' is not a file."
        
        # Read file content
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(path, 'r', encoding='latin-1') as f:
                content = f.read()
        
        if not content.strip():
            return f"Error: File '{file_path}' is empty."
        
        # Split content into chunks
        chunks = []
        start = 0
        while start < len(content):
            end = start + chunk_size
            chunk = content[start:end]
            
            # Ensure we don't cut words in half
            if end < len(content) and not content[end].isspace():
                last_space = chunk.rfind(' ')
                if last_space > start:
                    end = start + last_space
                    chunk = content[start:end]
            
            chunks.append(chunk.strip())
            start = end - overlap
        
        # Generate unique IDs for chunks
        chunk_ids = [f"{path.stem}_{i}_{uuid.uuid4().hex[:8]}" for i in range(len(chunks))]
        
        # Create metadata for each chunk
        metadatas = [{
            "source_file": str(path),
            "chunk_index": i,
            "total_chunks": len(chunks),
            "chunk_size": len(chunk)
        } for i, chunk in enumerate(chunks)]
        
        # Add chunks to ChromaDB collection
        collection.add(
            documents=chunks,
            metadatas=metadatas,
            ids=chunk_ids
        )
        
        logger.info(f"Successfully ingested {len(chunks)} chunks from '{file_path}'")
        return f"Successfully ingested '{file_path}' into vector database. Created {len(chunks)} chunks."
        
    except Exception as e:
        error_msg = f"Error ingesting file '{file_path}': {str(e)}"
        logger.error(error_msg)
        return error_msg

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
        
        # Group by source file
        file_info = {}
        for metadata in all_docs["metadatas"]:
            if metadata and "source_file" in metadata:
                source_file = metadata["source_file"]
                if source_file not in file_info:
                    file_info[source_file] = {
                        "total_chunks": metadata.get("total_chunks", 0),
                        "chunks_found": 0
                    }
                file_info[source_file]["chunks_found"] += 1
        
        if not file_info:
            return "No files have been ingested yet."
        
        # Format response
        response = f"Ingested Files ({len(file_info)} total):\n\n"
        for i, (file_path, info) in enumerate(file_info.items(), 1):
            response += f"{i}. {file_path}\n"
            response += f"   Chunks: {info['chunks_found']}/{info['total_chunks']}\n\n"
        
        total_chunks = sum(info["chunks_found"] for info in file_info.values())
        response += f"Total chunks in database: {total_chunks}"
        
        return response
        
    except Exception as e:
        error_msg = f"Error listing ingested files: {str(e)}"
        logger.error(error_msg)
        return error_msg

@mcp.tool
def clear_database() -> str:
    """
    Clear all documents from the vector database.
    
    Returns:
        Status message indicating success or failure
    """
    global collection, chroma_client
    try:
        # Get count before clearing
        count_before = collection.count()
        
        # Delete the collection and recreate it
        chroma_client.delete_collection("rag_documents")
        
        collection = chroma_client.create_collection(
            name="rag_documents",
            metadata={"description": "Collection for RAG document storage"}
        )
        
        logger.info(f"Cleared {count_before} documents from the database")
        return f"Successfully cleared {count_before} documents from the vector database."
        
    except Exception as e:
        error_msg = f"Error clearing database: {str(e)}"
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
    mcp.run()