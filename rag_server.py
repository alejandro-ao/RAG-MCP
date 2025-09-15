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

# LlamaParse and LlamaIndex imports
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastMCP server with RAG capabilities
mcp = FastMCP("RAG Server", dependencies=["chromadb", "sentence-transformers"])

# Global ChromaDB client and collection
chroma_client = None
collection = None

def initialize_chromadb():
    """Initialize ChromaDB client and collection, then auto-ingest files from data directory"""
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
        
        # Auto-ingest files from data directory
        auto_ingest_files()
        
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB: {e}")
        raise

def auto_ingest_files():
    """Automatically ingest all files from the data directory"""
    global collection
    
    try:
        # Create data directory if it doesn't exist
        data_directory = "./data"
        os.makedirs(data_directory, exist_ok=True)
        
        # Check if data directory has any files
        data_path = Path(data_directory)
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
            input_dir=data_directory,
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
                
                # Generate unique ID for the document
                doc_id = str(uuid.uuid4())
                
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
    mcp.run("stdio")