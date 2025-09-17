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
from typing import Dict, Any
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
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastMCP server with RAG capabilities
mcp = FastMCP("RAG Server")

class RAGServer:
    def __init__(self):
        """Initializes the RAG server, setting up the database connection."""
        self.chroma_client = None
        self.collection = None
        self._initialize_chromadb()

    def _initialize_chromadb(self):
        """Initialize ChromaDB client and collection, then auto-ingest files from data directory"""
        try:
            # Get database directory using flexible resolution
            persist_directory = self._get_database_directory()
            safe_reset_chromadb(chroma_path=str(persist_directory))
            
            os.makedirs(persist_directory, exist_ok=True)
            
            self.chroma_client = chromadb.PersistentClient(
                path=str(persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Create fresh collection for RAG documents
            self.collection = self.chroma_client.create_collection(
                name="rag_documents",
                metadata={"description": "Collection for RAG document storage"}
            )
            
            logger.info(f"ChromaDB initialized successfully. Vector database has {self.collection.count()} documents.")
            
            # Auto-ingest files from data directory
            self._auto_ingest_files()
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

    def _get_database_directory(self):
        """Get the ChromaDB persistent directory path with flexible resolution strategy."""
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
        
        # 3. No environment variable and no accessible XDG directory - raise error
        error_msg = (
            "No database directory found. Please either:\n"
            "1. Set the LLAMA_RAG_DB_DIR environment variable to specify a database directory, or\n"
            "2. Ensure the XDG Base Directory standard location is accessible\n\n"
            "Examples:\n"
            "  export LLAMA_RAG_DB_DIR=/path/to/your/database\n"
            "  # Or ensure ~/.local/share is accessible for XDG standard"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    def _get_data_directory(self):
        """Get the data directory path with flexible resolution strategy."""
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

    def _check_data_directory_configured(self):
        """Check if data directory is properly configured."""
        try:
            data_path = self._get_data_directory()
            return True, f"Data directory configured: {data_path}"
        except ValueError:
            message = (
                "No data directory is configured for this RAG system. "
                "The system cannot access any documents without a data directory.\n\n"
                "To set up a data directory, you can:\n"
                "1. Set the LLAMA_RAG_DATA_DIR environment variable:\n"
                "   export LLAMA_RAG_DATA_DIR=/path/to/your/documents\n"
                "2. Create a 'data' directory in the current working directory:\n"
                "   mkdir data\n\n"
                "After setting up the data directory, add your documents to it and restart the server "
                "or use the reingest_data_directory tool to load them."
            )
            return False, message

    def _auto_ingest_files(self):
        """Automatically ingest all files from the data directory"""
        try:
            data_path = self._get_data_directory()
            
            os.makedirs(data_path, exist_ok=True)
            
            files = list(data_path.glob("*"))
            files = [f for f in files if f.is_file()]
            
            if not files:
                logger.info("No files found in data directory. Skipping auto-ingestion.")
                return
            
            logger.info(f"Found {len(files)} files in data directory. Starting auto-ingestion...")
            
            api_key = os.getenv("LLAMA_CLOUD_API_KEY")
            if not api_key:
                logger.warning("LLAMA_CLOUD_API_KEY not set. PDF parsing will be limited.")
                parser = None
            else:
                parser = LlamaParse(api_key=api_key, result_type="text")
            
            file_extractor = {}
            if parser:
                file_extractor = {
                    ".pdf": parser,
                    ".docx": parser,
                    ".pptx": parser,
                    ".doc": parser,
                    ".ppt": parser
                }
            
            documents = SimpleDirectoryReader(
                input_dir=str(data_path),
                file_extractor=file_extractor,
                recursive=True
            ).load_data()
            
            if not documents:
                logger.info("No documents loaded from data directory.")
                return
            
            logger.info(f"Loaded {len(documents)} documents from data directory.")
            
            for doc in documents:
                try:
                    content = doc.text
                    metadata = doc.metadata or {}
                    id = doc.id_
                    
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
                    
                    self.collection.add(
                        documents=[content],
                        metadatas=[metadata],
                        ids=[id]
                    )
                    
                    logger.info(f"Successfully ingested: {file_name}")
                    
                except Exception as e:
                    logger.error(f"Failed to ingest document: {e}")
                    continue
            
            final_count = self.collection.count()
            logger.info(f"Auto-ingestion completed. Collection now has {final_count} documents.")
            
        except ValueError as e:
            logger.warning(f"Skipping auto-ingestion: {e}")
            return
        except Exception as e:
            logger.error(f"Failed during auto-ingestion: {e}")

    @mcp.tool
    def query_documents(self, query: str, n_results: int = 5, include_metadata: bool = True) -> str:
        """
        Query the local document knowledge base (vector database) to retrieve relevant information based on your search query.
        
        This tool allows you to search through all documents that have been ingested into the system's vector database.
        It uses semantic search to find the most relevant text passages that match your query, going beyond simple
        keyword matching to understand the meaning and context of your search.
        
        The tool is particularly useful for:
        - Finding specific information across multiple documents
        - Discovering connections between different pieces of content
        - Getting quick access to relevant passages without reading entire documents
        - Verifying facts or finding supporting evidence in your document collection
        
        Each result includes:
        - The relevant text passage
        - Source document information
        - Similarity score showing how well it matches your query
        - Additional metadata about the document chunk
        
        You can control the number of results returned and whether to include detailed metadata
        in the response.
        
        Args:
            query: The search query
            n_results: Number of results to return (default: 5, min: 5, max: 20)
            include_metadata: Whether to include metadata in results (default: True)
        
        Returns:
            Formatted string with relevant documents and their metadata
        """
        try:
            is_configured, config_message = self._check_data_directory_configured()
            if not is_configured:
                return config_message
            
            if not query.strip():
                return "Error: Query cannot be empty."
            
            if n_results <= 0:
                n_results = 5
            elif n_results > 20:
                n_results = 20
            
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            if not results["documents"] or not results["documents"][0]:
                return "No relevant documents found for your query."
            
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
    def list_ingested_files(self) -> str:
        """
        Provides a comprehensive list of all files that have been successfully ingested into the vector database.

        This tool is useful for:
        - Verifying which files are included in the knowledge base.
        - Checking the metadata of each file, such as file type, size, and ingestion date.
        - Understanding how documents are chunked and stored in the database.

        The output includes:
        - A summary of each ingested file with its path, type, size, and modification dates.
        - The number of chunks each file has been divided into.
        - The total size of the content stored in the database.
        """
        try:
            is_configured, config_message = self._check_data_directory_configured()
            if not is_configured:
                return config_message
            
            all_docs = self.collection.get(include=["metadatas"])
            
            if not all_docs["metadatas"]:
                return "No files have been ingested yet."
            
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
    def reingest_data_directory(self) -> str:
        """
        Performs a complete re-ingestion of all files from the configured data directory into the vector database.

        This tool first clears the existing database of all documents and then processes all files in the data directory from scratch.
        It is particularly useful when:
        - New files have been added to the data directory and need to be included in the knowledge base.
        - Existing files have been updated and their contents need to be re-indexed.
        - You want to ensure the vector database is in a clean, consistent state with the latest file versions.

        The process is atomic; it completely replaces the old database with a new one.

        Returns:
            A status message indicating the success or failure of the re-ingestion process, including the final document count.
        """
        try:
            is_configured, config_message = self._check_data_directory_configured()
            if not is_configured:
                return config_message
            
            if not self.collection:
                return "Error: Database is not initialized."
            
            logger.info("Clearing existing database before reingestion...")
            self.chroma_client.delete_collection(name="rag_documents")
            self.collection = self.chroma_client.create_collection(
                name="rag_documents",
                metadata={"description": "Collection for RAG document storage"}
            )
            
            logger.info("Starting reingestion of data directory...")
            self._auto_ingest_files()
            
            final_count = self.collection.count()
            
            success_msg = f"Successfully reingested data directory. Database now contains {final_count} documents."
            logger.info(success_msg)
            return success_msg
            
        except Exception as e:
            error_msg = f"Error during reingestion: {str(e)}"
            logger.error(error_msg)
            return error_msg

    @mcp.tool()
    def get_rag_status(self) -> Dict[str, Any]:
        """
        Provides a comprehensive overview of the RAG system's status and configuration.

        This tool is essential for diagnosing issues and understanding the current state of the system.
        It returns a detailed report including:
        - System Status: Whether the server is active, the database is initialized, and the total number of documents.
        - Database Configuration: The type of database, its storage directory, and collection details.
        - Data Directory: The path to the data directory, whether it exists, and how it's configured.
        - Environment Variables: The status of relevant environment variables (e.g., API keys, custom paths).
        - Configuration Priority: The order in which the system looks for configuration settings.

        This tool helps you to:
        - Verify that the system is running correctly.
        - Debug configuration problems related to data and database directories.
        - Check which environment variables are being used.
        """
        try:
            doc_count = self.collection.count() if self.collection else 0
            
            db_directory = self._get_database_directory()
            
            data_directory_status = {}
            try:
                data_directory = self._get_data_directory()
                data_directory_status = {
                    "path": str(data_directory),
                    "exists": data_directory.exists(),
                    "configured": True,
                    "source": "environment" if os.getenv('LLAMA_RAG_DATA_DIR') else "workspace"
                }
            except ValueError as e:
                data_directory_status = {
                    "path": None,
                    "exists": False,
                    "configured": False,
                    "error": str(e),
                    "source": "none"
                }
            
            env_vars = {
                "LLAMA_RAG_DATA_DIR": {
                    "set": bool(os.getenv('LLAMA_RAG_DATA_DIR')),
                    "value": os.getenv('LLAMA_RAG_DATA_DIR')
                },
                "LLAMA_RAG_DB_DIR": {
                    "set": bool(os.getenv('LLAMA_RAG_DB_DIR')),
                    "value": os.getenv('LLAMA_RAG_DB_DIR')
                },
                "LLAMA_CLOUD_API_KEY": {
                    "set": bool(os.getenv('LLAMA_CLOUD_API_KEY')),
                    "value": "[REDACTED]" if os.getenv('LLAMA_CLOUD_API_KEY') else None
                }
            }
            
            db_config = {
                "type": "ChromaDB",
                "directory": str(db_directory),
                "exists": db_directory.exists(),
                "collection_name": "rag_documents",
                "source": "environment" if os.getenv('LLAMA_RAG_DB_DIR') else "standard"
            }
            
            system_status = {
                "server_active": True,
                "database_initialized": self.chroma_client is not None,
                "collection_ready": self.collection is not None,
                "total_documents": doc_count,
                "auto_ingestion_enabled": data_directory_status["configured"]
            }
            
            return {
                "status": "active",
                "system": system_status,
                "database": db_config,
                "data_directory": data_directory_status,
                "environment_variables": env_vars,
                "configuration": {
                    "data_dir_priority": [
                        "LLAMA_RAG_DATA_DIR environment variable",
                        "./data in current working directory",
                        "Error if neither found"
                    ],
                    "db_dir_priority": [
                        "LLAMA_RAG_DB_DIR environment variable",
                        "~/.local/share/rag-server (XDG standard)",
                        "./chroma in current working directory"
                    ]
                }
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "system": {
                    "server_active": True,
                    "database_initialized": False,
                    "collection_ready": False,
                    "total_documents": 0,
                    "auto_ingestion_enabled": False
                }
            }

    @mcp.prompt
    def rag_analysis_prompt(self, topic: str) -> PromptMessage:
        """
        Generates a sophisticated prompt to guide the AI in performing an in-depth analysis of documents related to a specific topic.

        This tool is designed to kickstart a detailed investigation into a subject by leveraging the RAG system's capabilities.
        It constructs a prompt that instructs the AI to:
        1. Query the knowledge base for information on the given topic.
        2. Synthesize the findings into a comprehensive summary.
        3. Identify key insights, patterns, and connections within the documents.
        4. Suggest areas for further exploration.
        5. Cite the sources from the retrieved documents to support its analysis.

        Use this tool to:
        - Initiate a research task on a particular topic.
        - Generate a structured analysis of your documents.
        - Uncover deeper insights that may not be apparent from a simple query.
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
    # Create an instance of our service, which will register the tools.
    RAGServer()
    
    # Run the MCP server
    logger.info("Starting RAG MCP Server...")
    mcp.run("stdio")