import logging
from rag_server import RAGServer, mcp

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Create an instance of our service, which will register the tools.
    RAGServer()
    
    # Run the MCP server
    logger.info("Starting RAG MCP Server...")
    mcp.run("stdio")