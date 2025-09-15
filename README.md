# RAG MCP Server

A **Retrieval Augmented Generation (RAG)** server built with [FastMCP](https://github.com/jlowin/fastmcp) <mcreference link="https://github.com/jlowin/fastmcp" index="1">1</mcreference> and [ChromaDB](https://docs.trychroma.com/) <mcreference link="https://docs.trychroma.com/docs/overview/getting-started" index="2">2</mcreference> that provides MCP (Model Context Protocol) tools for ingesting documents into a local vector database and retrieving relevant information based on queries.

## Features

### 🔧 **Tools**
- **`ingest_file`**: Ingest text files into the vector database with configurable chunking
- **`query_documents`**: Search for relevant documents using semantic similarity
- **`list_ingested_files`**: View all files currently stored in the database
- **`clear_database`**: Remove all documents from the vector database

### 📊 **Resources**
- **`rag://status`**: Get real-time status of the RAG system including document count and configuration

### 💬 **Prompts**
- **`rag_analysis_prompt`**: Generate structured prompts for analyzing documents on specific topics

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install manually
pip install fastmcp chromadb sentence-transformers
```

### 2. Run the Server

```bash
# Start the MCP server
python rag_server.py

# Or use FastMCP CLI for development with inspector
fastmcp dev rag_server.py
```

### 3. Test the Server

```bash
# Run the test suite
python test_rag_server.py
```

## Usage Examples

### Ingesting Documents

```python
# The server will chunk your document automatically
result = ingest_file(
    file_path="sample_document.txt",
    chunk_size=1000,  # Characters per chunk
    overlap=200       # Overlap between chunks
)
```

### Querying Documents

```python
# Search for relevant information
results = query_documents(
    query="What is machine learning?",
    n_results=5,
    include_metadata=True
)
```

### Checking System Status

```python
# Get current system information
status = get_rag_status()
# Returns: {"status": "active", "total_documents": 42, ...}
```

## Architecture

### Components

1. **FastMCP Server**: High-level MCP server framework <mcreference link="https://github.com/jlowin/fastmcp" index="1">1</mcreference>
2. **ChromaDB**: Local vector database for document storage <mcreference link="https://docs.trychroma.com/docs/overview/getting-started" index="2">2</mcreference>
3. **Sentence Transformers**: Embedding model for semantic search

### Data Flow

```
Text File → Chunking → Embeddings → ChromaDB → Query → Relevant Chunks
```

### File Structure

```
mcp-rag/
├── rag_server.py           # Main MCP server implementation
├── requirements.txt        # Python dependencies
├── test_rag_server.py     # Test suite
├── sample_document.txt    # Example document for testing
├── README.md              # This file
└── chroma_db/             # ChromaDB persistent storage (created automatically)
```

## Configuration

### Environment Variables

The server uses sensible defaults, but you can customize:

- **Database Location**: Modify `persist_directory` in `rag_server.py`
- **Collection Name**: Change `rag_documents` to your preferred name
- **Chunk Settings**: Adjust default `chunk_size` and `overlap` parameters

### ChromaDB Settings

```python
# Persistent storage configuration
chroma_client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(
        anonymized_telemetry=False,
        allow_reset=True
    )
)
```

## Integration with MCP Clients

### Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "rag-server": {
      "command": "python",
      "args": ["/path/to/your/rag_server.py"],
      "cwd": "/path/to/your/mcp-rag"
    }
  }
}
```

### Cursor IDE

Add to your MCP configuration:

```json
{
  "mcpServers": {
    "rag-server": {
      "command": "python",
      "args": ["rag_server.py"],
      "cwd": "/path/to/mcp-rag"
    }
  }
}
```

## Development

### Testing with MCP Inspector

FastMCP includes a built-in web interface for testing:

```bash
# Install with CLI tools
pip install "fastmcp[cli]"

# Run with inspector
fastmcp dev rag_server.py

# Open browser to http://127.0.0.1:6274
```

### Adding New Tools

```python
@mcp.tool
def your_new_tool(param: str) -> str:
    """
    Description of your tool.
    
    Args:
        param: Description of parameter
    
    Returns:
        Description of return value
    """
    # Your implementation here
    return "result"
```

### Adding Resources

```python
@mcp.resource("your://resource-uri")
def your_resource() -> dict:
    """
    Description of your resource.
    """
    return {"data": "value"}
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install --upgrade fastmcp chromadb
   ```

2. **ChromaDB Permission Issues**
   ```bash
   # Ensure write permissions for chroma_db directory
   chmod -R 755 ./chroma_db
   ```

3. **Memory Issues with Large Files**
   - Reduce `chunk_size` parameter
   - Process files in smaller batches
   - Monitor system memory usage

4. **Slow Query Performance**
   - Reduce `n_results` parameter
   - Consider using more specific queries
   - Check ChromaDB index status

### Logging

The server includes comprehensive logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)  # Enable debug logging
```

## Performance Considerations

### Optimization Tips

1. **Chunk Size**: Balance between context and performance (500-2000 characters)
2. **Overlap**: Prevent context loss at chunk boundaries (10-20% of chunk size)
3. **Query Results**: Limit `n_results` to avoid overwhelming responses (3-10 results)
4. **File Size**: Consider splitting very large files before ingestion

### Scaling

For production use:

- Consider ChromaDB's client-server mode
- Implement batch processing for large document sets
- Add caching for frequently accessed documents
- Monitor disk space for the vector database

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is open source. Feel free to use, modify, and distribute according to your needs.

## References

- [Model Context Protocol Documentation](https://modelcontextprotocol.io/) <mcreference link="https://modelcontextprotocol.io/llms-full.txt" index="0">0</mcreference>
- [FastMCP Framework](https://github.com/jlowin/fastmcp) <mcreference link="https://github.com/jlowin/fastmcp" index="1">1</mcreference>
- [ChromaDB Documentation](https://docs.trychroma.com/) <mcreference link="https://docs.trychroma.com/docs/overview/getting-started" index="2">2</mcreference>

---

**Built with ❤️ using FastMCP and ChromaDB**