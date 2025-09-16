# Directory Configuration

The RAG MCP Server supports flexible configuration for both data and database directories through environment variables, allowing you to specify custom locations for your document storage and ChromaDB database.

## Data Directory Configuration

The server uses the following priority order to determine the data directory:

1. **`LLAMA_RAG_DATA_DIR` environment variable** (highest priority)
2. **`./data` in current working directory** (workspace-relative)
3. **`./data` relative to server file** (fallback)

## Database Directory Configuration

The server uses the following priority order to determine the ChromaDB database directory:

1. **`LLAMA_RAG_DB_DIR` environment variable** (highest priority)
2. **`~/.local/share/rag-server`** (XDG Base Directory standard)
3. **`./chroma` relative to current working directory** (fallback)

## Usage Examples

### 1. Data Directory Environment Variable

#### Set for current session:
```bash
export LLAMA_RAG_DATA_DIR=/path/to/your/data
python rag_server.py
```

#### Set for single command:
```bash
LLAMA_RAG_DATA_DIR=/path/to/your/data python rag_server.py
```

### 2. Database Directory Environment Variable

#### Set for current session:
```bash
export LLAMA_RAG_DB_DIR=/path/to/your/database
python rag_server.py
```

#### Set for single command:
```bash
LLAMA_RAG_DB_DIR=/path/to/your/database python rag_server.py
```

#### Use both together:
```bash
export LLAMA_RAG_DATA_DIR=/path/to/documents
export LLAMA_RAG_DB_DIR=/path/to/database
python rag_server.py
```

### 3. Common Use Cases

#### Development Environment:
```bash
# Use project-local directories
LLAMA_RAG_DATA_DIR=./dev_data LLAMA_RAG_DB_DIR=./dev_db python rag_server.py
```

#### Production Environment:
```bash
# Use system-wide directories
LLAMA_RAG_DATA_DIR=/var/lib/rag-server/data LLAMA_RAG_DB_DIR=/var/lib/rag-server/db python rag_server.py
```

#### User-specific Setup:
```bash
# Use user's home directory
LLAMA_RAG_DATA_DIR=~/Documents/rag-data LLAMA_RAG_DB_DIR=~/Documents/rag-db python rag_server.py
```

#### Temporary Testing:
```bash
# Use temporary directories for testing
LLAMA_RAG_DATA_DIR=/tmp/test_data LLAMA_RAG_DB_DIR=/tmp/test_db python rag_server.py
```

### 4. Default Behavior (No Environment Variables)

#### Data Directory (`LLAMA_RAG_DATA_DIR` not set):
- If `./data` exists in current working directory → uses workspace-relative `./data`
- Otherwise → uses server-relative `./data` (fallback)

#### Database Directory (`LLAMA_RAG_DB_DIR` not set):
1. Server first tries `~/.local/share/rag-server` (XDG Base Directory standard)
2. Falls back to `./chroma` relative to the current working directory

```bash
# This will use default directory resolution for both
python rag_server.py
```

## Features

### Data Directory Features
- **Environment Variable Support**: Use `LLAMA_RAG_DATA_DIR` to specify custom data directory
- **Flexible Path Resolution**: Supports absolute paths, relative paths, and tilde expansion (`~`)
- **Workspace Awareness**: Automatically detects if running in a workspace with existing data directory
- **Graceful Fallbacks**: Multiple fallback options ensure the server always finds a valid directory
- **Directory Creation**: Creates the data directory if it doesn't exist
- **Logging**: Logs which data directory is being used for transparency

### Database Directory Features
- **Environment Variable Support**: Use `LLAMA_RAG_DB_DIR` to specify custom database directory
- **XDG Base Directory Standard**: Follows Linux/Unix standards for user data storage
- **Flexible Path Resolution**: Supports absolute paths, relative paths, and tilde expansion (`~`)
- **Cross-Platform Compatibility**: Works on Windows, macOS, and Linux with appropriate fallbacks
- **Persistent Storage**: ChromaDB database persists across server restarts

## Testing

### Test Data Directory Configuration:

```bash
# Test with custom data directory
LLAMA_RAG_DATA_DIR=/tmp/test_rag_data python rag_server.py

# Check the logs to confirm the directory is being used
# Look for: "Using data directory: /tmp/test_rag_data"
```

### Test Database Directory Configuration:

```bash
# Test with custom database directory
LLAMA_RAG_DB_DIR=/tmp/test_rag_database python rag_server.py

# Check the logs to confirm the directory is being used
# Look for: "Using database directory from LLAMA_RAG_DB_DIR: /tmp/test_rag_database"
```

### Test Both Together:

```bash
# Test with both custom directories
LLAMA_RAG_DATA_DIR=/tmp/test_data LLAMA_RAG_DB_DIR=/tmp/test_db python rag_server.py
```

### Verification Steps:
1. Set the environment variable(s)
2. Start the server
3. Check the server logs for directory confirmation
4. Verify files are created in the specified directories
5. Test document ingestion and querying
6. Verify ChromaDB persistence across server restarts

### Run Test Script:

```bash
python test_env_data_dir.py
```

This will test all scenarios:
- No environment variable set
- Custom absolute path
- Tilde expansion
- Relative path resolution

## Integration with MCP Clients

When using with MCP clients (like Claude Desktop), you can set the environment variable in your shell configuration:

### For bash/zsh:
```bash
# Add to ~/.bashrc or ~/.zshrc
export LLAMA_RAG_DATA_DIR=~/Documents/rag_documents
```

### For fish shell:
```fish
# Add to ~/.config/fish/config.fish
set -gx LLAMA_RAG_DATA_DIR ~/Documents/rag_documents
```

## Benefits

1. **Flexibility**: Use any directory on your system
2. **Project Isolation**: Different projects can use different data directories
3. **Shared Storage**: Multiple instances can share the same data directory
4. **User Control**: Users can organize their documents as they prefer
5. **Backward Compatibility**: Existing setups continue to work without changes

## Troubleshooting

### Check Current Configuration
Run the test script to see which directory is being used:
```bash
python test_env_data_dir.py
```

### Verify Environment Variable
```bash
echo $LLAMA_RAG_DATA_DIR
```

### Check Server Logs
The server logs which data directory it's using during startup:
```
2025-09-16 12:51:01,640 - __main__ - INFO - Using data directory from LLAMA_RAG_DATA_DIR: /path/to/custom/data
```

### Common Issues

1. **Permission Denied**: Ensure the specified directory is writable
2. **Path Not Found**: Check that parent directories exist or use absolute paths
3. **Environment Variable Not Set**: Verify the variable is exported in your shell

## Security Considerations

- The server will create directories if they don't exist
- Ensure the specified path is secure and not accessible by unauthorized users
- Use absolute paths when possible to avoid confusion
- Be cautious with shared directories in multi-user environments