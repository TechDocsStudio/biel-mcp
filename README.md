# Biel.ai MCP Server

MCP (Model Context Protocol) server to integrate Biel.ai with editors like Cursor.

## 🚀 Cursor Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Cursor

1. **Open Cursor settings:**
   - Press `Ctrl+Shift+J` (Windows/Linux) or `Cmd+Shift+J` (Mac)
   -  Go to "MCP" and "Add new global MCP Server"

2. **Add MCP configuration:**
   Add this configuration:

```json
{
  "mcpServers": {
    "biel-ai": {
      "command": "python",
      "args": ["/ABSOLUTE/PATH/TO/PARENT/FOLDER/biel-mcp/biel_mcp_server.py"],
       "env": {
         "BIEL_API_KEY": "your-api-key",
         "BIEL_BASE_URL": "http://localhost:8000",
         "BIEL_PROJECT_SLUG": "your-project-slug"
       }
    }
  }
}
```

**⚠️ Important:** Change the paths to the absolute path where you have this project.

### 3. Restart Cursor

After adding the configuration, restart Cursor completely.

### 4. Verify connection

1. Open Cursor chat (Ctrl+L)
2. You can ask: "What is Biel.ai?"

## 🔧 Usage

Once configured, you can use Biel.ai directly from Cursor:

### Conversation management

The server **automatically maintains context** between queries:

1. **First query:** Starts a new conversation
2. **Following queries:** Automatically continue the previous conversation
3. **New conversation:** Use `new_conversation: true`

### Usage examples

```bash
# First question (starts conversation)
@Biel.ai What is Biel?

# Second question (automatically maintains context)
@Biel.ai How to add Biel in Docusaurus?

# Start new conversation
@Biel.ai {"message": "How do I use Biel.ai in React?", "new_conversation": true}

# Use a specific chat_uuid
@Biel.ai {"message": "Continue explaining", "chat_uuid": "abc123..."}
```

Or simply ask in the chat and Cursor will automatically use the tool when relevant.

## 📚 More information

- [MCP Documentation](https://modelcontextprotocol.io/)
- [Cursor MCP Integration](https://docs.cursor.com/context/model-context-protocol) 