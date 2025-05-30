"""
MCP Server for Biel.ai
Allows querying your AI from editors like Cursor via MCP
"""

import asyncio
import logging
import os
import httpx
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    TextContent,
    Tool,
)

# Configuration from environment variables
BIEL_API_URL = os.getenv("BIEL_API_URL", "/api/v1/chat/")
PROJECT_SLUG = os.getenv("BIEL_PROJECT_SLUG", "")
BASE_URL = os.getenv("BIEL_BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("BIEL_API_KEY", "")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("biel-mcp")

# Global variable to maintain the last chat_uuid
last_chat_uuid = ""

# Create MCP server
server = Server("biel-ai")

@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """List available tools"""
    return [
        Tool(
            name="query_biel_ai",
            description="Query Biel.ai's specialized AI about code, SDKs and documentation",
            inputSchema={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Your question about code, SDK or documentation"
                    },
                    "chat_uuid": {
                        "type": "string",
                        "description": "Chat UUID to continue conversation (optional)",
                        "default": ""
                    }
                },
                "required": ["message"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls"""
    
    if name == "query_biel_ai":
        return await query_biel_ai(arguments)
    else:
        raise ValueError(f"Unknown tool: {name}")

async def query_biel_ai(arguments: dict) -> list[TextContent]:
    """Query your Biel.ai API"""
    
    message = arguments.get("message", "")
    chat_uuid = arguments.get("chat_uuid", "")
    
    if not message:
        return [TextContent(type="text", text="Error: Message cannot be empty")]
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            payload = {
                "message": message,
                "project_slug": PROJECT_SLUG,
                "url": BASE_URL
            }
            
            # If there's a chat_uuid, include it to continue conversation
            if chat_uuid:
                payload["chat_uuid"] = chat_uuid
            
            headers = {
                "Content-Type": "application/json",
            }
            
            if API_KEY:
                headers["Authorization"] = f"Api-Key {API_KEY}"
            
            logger.info(f"Querying Biel.ai: {message[:50]}...")
            
            response = await client.post(
                BASE_URL + BIEL_API_URL + PROJECT_SLUG + "/",
                json=payload,
                headers=headers
            )
            
            if response.status_code == 200 or response.status_code == 201:
                data = response.json()
                
                # Extract AI response
                ai_response = data.get("ai_message", {}).get("message", "No response received")
                chat_uuid = data.get("chat_uuid", "")
                sources = data.get("ai_message", {}).get("sources", [])
                
                # Format response
                response_text = f"🤖 **Biel.ai responds:**\n\n{ai_response}"
                
                # Add sources if they exist
                if sources:
                    response_text += "\n\n📚 **Sources consulted:**\n"
                    for source in sources:
                        response_text += f"• [{source['title']}]({source['url']})\n"
                
                # Add chat_uuid for possible future queries
                if chat_uuid:
                    response_text += f"\n💬 *Chat UUID: {chat_uuid}* (to continue conversation)"
                
                return [TextContent(type="text", text=response_text)]
            else:
                error_msg = f"HTTP Error {response.status_code}: {response.text}"
                logger.error(error_msg)
                return [TextContent(type="text", text=f"Error querying Biel.ai: {error_msg}")]
                
    except httpx.TimeoutException:
        return [TextContent(type="text", text="⏱️ Timeout: Biel.ai took too long to respond")]
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return [TextContent(type="text", text=f"Unexpected error: {str(e)}")]

async def main():
    """Main function"""
    logger.info("🚀 Starting Biel.ai MCP server...")
    logger.info(f"📁 Project Slug: {PROJECT_SLUG}")
    logger.info(f"🌐 Base URL: {BASE_URL}")
    logger.info(f"🔑 API Key: {'✅ Configured' if API_KEY else '❌ Not configured'}")
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="biel-ai",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={}
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())