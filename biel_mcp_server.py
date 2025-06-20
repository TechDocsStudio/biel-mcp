"""
MCP Server for Biel.ai
Remote MCP server accessible via HTTP
Allows querying your AI from editors like Cursor via MCP over HTTP
"""

import asyncio
import json
import logging
from typing import Dict, Optional, Any

import httpx
import uvicorn
from fastapi import FastAPI, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette import EventSourceResponse

# Constants
SERVER_VERSION = "0.2.0"
SERVER_NAME = "biel-ai-mcp"
DEFAULT_PORT = 7832
DEFAULT_BASE_URL = "https://app.biel.ai"
BIEL_API_PATH = "/api/v1/chats"
MCP_PROTOCOL_VERSION = "2024-11-05"
REQUEST_TIMEOUT = 30.0
KEEPALIVE_INTERVAL = 30

# Error codes
JSON_PARSE_ERROR = -32700
UNKNOWN_METHOD_ERROR = -1

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("biel-mcp")

# Tool definitions
TOOLS = [
    {
        "name": "biel_ai",
        "description": "Query Biel.ai's specialized AI about code, SDKs and documentation",
        "inputSchema": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Your question about code, SDK or documentation"
                },
                "base_url": {
                    "type": "string",
                    "description": "Base URL of your Biel.ai instance",
                    "default": DEFAULT_BASE_URL
                },
                "project_slug": {
                    "type": "string",
                    "description": "Project slug for your Biel.ai project"
                },
                "api_key": {
                    "type": "string",
                    "description": "API key for authentication (optional)",
                    "default": ""
                },
                "chat_uuid": {
                    "type": "string",
                    "description": "Chat UUID to continue conversation (optional)",
                    "default": ""
                },
                "domain": {
                    "type": "string",
                    "description": "Domain URL to pass to Biel.ai as context (optional)",
                    "default": "app.biel.ai"
                }
            },
            "required": ["message"]
        }
    }
]

# FastAPI app setup
app = FastAPI(title="Biel.ai MCP Server", version=SERVER_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def create_error_response(message: str) -> Dict[str, str]:
    """Create a standardized error response."""
    return {"type": "text", "text": f"Error: {message}"}


def create_success_response(text: str) -> Dict[str, str]:
    """Create a standardized success response."""
    return {"type": "text", "text": text}


def validate_biel_request(arguments: Dict[str, Any]) -> Optional[str]:
    """Validate Biel.ai request arguments. Returns error message if invalid, None if valid."""
    if not arguments.get("message", "").strip():
        return "Message cannot be empty"
    
    if not arguments.get("project_slug", "").strip():
        return "Project slug is required"
    
    return None


def format_biel_response(data: Dict[str, Any]) -> str:
    """Format the response from Biel.ai API into a readable string."""
    ai_message = data.get("ai_message", {})
    ai_response = ai_message.get("message", "No response received")
    chat_uuid = data.get("chat_uuid", "")
    sources = ai_message.get("sources", [])
    
    response_parts = [f"🤖 **Biel.ai responds:**\n\n{ai_response}"]
    
    if sources:
        response_parts.append("\n\n📚 **Sources consulted:**")
        for source in sources:
            response_parts.append(f"• [{source['title']}]({source['url']})")
    
    if chat_uuid:
        response_parts.append(f"\n💬 *Chat UUID: {chat_uuid}* (to continue conversation)")
    
    return "\n".join(response_parts)


async def query_biel_ai(arguments: Dict[str, Any], defaults: Dict[str, str] = None) -> Dict[str, str]:
    """Query Biel.ai API with the provided arguments."""
    # Apply defaults from connection if not provided in arguments
    if defaults:
        if not arguments.get("project_slug") and defaults.get("project_slug"):
            arguments["project_slug"] = defaults["project_slug"]
        
        if not arguments.get("api_key") and defaults.get("api_key"):
            arguments["api_key"] = defaults["api_key"]
        
        if not arguments.get("base_url") and defaults.get("base_url"):
            arguments["base_url"] = defaults["base_url"]
        
        if not arguments.get("domain") and defaults.get("domain"):
            arguments["domain"] = defaults["domain"]
    
    # Validate input
    validation_error = validate_biel_request(arguments)
    if validation_error:
        return create_error_response(validation_error)
    
    # Extract arguments
    message = arguments["message"]
    base_url = arguments.get("base_url", DEFAULT_BASE_URL)
    project_slug = arguments["project_slug"]
    api_key = arguments.get("api_key", "")
    chat_uuid = arguments.get("chat_uuid", "")
    domain = arguments.get("domain", "")
    
    # Prepare request
    payload = {
        "message": message,
        "project_slug": project_slug,
        "url": domain if domain else base_url
    }
    
    if chat_uuid:
        payload["chat_uuid"] = chat_uuid
    
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Api-Key {api_key}"
    
    full_url = f"{base_url.rstrip('/')}{BIEL_API_PATH}/"
    
    logger.info(f"Querying Biel.ai: {message[:50]}... (project: {project_slug})")
    
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            response = await client.post(full_url, json=payload, headers=headers)
            
            if response.status_code in (200, 201):
                data = response.json()
                formatted_response = format_biel_response(data)
                return create_success_response(formatted_response)
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                logger.error(f"Biel.ai API error: {error_msg}")
                return create_error_response(f"Biel.ai API error: {error_msg}")
                
    except httpx.TimeoutException:
        logger.error("Timeout querying Biel.ai")
        return create_error_response("⏱️ Timeout: Biel.ai took too long to respond")
    except Exception as e:
        logger.error(f"Unexpected error querying Biel.ai: {e}")
        return create_error_response(f"Unexpected error: {str(e)}")


def create_mcp_response(msg_id: Optional[str], result: Optional[Dict] = None, 
                       error: Optional[Dict] = None) -> Dict[str, Any]:
    """Create a standardized MCP JSON-RPC response."""
    response = {
        "jsonrpc": "2.0",
        "id": msg_id
    }
    
    if error:
        response["error"] = error
    else:
        response["result"] = result or {}
    
    return response


async def handle_mcp_request(data: Dict[str, Any], defaults: Dict[str, str] = None) -> Dict[str, Any]:
    """Handle MCP protocol messages."""
    try:
        method = data.get("method")
        msg_id = data.get("id")
        
        logger.info(f"Handling MCP request: {method}")
        
        if method == "initialize":
            return create_mcp_response(msg_id, {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": SERVER_NAME,
                    "version": SERVER_VERSION
                }
            })
        
        elif method == "tools/list":
            return create_mcp_response(msg_id, {"tools": TOOLS})
        
        elif method == "tools/call":
            params = data.get("params", {})
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            
            if tool_name == "biel_ai":
                result = await query_biel_ai(arguments, defaults)
                return create_mcp_response(msg_id, {"content": [result]})
            else:
                return create_mcp_response(
                    msg_id, 
                    error={"code": UNKNOWN_METHOD_ERROR, "message": f"Unknown tool: {tool_name}"}
                )
        
        else:
            return create_mcp_response(
                msg_id,
                error={"code": UNKNOWN_METHOD_ERROR, "message": f"Unknown method: {method}"}
            )
            
    except Exception as e:
        logger.error(f"Error handling MCP message: {e}")
        return create_mcp_response(
            data.get("id") if isinstance(data, dict) else None,
            error={"code": UNKNOWN_METHOD_ERROR, "message": str(e)}
        )


async def mcp_sse_generator(request: Request, message: Optional[str] = None, defaults: Dict[str, str] = None):
    """Generate SSE events for MCP protocol."""
    try:
        if message:
            try:
                mcp_request = json.loads(message)
                logger.info(f"Processing MCP request: {mcp_request.get('method', 'unknown')}")
                
                response = await handle_mcp_request(mcp_request, defaults)
                yield {
                    "event": "message",
                    "data": json.dumps(response)
                }
                
            except json.JSONDecodeError:
                yield {
                    "event": "message",
                    "data": json.dumps(create_mcp_response(
                        None,
                        error={"code": JSON_PARSE_ERROR, "message": "Parse error"}
                    ))
                }
        else:
            # Send initial connection event
            yield {
                "event": "message",
                "data": json.dumps({
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                    "params": {}
                })
            }
            
            # Keep connection alive
            while True:
                await asyncio.sleep(KEEPALIVE_INTERVAL)
                yield {"event": "ping", "data": ""}
            
    except asyncio.CancelledError:
        logger.info("SSE connection cancelled")
    except Exception as e:
        logger.error(f"SSE error: {e}")


# API Routes
@app.get("/")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy", 
        "service": SERVER_NAME, 
        "version": SERVER_VERSION,
        "usage": {
            "endpoint": "/sse",
            "query_params": {
                "project_slug": "Your Biel.ai project slug",
                "api_key": "Your API key (optional)",
                "base_url": "Biel.ai instance URL (optional, defaults to https://app.biel.ai)",
                "domain": "Domain URL to pass as context to Biel.ai (optional)"
            },
            "example": "/sse?project_slug=your-slug&api_key=your-key&domain=https://example.com"
        }
    }


@app.get("/sse")
async def sse_endpoint(
    request: Request, 
    message: Optional[str] = Query(None),
    project_slug: Optional[str] = Query(None),
    api_key: Optional[str] = Query(None),
    base_url: Optional[str] = Query(None),
    domain: Optional[str] = Query(None)
):
    """MCP Server-Sent Events endpoint with query parameters for configuration."""
    # Build defaults from query parameters
    defaults = {}
    if project_slug:
        defaults["project_slug"] = project_slug
    if api_key:
        defaults["api_key"] = api_key
    if base_url:
        defaults["base_url"] = base_url
    if domain:
        defaults["domain"] = domain
    
    return EventSourceResponse(mcp_sse_generator(request, message, defaults))


@app.post("/sse")
async def sse_post_endpoint(
    request: Request,
    project_slug: Optional[str] = Query(None),
    api_key: Optional[str] = Query(None),
    base_url: Optional[str] = Query(None),
    domain: Optional[str] = Query(None)
):
    """Handle POST requests to SSE endpoint with query parameters for configuration."""
    # Build defaults from query parameters
    defaults = {}
    if project_slug:
        defaults["project_slug"] = project_slug
    if api_key:
        defaults["api_key"] = api_key
    if base_url:
        defaults["base_url"] = base_url
    if domain:
        defaults["domain"] = domain
    
    try:
        data = await request.json()
        response = await handle_mcp_request(data, defaults)
        return JSONResponse(response)
    except Exception as e:
        logger.error(f"Error handling POST to SSE: {e}")
        return JSONResponse(
            create_mcp_response(None, error={"code": UNKNOWN_METHOD_ERROR, "message": str(e)}),
            status_code=500
        )


@app.options("/sse")
async def sse_options():
    """Handle OPTIONS requests for CORS preflight."""
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization"
        }
    )


if __name__ == "__main__":
    logger.info(f"🚀 Starting {SERVER_NAME} server v{SERVER_VERSION} on port {DEFAULT_PORT}")
    logger.info(f"🌐 Access via: http://localhost:{DEFAULT_PORT}/sse")
    
    uvicorn.run(app, host="0.0.0.0", port=DEFAULT_PORT)