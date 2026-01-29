"""
MCP Server for Biel.ai - v2 with Streamable HTTP Support
Remote MCP server accessible via HTTP with both legacy SSE (v1) and modern Streamable HTTP (v2)
Allows querying your AI from editors like Cursor via MCP over HTTP
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

import httpx
import uvicorn
from fastapi import FastAPI, Header, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette import EventSourceResponse

# Constants
SERVER_VERSION = "2.0.0"
SERVER_NAME = "biel-ai-mcp"
DEFAULT_PORT = 7832
DEFAULT_BASE_URL = "https://app.biel.ai"
BIEL_API_PATH = "/api/v1/chats"
MCP_PROTOCOL_VERSION = "2024-11-05"
MCP_PROTOCOL_VERSION_V2 = "2025-11-25"
REQUEST_TIMEOUT = 30.0
KEEPALIVE_INTERVAL = 30
SESSION_TIMEOUT = 300  # 5 minutes

# Error codes
JSON_PARSE_ERROR = -32700
UNKNOWN_METHOD_ERROR = -1

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("biel-mcp")

# Session management for Streamable HTTP
class SessionManager:
    """Manages MCP sessions for Streamable HTTP transport."""
    
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
    
    def create_session(self, project_slug: str, api_key: str = "", 
                      base_url: str = DEFAULT_BASE_URL, 
                      domain: str = "", metadata: str = "") -> str:
        """Create a new session and return session ID."""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "id": session_id,
            "project_slug": project_slug,
            "api_key": api_key,
            "base_url": base_url,
            "domain": domain,
            "metadata": metadata,
            "chat_uuid": "",  # Store conversation ID
            "created_at": datetime.now(),
            "last_active": datetime.now()
        }
        logger.info(f"Created session {session_id} for project {project_slug}")
        return session_id
    
    def update_session(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """Update session data."""
        if session_id in self.sessions:
            self.sessions[session_id].update(updates)
            self.sessions[session_id]["last_active"] = datetime.now()
            return True
        return False

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data by ID."""
        session = self.sessions.get(session_id)
        if session:
            # Update last active time
            session["last_active"] = datetime.now()
            return session
        return None
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Deleted session {session_id}")
            return True
        return False
    
    def cleanup_expired_sessions(self):
        """Remove sessions that have been inactive for too long."""
        now = datetime.now()
        expired = [
            sid for sid, session in self.sessions.items()
            if (now - session["last_active"]).seconds > SESSION_TIMEOUT
        ]
        for sid in expired:
            self.delete_session(sid)
            logger.info(f"Expired session {sid}")

# Global session manager
session_manager = SessionManager()

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
                    "description": "Domain URL. Required only if 'Allowed domains' is enabled in project settings.",
                    "default": ""
                },
                "metadata": {
                    "type": "string",
                    "description": "Metadata to tag the conversation source (optional)",
                    "default": ""
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


async def query_biel_ai(arguments: Dict[str, Any], defaults: Dict[str, str] = None) -> tuple[Dict[str, str], Optional[str]]:
    """
    Query Biel.ai API with the provided arguments.
    Returns a tuple of (response_dict, new_chat_uuid).
    """
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
        
        if not arguments.get("metadata") and defaults.get("metadata"):
            arguments["metadata"] = defaults["metadata"]

        if not arguments.get("chat_uuid") and defaults.get("chat_uuid"):
            arguments["chat_uuid"] = defaults["chat_uuid"]
    
    # Validate input
    validation_error = validate_biel_request(arguments)
    if validation_error:
        return create_error_response(validation_error), None
    
    # Extract arguments
    message = arguments["message"]
    base_url = arguments.get("base_url", DEFAULT_BASE_URL)
    project_slug = arguments["project_slug"]
    api_key = arguments.get("api_key", "")
    chat_uuid = arguments.get("chat_uuid", "")
    domain = arguments.get("domain", "")
    metadata = arguments.get("metadata", "")
    
    # Prepare request
    payload = {
        "message": message,
        "project_slug": project_slug,
        "url": domain if domain else base_url,
        "metadata": metadata
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
                # Return response and the chat_uuid from the server to update session
                return create_success_response(formatted_response), data.get("chat_uuid")
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                logger.error(f"Biel.ai API error: {error_msg}")
                return create_error_response(f"Biel.ai API error: {error_msg}"), None
                
    except httpx.TimeoutException:
        logger.error("Timeout querying Biel.ai")
        return create_error_response("⏱️ Timeout: Biel.ai took too long to respond"), None
    except Exception as e:
        logger.error(f"Unexpected error querying Biel.ai: {e}")
        return create_error_response(f"Unexpected error: {str(e)}"), None


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


async def handle_mcp_request(data: Dict[str, Any], defaults: Dict[str, str] = None, 
                            session_id: Optional[str] = None) -> Dict[str, Any]:
    """Handle MCP protocol messages."""
    try:
        method = data.get("method")
        msg_id = data.get("id")
        
        logger.info(f"Handling MCP request: {method} (session: {session_id})")
        
        if method == "initialize":
            # For v2 (Streamable HTTP), we might need to create a session
            # The session_id will be returned in the Mcp-Session-Id header
            # Use V1 version for SSE (no session_id) and V2 for Streamable HTTP
            protocol_version = MCP_PROTOCOL_VERSION_V2 if session_id else MCP_PROTOCOL_VERSION
            
            result = {
                "protocolVersion": protocol_version,
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": SERVER_NAME,
                    "version": SERVER_VERSION
                }
            }
            return create_mcp_response(msg_id, result)
        
        elif method == "tools/list":
            return create_mcp_response(msg_id, {"tools": TOOLS})
        
        elif method == "tools/call":
            params = data.get("params", {})
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            
            if tool_name == "biel_ai":
                result, new_chat_uuid = await query_biel_ai(arguments, defaults)
                
                # Update session with new chat_uuid if available (only for V2 with session)
                if session_id and new_chat_uuid:
                    session_manager.update_session(session_id, {"chat_uuid": new_chat_uuid})
                    # Also log it for debugging
                    logger.info(f"Updated session {session_id} with chat_uuid: {new_chat_uuid}")
                
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
    """Generate SSE events for MCP protocol (legacy v1)."""
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


# ============================================================================
# V1 API Routes (Legacy SSE) - Backwards Compatible
# ============================================================================

@app.get("/sse")
async def sse_endpoint_v1(
    request: Request, 
    message: Optional[str] = Query(None),
    project_slug: Optional[str] = Query(None),
    api_key: Optional[str] = Query(None),
    base_url: Optional[str] = Query(None),
    domain: Optional[str] = Query(None),
    metadata: Optional[str] = Query(None)
):
    """V1: MCP Server-Sent Events endpoint with query parameters for configuration."""
    logger.info("V1 SSE endpoint accessed")
    
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
    if metadata:
        defaults["metadata"] = metadata
    
    return EventSourceResponse(mcp_sse_generator(request, message, defaults))


@app.post("/sse")
async def sse_post_endpoint_v1(
    request: Request,
    project_slug: Optional[str] = Query(None),
    api_key: Optional[str] = Query(None),
    base_url: Optional[str] = Query(None),
    domain: Optional[str] = Query(None),
    metadata: Optional[str] = Query(None)
):
    """V1: Handle POST requests to SSE endpoint with query parameters for configuration."""
    logger.info("V1 SSE POST endpoint accessed")
    
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
    if metadata:
        defaults["metadata"] = metadata
    
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
async def sse_options_v1():
    """V1: Handle OPTIONS requests for CORS preflight."""
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization"
        }
    )


# ============================================================================
# V2 API Routes (Streamable HTTP) - Modern Standard
# ============================================================================

@app.post("/v2/{project_slug}/mcp")
@app.get("/v2/{project_slug}/mcp")
async def streamable_http_endpoint_v2(
    project_slug: str,
    request: Request,
    response: Response,
    mcp_session_id: Optional[str] = Header(None, alias="MCP-Session-Id"),
    mcp_protocol_version: Optional[str] = Header(None, alias="MCP-Protocol-Version"),
    api_key: Optional[str] = Query(None),
    base_url: Optional[str] = Query(None),
    domain: Optional[str] = Query(None),
    metadata: Optional[str] = Query(None)
):
    """
    V2: Streamable HTTP endpoint following MCP 2025-11-25 specification.
    
    Supports both POST (for requests) and GET (for SSE streaming).
    Uses project_slug from URL path and session management via MCP-Session-Id header.
    """
    logger.info(f"V2 Streamable HTTP endpoint accessed: {request.method} (project: {project_slug})")
    
    # Validate protocol version header (required for all requests except OPTIONS)
    if request.method != "OPTIONS":
        if not mcp_protocol_version:
            # For backwards compatibility, assume 2025-03-26 if not provided
            mcp_protocol_version = "2025-03-26"
            logger.warning(f"No MCP-Protocol-Version header provided, assuming {mcp_protocol_version}")
        elif mcp_protocol_version not in [MCP_PROTOCOL_VERSION, MCP_PROTOCOL_VERSION_V2, "2025-03-26"]:
            return JSONResponse(
                {"error": f"Unsupported MCP-Protocol-Version: {mcp_protocol_version}"},
                status_code=400
            )
    
    # Validate Origin header to prevent DNS rebinding attacks
    origin = request.headers.get("Origin")
    if origin:
        logger.info(f"Request from origin: {origin}")
    
    # Cleanup expired sessions periodically
    session_manager.cleanup_expired_sessions()
    
    # Build defaults from URL path and query parameters
    defaults = {
        "project_slug": project_slug,
        "api_key": api_key or "",
        "base_url": base_url or DEFAULT_BASE_URL,
        "domain": domain or "",
        "metadata": metadata or ""
    }
    
    # Handle POST requests (client-to-server messages)
    if request.method == "POST":
        try:
            data = await request.json()
            method = data.get("method")
            
            # Handle initialization specially
            if method == "initialize":
                # Get or create session
                if mcp_session_id:
                    session = session_manager.get_session(mcp_session_id)
                    if not session:
                        return JSONResponse(
                            create_mcp_response(
                                data.get("id"),
                                error={"code": -32000, "message": "Invalid session ID"}
                            ),
                            status_code=400
                        )
                else:
                    # Create new session
                    session_id = session_manager.create_session(
                        project_slug=project_slug,
                        api_key=api_key or "",
                        base_url=base_url or DEFAULT_BASE_URL,
                        domain=domain or "",
                        metadata=metadata or ""
                    )
                    mcp_session_id = session_id
                
                # Handle the initialize request
                mcp_response = await handle_mcp_request(data, defaults, mcp_session_id)
                
                # Return response with session ID in header (capital letters as per spec)
                return JSONResponse(
                    content=mcp_response,
                    headers={"MCP-Session-Id": mcp_session_id}
                )
            
            # For other requests, session is required
            if not mcp_session_id:
                return JSONResponse(
                    create_mcp_response(
                        data.get("id"),
                        error={"code": -32000, "message": "MCP-Session-Id header required"}
                    ),
                    status_code=400
                )
            
            # Validate session
            session = session_manager.get_session(mcp_session_id)
            if not session:
                return JSONResponse(
                    create_mcp_response(
                        data.get("id"),
                        error={"code": -32000, "message": "Invalid session ID"}
                    ),
                    status_code=400
                )
            
            # Use session defaults
            session_defaults = {
                "project_slug": session["project_slug"],
                "api_key": session["api_key"],
                "base_url": session["base_url"],
                "domain": session["domain"],
                "metadata": session["metadata"],
                "chat_uuid": session.get("chat_uuid", "")  # Pass current chat_uuid to maintain context
            }
            
            # Handle the request
            mcp_response = await handle_mcp_request(data, session_defaults, mcp_session_id)
            return JSONResponse(mcp_response)
            
        except json.JSONDecodeError:
            return JSONResponse(
                create_mcp_response(
                    None,
                    error={"code": JSON_PARSE_ERROR, "message": "Invalid JSON"}
                ),
                status_code=400
            )
        except Exception as e:
            logger.error(f"Error handling V2 POST: {e}")
            return JSONResponse(
                create_mcp_response(
                    None,
                    error={"code": -32000, "message": str(e)}
                ),
                status_code=500
            )
    
    # Handle GET requests (SSE streaming for server-to-client messages)
    elif request.method == "GET":
        # Check for Last-Event-ID header for resumability
        last_event_id = request.headers.get("Last-Event-ID")
        
        if last_event_id:
            logger.info(f"Resuming stream from Last-Event-ID: {last_event_id}")
        
        # Session is required for GET (unless resuming with Last-Event-ID)
        if not mcp_session_id and not last_event_id:
            return JSONResponse(
                {"error": "MCP-Session-Id header required"},
                status_code=400
            )
        
        # Validate session if provided
        if mcp_session_id:
            session = session_manager.get_session(mcp_session_id)
            if not session:
                return JSONResponse(
                    {"error": "Invalid session ID"},
                    status_code=404  # 404 as per spec when session not found
                )
        
        # Return SSE stream for server-initiated messages
        async def sse_stream():
            try:
                # Send initial event with ID (for resumability)
                event_counter = 0
                session_prefix = mcp_session_id[:8] if mcp_session_id else "default"
                
                # Send priming event with empty data
                yield {
                    "id": f"{session_prefix}-{event_counter}",
                    "event": "message",
                    "data": ""
                }
                event_counter += 1
                
                # Keep connection alive with periodic pings
                while True:
                    await asyncio.sleep(KEEPALIVE_INTERVAL)
                    yield {
                        "id": f"{session_prefix}-{event_counter}",
                        "event": "ping",
                        "data": "",
                        "retry": str(KEEPALIVE_INTERVAL * 1000)  # retry in milliseconds
                    }
                    event_counter += 1
            except asyncio.CancelledError:
                logger.info(f"SSE stream cancelled for session {mcp_session_id}")
            except Exception as e:
                logger.error(f"SSE stream error: {e}")
        
        return EventSourceResponse(sse_stream())


@app.delete("/v2/{project_slug}/mcp")
async def delete_session_v2(
    project_slug: str,
    mcp_session_id: Optional[str] = Header(None, alias="MCP-Session-Id")
):
    """V2: Delete/terminate a session."""
    logger.info(f"V2 DELETE endpoint accessed (project: {project_slug})")
    
    if not mcp_session_id:
        return JSONResponse(
            {"error": "MCP-Session-Id header required"},
            status_code=400
        )
    
    if session_manager.delete_session(mcp_session_id):
        return JSONResponse({"status": "session terminated"})
    else:
        return JSONResponse(
            {"error": "Session not found"},
            status_code=404
        )


@app.options("/v2/{project_slug}/mcp")
async def streamable_http_options_v2(project_slug: str):
    """V2: Handle OPTIONS requests for CORS preflight."""
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization, MCP-Session-Id, MCP-Protocol-Version, Origin, Last-Event-ID"
        }
    )


# ============================================================================
# Health & Info Routes
# ============================================================================

@app.get("/")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy", 
        "service": SERVER_NAME, 
        "version": SERVER_VERSION,
        "transports": {
            "v1": {
                "type": "SSE (legacy)",
                "status": "supported",
                "endpoint": "/sse",
                "protocol_version": MCP_PROTOCOL_VERSION
            },
            "v2": {
                "type": "Streamable HTTP",
                "status": "recommended",
                "endpoint": "/v2/{PROJECT_SLUG}/mcp",
                "protocol_version": MCP_PROTOCOL_VERSION_V2
            }
        },
        "usage": {
            "v1_sse": {
                "endpoint": "/sse",
                "query_params": {
                    "project_slug": "Your Biel.ai project slug",
                    "api_key": "Your API key (optional)",
                    "base_url": "Biel.ai instance URL (optional)",
                    "domain": "Domain URL. Required only if 'Allowed domains' is enabled in project settings.",
                    "metadata": "Metadata to tag conversation (optional)"
                },
                "example": "/sse?project_slug=your-slug&api_key=your-key"
            },
            "v2_streamable_http": {
                "endpoint": "/v2/{project_slug}/mcp",
                "headers": {
                    "MCP-Session-Id": "Session ID (returned on initialize)",
                    "MCP-Protocol-Version": "Protocol version (e.g., 2025-11-25)"
                },
                "query_params": {
                    "api_key": "Your API key (optional)",
                    "base_url": "Biel.ai instance URL (optional)",
                    "domain": "Domain URL. Required only if 'Allowed domains' is enabled in project settings.",
                    "metadata": "Metadata to tag conversation (optional)"
                },
                "example": "/v2/your-slug/mcp?api_key=your-key",
                "config_example": {
                    "biel-ai": {
                        "url": "https://mcp.biel.ai/v2/YOUR_PROJECT_SLUG/mcp?api_key=YOUR_API_KEY"
                    }
                }
            }
        },
        "active_sessions": len(session_manager.sessions)
    }


@app.get("/health")
async def health():
    """Simple health check."""
    return {"status": "ok", "version": SERVER_VERSION}


if __name__ == "__main__":
    logger.info(f"🚀 Starting {SERVER_NAME} server v{SERVER_VERSION} on port {DEFAULT_PORT}")
    logger.info(f"🌐 V1 (SSE): http://localhost:{DEFAULT_PORT}/sse")
    logger.info(f"🌐 V2 (Streamable HTTP): http://localhost:{DEFAULT_PORT}/v2/{{project_slug}}/mcp")
    
    uvicorn.run(app, host="0.0.0.0", port=DEFAULT_PORT)