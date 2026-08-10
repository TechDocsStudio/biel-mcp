"""
MCP Server for Biel.ai - v2 with Streamable HTTP Support
Remote MCP server accessible via HTTP with both legacy SSE (v1) and modern Streamable HTTP (v2)
Allows querying your AI from editors like Cursor via MCP over HTTP
"""

import asyncio
import json
import logging
import os
import re
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, Protocol

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
BIEL_API_PATH_TEMPLATE = "/api/v2/projects/{project_slug}/chats/"
MCP_PROTOCOL_VERSION = "2024-11-05"
MCP_PROTOCOL_VERSION_V2 = "2025-11-25"
REQUEST_TIMEOUT = 30.0
KEEPALIVE_INTERVAL = 30
SESSION_TIMEOUT = 300  # 5 minutes

# Error codes
JSON_PARSE_ERROR = -32700
UNKNOWN_METHOD_ERROR = -1

# A project slug becomes a path segment of the API URL this server calls, so it
# may not carry separators or anything else that could reshape that path.
PROJECT_SLUG_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")

# Client identity is self-reported and ends up in request headers, so it is
# capped: a header value has no length limit of its own, and an unbounded one
# would travel on every relayed message.
CLIENT_IDENTITY_MAX_LENGTH = 128

# Logging is configured by whoever hosts this app: the standalone entrypoint
# below calls basicConfig, and when mounted inside another ASGI process that
# host's configuration applies. Configuring it at import time would fight the
# host for the root logger.
logger = logging.getLogger("biel-mcp")

# Session management for Streamable HTTP
class SessionStore(Protocol):
    """The storage a ``SessionManager`` needs, and nothing more.

    A session carries the ``chat_uuid`` that threads a client's follow-up
    questions into one conversation, so every request for a session must see
    the same record. Keeping the seam this narrow is what lets that record sit
    in process memory when the server runs alone, and in a shared store when
    its host runs several instances behind a load balancer.
    """

    async def get(self, session_id: str) -> Optional[Dict[str, Any]]: ...

    async def set(self, session_id: str, session: Dict[str, Any]) -> None: ...

    async def delete(self, session_id: str) -> bool: ...

    async def touch(self, session_id: str) -> None: ...

    async def claim_chat_uuid(self, session_id: str, chat_uuid: str) -> str: ...

    async def purge_expired(self) -> None: ...


class InMemorySessionStore:
    """Process-local session storage, correct only where one process serves
    every request for a given session."""

    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}

    async def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self.sessions.get(session_id)

    async def set(self, session_id: str, session: Dict[str, Any]) -> None:
        self.sessions[session_id] = session

    async def delete(self, session_id: str) -> bool:
        return self.sessions.pop(session_id, None) is not None

    async def touch(self, session_id: str) -> None:
        session = self.sessions.get(session_id)
        if session is not None:
            session["last_active"] = datetime.now()

    async def claim_chat_uuid(self, session_id: str, chat_uuid: str) -> str:
        session = self.sessions.get(session_id)
        if session is None:
            return ""
        # Nothing is awaited between the read and the write, so the event loop
        # cannot interleave a competing claim.
        if not session["chat_uuid"]:
            session["chat_uuid"] = chat_uuid
        return session["chat_uuid"]

    async def purge_expired(self) -> None:
        """Drop sessions idle past the timeout. Stores with a native expiry
        have nothing to do here."""
        now = datetime.now()
        expired = [
            sid for sid, session in self.sessions.items()
            if (now - session["last_active"]).total_seconds() > SESSION_TIMEOUT
        ]
        for sid in expired:
            del self.sessions[sid]
            logger.info(f"Expired session {sid}")


class SessionManager:
    """Manages MCP sessions for Streamable HTTP transport."""

    def __init__(self, *, store: Optional[SessionStore] = None):
        self._store = store or InMemorySessionStore()

    async def create_session(self, project_slug: str, api_key: str = "",
                             base_url: str = DEFAULT_BASE_URL,
                             domain: str = "", metadata: str = "",
                             client_name: str = "", client_version: str = "") -> str:
        """Create a new session and return session ID."""
        session_id = str(uuid.uuid4())
        await self._store.set(session_id, {
            "id": session_id,
            "project_slug": project_slug,
            "api_key": api_key,
            "base_url": base_url,
            "domain": domain,
            "metadata": metadata,
            "chat_uuid": "",  # Store conversation ID
            # Which MCP client is connected, as reported at initialize. A
            # session recreated after expiry never sees that handshake, so
            # these stay empty and the User-Agent is all the identity left.
            "client_name": client_name,
            "client_version": client_version,
            "created_at": datetime.now(),
            "last_active": datetime.now()
        })
        logger.info(f"Created session {session_id} for project {project_slug}")
        return session_id

    async def record_client_info(self, session_id: str, client_name: str,
                                 client_version: str) -> None:
        """Attach client identity to a session that already exists.

        A client re-initializing over a live session announces itself again,
        and that handshake is the only place the identity appears. Unlike a
        tool call this writes the whole record back, which is only safe
        because initialize carries no conversation of its own: the record goes
        back with the ``chat_uuid`` it was read with, so whichever key the
        store keeps that id under still holds the claim in force.
        """
        if not client_name:
            return
        session = await self._store.get(session_id)
        if session is None:
            return
        if (session.get("client_name") == client_name
                and session.get("client_version") == client_version):
            return
        session["client_name"] = client_name
        session["client_version"] = client_version
        await self._store.set(session_id, session)

    async def record_chat_uuid(self, session_id: str, chat_uuid: str) -> str:
        """Bind the session to the conversation it threads, and return the one
        in force.

        A session threads exactly one conversation, so the first request to be
        handed a conversation id decides it and later ones adopt that answer.
        Written as a claim rather than a read-modify-write because two
        overlapping tool calls would otherwise each write back the whole
        record, and the last writer would discard the other's conversation.
        """
        return await self._store.claim_chat_uuid(session_id, chat_uuid)

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data by ID, refreshing its idle window.

        The refresh is a touch rather than a write-back of the whole record: a
        store that hands out copies would otherwise let a read in flight
        overwrite a ``chat_uuid`` another request had just committed, splitting
        the conversation this session exists to hold together.
        """
        session = await self._store.get(session_id)
        if session is None:
            return None
        await self._store.touch(session_id)
        return session

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if await self._store.delete(session_id):
            logger.info(f"Deleted session {session_id}")
            return True
        return False

    async def cleanup_expired_sessions(self) -> None:
        """Remove sessions that have been inactive for too long."""
        await self._store.purge_expired()

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

# Any origin may call the transport — MCP clients are not browsers and have no
# origin of their own. Credentials stay off: the transport authenticates by API
# key, never by cookie, and allowing them would let any page on the web read a
# credentialed response from whichever host serves this app.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sessions hang off app state so a host that scales past one process can swap
# in a shared store before serving traffic. Standalone, the default is correct.
app.state.session_manager = SessionManager()

def normalize_base_url(base_url: str) -> str:
    """Compare base URLs without tripping over a trailing slash."""
    return base_url.strip().rstrip("/")


def resolve_base_url(requested: Optional[str], allowed: frozenset,
                     default: str) -> Optional[str]:
    """The Biel API origin to call, or None when ``requested`` is not allowed.

    The origin decides where this server sends a user's message along with any
    API key that came with it, so it is the host's to choose, never the
    caller's: an unchecked value turns every request into an outbound one to an
    arbitrary address. A caller may still name an origin, but only one the host
    has already sanctioned; omitting it takes the host's own.
    """
    if not requested:
        return default
    candidate = normalize_base_url(requested)
    if candidate in {normalize_base_url(url) for url in allowed}:
        return candidate
    return None


# The Biel API this server talks to, and the origins a caller is allowed to
# name instead. A host that fronts a different instance replaces both.
app.state.api_base_url = normalize_base_url(DEFAULT_BASE_URL)
app.state.allowed_base_urls = frozenset({normalize_base_url(DEFAULT_BASE_URL)})


def create_error_response(message: str) -> Dict[str, str]:
    """Create a standardized error response."""
    return {"type": "text", "text": f"Error: {message}"}


def get_client_ip(request: Request) -> str:
    """Extract real client IP considering common proxy headers"""
    if "cf-connecting-ip" in request.headers:
        return request.headers["cf-connecting-ip"].strip()

    if "x-real-ip" in request.headers:
        return request.headers["x-real-ip"].strip()

    x_forwarded_for = request.headers.get("x-forwarded-for", "")
    if x_forwarded_for:
        return x_forwarded_for.split(",")[0].strip()

    return request.client.host if request.client else ""


def header_safe(value: Any) -> str:
    """Reduce a self-reported value to something that can be a header.

    Everything here reaches us from the client — the ``clientInfo`` it sends at
    initialize, the User-Agent it sets — and leaves again as a header on the
    relayed API call. A newline in one of those would let the client append
    headers of its own, and httpx refuses non-latin-1 bytes outright, which
    would surface as a failed query rather than a missing label.
    """
    if not isinstance(value, str):
        return ""
    collapsed = " ".join(value.split())
    encodable = collapsed.encode("latin-1", "ignore").decode("latin-1")
    return encodable[:CLIENT_IDENTITY_MAX_LENGTH]


def get_user_agent(request: Request) -> str:
    """The client's User-Agent — identity for transports with no handshake."""
    return header_safe(request.headers.get("user-agent", ""))


def extract_client_info(data: Dict[str, Any]) -> tuple[str, str]:
    """Read the (name, version) an MCP client reports in its initialize call.

    Every MCP client sends ``clientInfo`` as part of the handshake, which is
    what tells Claude Code apart from Copilot, Codex or Cursor downstream.
    """
    if not isinstance(data, dict):
        return "", ""
    params = data.get("params")
    client_info = params.get("clientInfo") if isinstance(params, dict) else None
    if not isinstance(client_info, dict):
        return "", ""
    return header_safe(client_info.get("name")), header_safe(client_info.get("version"))


def create_success_response(text: str) -> Dict[str, str]:
    """Create a standardized success response."""
    return {"type": "text", "text": text}


def validate_biel_request(arguments: Dict[str, Any]) -> Optional[str]:
    """Validate Biel.ai request arguments. Returns error message if invalid, None if valid."""
    if not arguments.get("message", "").strip():
        return "Message cannot be empty"

    project_slug = arguments.get("project_slug", "").strip()
    if not project_slug:
        return "Project slug is required"

    if not PROJECT_SLUG_PATTERN.match(project_slug):
        return "Invalid project slug"

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

    # Extract arguments. The API origin comes only from the host-vetted
    # defaults — a caller-supplied one would point this request anywhere.
    message = arguments["message"]
    base_url = (defaults or {}).get("base_url") or DEFAULT_BASE_URL
    project_slug = arguments["project_slug"]
    api_key = arguments.get("api_key", "")
    chat_uuid = arguments.get("chat_uuid", "")
    domain = arguments.get("domain", "")
    metadata = arguments.get("metadata", "")
    client_ip = (defaults or {}).get("client_ip", "")
    client_name = (defaults or {}).get("client_name", "")
    client_version = (defaults or {}).get("client_version", "")
    user_agent = (defaults or {}).get("user_agent", "")

    # Prepare request
    payload = {
        "message": message,
        "url": domain if domain else base_url,
        "metadata": metadata
    }

    if chat_uuid:
        payload["chat_uuid"] = chat_uuid

    headers = {
        "Content-Type": "application/json",
        "X-Biel-Source": "mcp",  # Identifies requests coming from the MCP server
    }
    if api_key:
        headers["Authorization"] = f"Api-Key {api_key}"
    # Forward the real client IP so the backend can store it for analytics
    if client_ip:
        headers["X-Client-IP"] = client_ip
    # Forward which MCP client asked, so a query can be attributed to Claude
    # Code, Copilot, Codex, Cursor and the rest. The handshake identity is the
    # reliable one; the User-Agent goes too because transports without a
    # session (v1 SSE) and sessions recreated after expiry have nothing else.
    if client_name:
        headers["X-MCP-Client-Name"] = client_name
    if client_version:
        headers["X-MCP-Client-Version"] = client_version
    if user_agent:
        headers["X-MCP-Client-UA"] = user_agent

    full_url = f"{base_url.rstrip('/')}{BIEL_API_PATH_TEMPLATE.format(project_slug=project_slug)}"

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
                            session_id: Optional[str] = None,
                            sessions: Optional[SessionManager] = None) -> Dict[str, Any]:
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

            client_name, client_version = extract_client_info(data)
            if client_name:
                logger.info(
                    f"MCP client identified: {client_name} v{client_version} "
                    f"(session: {session_id})"
                )

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

                # Only V2 requests carry a session to bind the conversation to.
                if session_id and new_chat_uuid and sessions:
                    in_force = await sessions.record_chat_uuid(session_id, new_chat_uuid)
                    logger.info(f"Session {session_id} threads chat_uuid: {in_force}")

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
    # Capture the real client IP and User-Agent to forward to the Biel.ai
    # backend for analytics. This transport keeps no session, so the
    # handshake's clientInfo cannot outlive the request that carried it.
    client_ip = get_client_ip(request)
    user_agent = get_user_agent(request)

    resolved_base_url = resolve_base_url(
        base_url, request.app.state.allowed_base_urls, request.app.state.api_base_url
    )
    if resolved_base_url is None:
        logger.warning(f"Rejected base_url {base_url!r} (project: {project_slug})")
        return JSONResponse(
            {"error": "base_url is not an allowed Biel.ai instance"},
            status_code=400
        )

    # Build defaults from query parameters
    defaults = {
        "client_ip": client_ip,
        "user_agent": user_agent,
        "base_url": resolved_base_url,
    }
    if project_slug:
        defaults["project_slug"] = project_slug
    if api_key:
        defaults["api_key"] = api_key
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
    # Capture the real client IP and User-Agent to forward to the Biel.ai
    # backend for analytics. This transport keeps no session, so the
    # handshake's clientInfo cannot outlive the request that carried it.
    client_ip = get_client_ip(request)
    user_agent = get_user_agent(request)

    resolved_base_url = resolve_base_url(
        base_url, request.app.state.allowed_base_urls, request.app.state.api_base_url
    )
    if resolved_base_url is None:
        logger.warning(f"Rejected base_url {base_url!r} (project: {project_slug})")
        return JSONResponse(
            {"error": "base_url is not an allowed Biel.ai instance"},
            status_code=400
        )

    # Build defaults from query parameters
    defaults = {
        "client_ip": client_ip,
        "user_agent": user_agent,
        "base_url": resolved_base_url,
    }
    if project_slug:
        defaults["project_slug"] = project_slug
    if api_key:
        defaults["api_key"] = api_key
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
#
# Two path shapes reach the same handlers. ``/mcp/{project_slug}`` is the
# canonical one, served under the Django app's own domain; ``/v2/{slug}/mcp``
# is the shape the standalone mcp.biel.ai deployment published and stays live
# for clients whose config still points at it.
# ============================================================================

@app.post("/mcp/{project_slug}")
@app.get("/mcp/{project_slug}")
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

    sessions: SessionManager = request.app.state.session_manager

    # Cleanup expired sessions periodically
    await sessions.cleanup_expired_sessions()

    # Capture the real client IP and User-Agent for analytics
    client_ip = get_client_ip(request)
    user_agent = get_user_agent(request)

    resolved_base_url = resolve_base_url(
        base_url, request.app.state.allowed_base_urls, request.app.state.api_base_url
    )
    if resolved_base_url is None:
        logger.warning(f"Rejected base_url {base_url!r} (project: {project_slug})")
        return JSONResponse(
            {"error": "base_url is not an allowed Biel.ai instance"},
            status_code=400
        )

    # Build defaults from URL path and query parameters
    defaults = {
        "client_ip": client_ip,
        "user_agent": user_agent,
        "project_slug": project_slug,
        "api_key": api_key or "",
        "base_url": resolved_base_url,
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
                # The handshake is the one message carrying the client's name,
                # so the session takes it now or never learns it.
                client_name, client_version = extract_client_info(data)

                # Get or create session
                if mcp_session_id:
                    session = await sessions.get_session(mcp_session_id)
                    if not session:
                        return JSONResponse(
                            create_mcp_response(
                                data.get("id"),
                                error={"code": -32000, "message": "Invalid session ID"}
                            ),
                            status_code=400
                        )
                    await sessions.record_client_info(
                        mcp_session_id, client_name, client_version
                    )
                else:
                    # Create new session
                    session_id = await sessions.create_session(
                        project_slug=project_slug,
                        api_key=api_key or "",
                        base_url=resolved_base_url,
                        domain=domain or "",
                        metadata=metadata or "",
                        client_name=client_name,
                        client_version=client_version
                    )
                    mcp_session_id = session_id

                # Handle the initialize request
                mcp_response = await handle_mcp_request(
                    data, defaults, mcp_session_id, sessions
                )

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

            # Validate session - auto-recreate if expired
            session = await sessions.get_session(mcp_session_id)
            session_recreated = False
            if not session:
                if not project_slug:
                    return JSONResponse(
                        create_mcp_response(
                            data.get("id"),
                            error={"code": -32000, "message": "Invalid session ID"}
                        ),
                        status_code=400
                    )
                logger.warning(
                    f"Session {mcp_session_id[:8]}... expired or not found, recreating for project {project_slug}"
                )
                mcp_session_id = await sessions.create_session(
                    project_slug=project_slug,
                    api_key=api_key or "",
                    base_url=resolved_base_url,
                    domain=domain or "",
                    metadata=metadata or ""
                )
                session = await sessions.get_session(mcp_session_id)
                if not session:
                    # A shared store can lose the record between the write and
                    # this read (eviction, failover). Say so, rather than
                    # dereferencing None into an opaque 500.
                    return JSONResponse(
                        create_mcp_response(
                            data.get("id"),
                            error={
                                "code": -32000,
                                "message": "Session storage unavailable, retry"
                            }
                        ),
                        status_code=503
                    )
                session_recreated = True

            # Use session defaults
            session_defaults = {
                "client_ip": client_ip,
                "user_agent": user_agent,
                "project_slug": session["project_slug"],
                "api_key": session["api_key"],
                "base_url": session["base_url"],
                "domain": session["domain"],
                "metadata": session["metadata"],
                "chat_uuid": session.get("chat_uuid", ""),  # Pass current chat_uuid to maintain context
                # Recorded at initialize; absent on a session recreated after
                # expiry, where the User-Agent carries what identity is left.
                "client_name": session.get("client_name", ""),
                "client_version": session.get("client_version", "")
            }

            # Handle the request
            mcp_response = await handle_mcp_request(
                data, session_defaults, mcp_session_id, sessions
            )
            response_headers = {"MCP-Session-Id": mcp_session_id} if session_recreated else None
            return JSONResponse(mcp_response, headers=response_headers)

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
            session = await sessions.get_session(mcp_session_id)
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
                        "retry": KEEPALIVE_INTERVAL * 1000  # retry in milliseconds
                    }
                    event_counter += 1
            except asyncio.CancelledError:
                logger.info(f"SSE stream cancelled for session {mcp_session_id}")
            except Exception as e:
                logger.error(f"SSE stream error: {e}")

        return EventSourceResponse(sse_stream())


@app.delete("/mcp/{project_slug}")
@app.delete("/v2/{project_slug}/mcp")
async def delete_session_v2(
    project_slug: str,
    request: Request,
    mcp_session_id: Optional[str] = Header(None, alias="MCP-Session-Id")
):
    """V2: Delete/terminate a session."""
    logger.info(f"V2 DELETE endpoint accessed (project: {project_slug})")

    if not mcp_session_id:
        return JSONResponse(
            {"error": "MCP-Session-Id header required"},
            status_code=400
        )

    sessions: SessionManager = request.app.state.session_manager
    if await sessions.delete_session(mcp_session_id):
        return JSONResponse({"status": "session terminated"})
    else:
        return JSONResponse(
            {"error": "Session not found"},
            status_code=404
        )


@app.options("/mcp/{project_slug}")
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
        }
    }


@app.get("/health")
async def health():
    """Simple health check."""
    return {"status": "ok", "version": SERVER_VERSION}


def main() -> None:
    """Run the standalone MCP server."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    # A self-hosted Biel instance is named here, by the operator, rather than
    # by whoever calls the server.
    app.state.api_base_url = normalize_base_url(
        os.environ.get("BIEL_BASE_URL", DEFAULT_BASE_URL)
    )
    app.state.allowed_base_urls = frozenset(
        normalize_base_url(url)
        for url in os.environ.get(
            "BIEL_ALLOWED_BASE_URLS", app.state.api_base_url
        ).split(",")
        if url.strip()
    )
    logger.info(f"🚀 Starting {SERVER_NAME} server v{SERVER_VERSION} on port {DEFAULT_PORT}")
    logger.info(f"🌐 V1 (SSE): http://localhost:{DEFAULT_PORT}/sse")
    logger.info(f"🌐 V2 (Streamable HTTP): http://localhost:{DEFAULT_PORT}/v2/{{project_slug}}/mcp")

    uvicorn.run(app, host="0.0.0.0", port=DEFAULT_PORT)


if __name__ == "__main__":
    main()
