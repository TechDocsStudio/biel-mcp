"""
MCP Server for Biel.ai - v2 with Streamable HTTP Support
Remote MCP server accessible via HTTP with both legacy SSE (v1) and modern Streamable HTTP (v2)
Allows querying your AI from editors like Cursor via MCP over HTTP
"""

import asyncio
import json
import logging
import math
import os
import re
import time
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
BIEL_SEARCH_PATH_TEMPLATE = "/api/v2/projects/{project_slug}/search/"
MCP_PROTOCOL_VERSION = "2024-11-05"
MCP_PROTOCOL_VERSION_V2 = "2025-11-25"
API_READ_TIMEOUT_SECONDS = float(os.environ.get("BIEL_MCP_READ_TIMEOUT_SECONDS", "60"))
if not math.isfinite(API_READ_TIMEOUT_SECONDS) or API_READ_TIMEOUT_SECONDS <= 0:
    raise ValueError("BIEL_MCP_READ_TIMEOUT_SECONDS must be a positive finite number")
REQUEST_TIMEOUT = httpx.Timeout(
    API_READ_TIMEOUT_SECONDS, connect=5.0, write=10.0, pool=5.0
)
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

    A session carries the ``chat_uuid`` and its matching restore capability so
    every follow-up can both identify and authorize the conversation. Keeping
    the seam this narrow is what lets that record sit in process memory when
    the server runs alone, and in a shared store when its host runs several
    instances behind a load balancer.
    """

    async def get(self, session_id: str) -> Optional[Dict[str, Any]]: ...

    async def set(self, session_id: str, session: Dict[str, Any]) -> None: ...

    async def delete(self, session_id: str) -> bool: ...

    async def touch(self, session_id: str) -> None: ...

    async def claim_chat_credentials(
        self, session_id: str, chat_uuid: str, restore_token: str
    ) -> tuple[str, str]: ...

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

    async def claim_chat_credentials(
        self, session_id: str, chat_uuid: str, restore_token: str
    ) -> tuple[str, str]:
        session = self.sessions.get(session_id)
        if session is None:
            return "", ""
        # Nothing is awaited between the read and the write, so the event loop
        # cannot interleave a competing claim.
        if not session["chat_uuid"]:
            session["chat_uuid"] = chat_uuid
            session["restore_token"] = restore_token
        return session["chat_uuid"], session["restore_token"]

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
            logger.info("MCP session expired", extra={"event": "mcp.session_expired"})


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
            "restore_token": "",  # Capability authorizing that conversation
            # Which MCP client is connected, as reported at initialize. A
            # session recreated after expiry never sees that handshake, so
            # these stay empty and the User-Agent is all the identity left.
            "client_name": client_name,
            "client_version": client_version,
            "created_at": datetime.now(),
            "last_active": datetime.now()
        })
        logger.info("MCP session created", extra={"event": "mcp.session_created"})
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

    async def record_chat_credentials(
        self, session_id: str, chat_uuid: str, restore_token: str
    ) -> tuple[str, str]:
        """Bind the session to a conversation and its restore capability.

        A session threads exactly one conversation, so the first request to be
        handed a credential pair decides it and later ones adopt that answer.
        The values are claimed together because mixing one request's UUID with
        another request's token would make every continuation fail closed.
        """
        return await self._store.claim_chat_credentials(
            session_id, chat_uuid, restore_token
        )

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
            logger.info("MCP session deleted", extra={"event": "mcp.session_deleted"})
            return True
        return False

    async def cleanup_expired_sessions(self) -> None:
        """Remove sessions that have been inactive for too long."""
        await self._store.purge_expired()

# Tool definitions
TOOLS = [
    {
        "name": "biel_ai",
        "description": (
            "Search this Biel.ai project's indexed product documentation and knowledge "
            "base, or get an answer grounded in those sources. Use for product setup, "
            "configuration, API and SDK usage, integrations, troubleshooting, and "
            "locating supporting documentation. Content can include product guides, "
            "API references, help articles, uploaded documents, repository content "
            "and OpenAPI sources. Choose mode='search' to retrieve ranked text chunks "
            "and source references without Biel.ai answer generation; use concise "
            "keywords and write your own answer from the retrieved sources. Choose "
            "mode='answer' for a generated Biel.ai response with conversational context. "
            "Results are limited to this project's indexed content."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Question for answer mode, or concise search terms for search mode"
                },
                "mode": {
                    "type": "string",
                    "enum": ["answer", "search"],
                    "default": "answer",
                    "description": (
                        "search: retrieve source text chunks and references without generating an answer. "
                        "answer: generate a response using Biel.ai; can take longer. "
                        "Private projects require a key with the corresponding search or chats_create scope."
                    )
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 5,
                    "description": "Maximum source chunks returned in search mode; ignored in answer mode"
                },
                "api_key": {
                    "type": "string",
                    "description": "API key for authentication (optional)",
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


def create_error_response(message: str) -> Dict[str, Any]:
    """Create a standardized error response."""
    return {"content": [{"type": "text", "text": f"Error: {message}"}], "isError": True}


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


def create_success_response(text: str) -> Dict[str, Any]:
    """Create a standardized success response."""
    return {"content": [{"type": "text", "text": text}], "isError": False}


def validate_biel_request(arguments: Dict[str, Any]) -> Optional[str]:
    """Validate Biel.ai request arguments. Returns error message if invalid, None if valid."""
    if not isinstance(arguments.get("message"), str) or not arguments["message"].strip():
        return "Message cannot be empty"

    if arguments.get("mode", "answer") not in ("answer", "search"):
        return "Mode must be 'answer' or 'search'"
    if arguments.get("mode") == "search":
        limit = arguments.get("limit", 5)
        if type(limit) is not int or not 1 <= limit <= 20:
            return "Search limit must be an integer from 1 to 20"

    project_slug = arguments.get("project_slug", "")
    if not isinstance(project_slug, str):
        return "Project slug must be a string"
    project_slug = project_slug.strip()
    if not project_slug:
        return "Project slug is required"

    if not PROJECT_SLUG_PATTERN.match(project_slug):
        return "Invalid project slug"

    return None


def format_biel_response(data: Dict[str, Any]) -> str:
    """Format the response from Biel.ai API into a readable string."""
    ai_message = data.get("ai_message", {})
    ai_response = ai_message.get("message", "No response received")
    sources = ai_message.get("sources", [])

    response_parts = [f"🤖 **Biel.ai responds:**\n\n{ai_response}"]

    if sources:
        response_parts.append("\n\n📚 **Sources consulted:**")
        for source in sources:
            response_parts.append(f"• [{source['title']}]({source['url']})")

    return "\n".join(response_parts)


def format_search_response(data: Dict[str, Any], limit: int = 5) -> str:
    """Return ranked source excerpts for the client to synthesize its own answer."""
    results = data["results"]
    if not isinstance(results, list):
        raise ValueError("Invalid search response")
    if not results:
        return "No matching documentation found. Try different search terms."
    parts = [
        f"Documentation search results: showing {min(limit, len(results))} of {len(results)} "
        "matches (source excerpts, no generated answer)."
    ]
    for number, result in enumerate(results[:limit], 1):
        location = [result.get("page_title") or result.get("title") or "Untitled source"]
        if result.get("page"):
            location.append(f"page {result['page']}")
        if result.get("sheet"):
            location.append(f"sheet {result['sheet']}")
        reference = f"URL: {result['url']}" if result.get("url") else "Reference: " + ", ".join(location)
        parts.append(
            f"{number}. {result.get('title', '')} ({result.get('source_type', 'page')})\n"
            f"{reference}\n"
            f"{result.get('content') or result.get('fragment', '')}"
        )
    return "\n\n".join(parts)


async def query_biel_ai(
    arguments: Dict[str, Any], defaults: Dict[str, str] = None
) -> tuple[Dict[str, Any], Optional[str], Optional[str]]:
    """
    Query Biel.ai API with the provided arguments.
    Returns ``(response_dict, new_chat_uuid, new_restore_token)``.
    """
    if not isinstance(arguments, dict):
        return create_error_response("Tool arguments must be an object."), None, None
    arguments = dict(arguments)
    mode = arguments.get("mode", "answer")
    if mode not in ("answer", "search"):
        return create_error_response("Mode must be 'answer' or 'search'"), None, None
    session_project = (defaults or {}).get("project_slug")
    if session_project and arguments.get("project_slug") not in (None, "", session_project):
        return create_error_response("The project is fixed by this connection."), None, None
    requested_chat = arguments.get("chat_uuid")
    session_chat = (defaults or {}).get("chat_uuid")
    session_token = (defaults or {}).get("restore_token")
    if mode == "answer" and requested_chat and (requested_chat != session_chat or not session_token):
        logger.warning(
            "MCP continuation unavailable",
            extra={"event": "mcp.continuation_rejected", "error_kind": "session_mismatch"},
        )
        return create_error_response(
            "This conversation cannot be resumed by this connection. "
            "Reconnect to start a new conversation; do not supply a chat UUID."
        ), None, None
    if mode == "answer" and session_chat and not session_token:
        logger.warning(
            "MCP continuation unavailable",
            extra={"event": "mcp.continuation_rejected", "error_kind": "missing_capability"},
        )
        return create_error_response(
            "This connection cannot resume its conversation. Reconnect to start a new conversation."
        ), None, None
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

        # A continuation requires the UUID and its matching bearer capability.
        if (
            mode == "answer"
            and not arguments.get("chat_uuid")
            and defaults.get("chat_uuid")
            and defaults.get("restore_token")
        ):
            arguments["chat_uuid"] = defaults["chat_uuid"]

    # Validate input
    validation_error = validate_biel_request(arguments)
    if validation_error:
        return create_error_response(validation_error), None, None

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
    restore_token = (defaults or {}).get("restore_token", "")

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
    # Restore capabilities are session state, never a caller-controlled tool
    # argument. Send one only for the exact UUID it was issued alongside.
    if (
        mode == "answer"
        and restore_token
        and chat_uuid
        and chat_uuid == (defaults or {}).get("chat_uuid")
    ):
        headers["X-Chat-Restore-Token"] = restore_token
    # Forward the real client IP so the backend can store it for analytics
    if client_ip:
        headers["X-Client-IP"] = client_ip
    # Forward which MCP client asked, so a query can be attributed to Claude
    # Code, Copilot, Codex, Cursor and the rest. The handshake identity is the
    # reliable one; the User-Agent goes too because transports without a
    # session (v1 SSE) have nothing else.
    if client_name:
        headers["X-MCP-Client-Name"] = client_name
    if client_version:
        headers["X-MCP-Client-Version"] = client_version
    if user_agent:
        headers["X-MCP-Client-UA"] = user_agent

    path_template = BIEL_SEARCH_PATH_TEMPLATE if mode == "search" else BIEL_API_PATH_TEMPLATE
    full_url = f"{base_url.rstrip('/')}{path_template.format(project_slug=project_slug)}"

    started = time.monotonic()
    outcome = "upstream_error"
    upstream_status = None
    upstream_request_id = ""
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            if mode == "search":
                response = await client.get(
                    full_url, params={"q": message, "url": domain or base_url, "content_scope": "all"}, headers=headers
                )
            else:
                response = await client.post(full_url, json=payload, headers=headers)
            upstream_status = response.status_code
            upstream_request_id = header_safe(response.headers.get("x-request-id", ""))

            if response.status_code in (200, 201):
                data = response.json()
                formatted_response = (
                    format_search_response(data, arguments.get("limit", 5))
                    if mode == "search" else format_biel_response(data)
                )
                outcome = "success"
                if mode == "search":
                    return create_success_response(formatted_response), None, None
                # Persist the pair together: the token is bound to this chat.
                return (
                    create_success_response(formatted_response),
                    data.get("chat_uuid"),
                    data.get("restore_token"),
                )
            else:
                error_msg = f"Biel.ai API returned HTTP {response.status_code}."
                if response.status_code == 403:
                    error_msg += (
                        " Check project search access and quota." if mode == "search"
                        else " Check project access or reconnect to start a new conversation."
                    )
                elif response.status_code == 404:
                    error_msg += (
                        " The project is unavailable; check the connection." if mode == "search"
                        else " The project or conversation is unavailable; check the connection and reconnect."
                    )
                return (
                    create_error_response(error_msg),
                    None,
                    None,
                )

    except httpx.TimeoutException:
        outcome = "timeout"
        return (
            create_error_response("⏱️ Timeout: Biel.ai took too long to respond"),
            None,
            None,
        )
    except Exception:
        outcome = "transport_error"
        return create_error_response("Biel.ai could not complete the request. Please try again."), None, None
    finally:
        logger.log(
            logging.INFO if outcome == "success" else logging.ERROR,
            "MCP upstream request completed",
            extra={
                "event": "mcp.upstream_completed",
                "error_kind": outcome,
                "duration_ms": int((time.monotonic() - started) * 1000),
                "upstream_status": upstream_status,
                "upstream_request_id": upstream_request_id,
                "has_continuation": mode == "answer" and bool(chat_uuid),
                "mode": mode,
                "read_timeout_seconds": API_READ_TIMEOUT_SECONDS,
            },
        )


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

        logger.debug("MCP request received", extra={"event": "mcp.request_received"})

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
                result, new_chat_uuid, new_restore_token = await query_biel_ai(
                    arguments, defaults
                )

                # Only V2 requests carry a session to bind the conversation to.
                if (
                    session_id
                    and new_chat_uuid
                    and new_restore_token
                    and sessions
                ):
                    await sessions.record_chat_credentials(
                        session_id, new_chat_uuid, new_restore_token
                    )
                    logger.debug("MCP conversation stored", extra={"event": "mcp.conversation_stored"})

                return create_mcp_response(msg_id, result)
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

    except Exception:
        logger.error("MCP message handling failed", extra={"event": "mcp.message_failed"})
        return create_mcp_response(
            data.get("id") if isinstance(data, dict) else None,
            error={"code": UNKNOWN_METHOD_ERROR, "message": "Request could not be completed"}
        )


async def mcp_sse_generator(request: Request, message: Optional[str] = None, defaults: Dict[str, str] = None):
    """Generate SSE events for MCP protocol (legacy v1)."""
    try:
        if message:
            try:
                mcp_request = json.loads(message)
                logger.debug("MCP SSE message received", extra={"event": "mcp.sse_message_received"})

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
    except Exception:
        logger.error("MCP SSE failed", extra={"event": "mcp.sse_failed"})


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
        logger.warning("MCP API origin rejected", extra={"event": "mcp.origin_rejected"})
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
        logger.warning("MCP API origin rejected", extra={"event": "mcp.origin_rejected"})
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
    except Exception:
        logger.error("MCP SSE POST failed", extra={"event": "mcp.sse_post_failed"})
        return JSONResponse(
            create_mcp_response(None, error={"code": UNKNOWN_METHOD_ERROR, "message": "Request could not be completed"}),
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
    logger.debug("MCP transport accessed", extra={"event": "mcp.transport_accessed"})

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
        logger.debug("MCP Origin header supplied", extra={"event": "mcp.origin_received"})

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
        logger.warning("MCP API origin rejected", extra={"event": "mcp.origin_rejected"})
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
                    if not session or session["project_slug"] != project_slug:
                        return JSONResponse(
                            create_mcp_response(
                                data.get("id"),
                                error={"code": -32000, "message": "Invalid session ID"}
                            ),
                            status_code=404
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

            session = await sessions.get_session(mcp_session_id)
            if not session or session["project_slug"] != project_slug:
                logger.warning(
                    "MCP session unavailable",
                    extra={"event": "mcp.session_rejected", "error_kind": "session_unavailable"},
                )
                return JSONResponse(
                    create_mcp_response(
                        data.get("id"),
                        error={"code": -32000, "message": "Session unavailable. Initialize a new connection."},
                    ),
                    status_code=404,
                )
            if method == "notifications/initialized":
                return Response(status_code=202)

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
                "restore_token": session.get("restore_token", ""),
                # Recorded at initialize; User-Agent covers clients with no identity.
                "client_name": session.get("client_name", ""),
                "client_version": session.get("client_version", "")
            }

            # Handle the request
            mcp_response = await handle_mcp_request(
                data, session_defaults, mcp_session_id, sessions
            )
            return JSONResponse(mcp_response)

        except json.JSONDecodeError:
            return JSONResponse(
                create_mcp_response(
                    None,
                    error={"code": JSON_PARSE_ERROR, "message": "Invalid JSON"}
                ),
                status_code=400
            )
        except Exception:
            logger.error("MCP POST failed", extra={"event": "mcp.post_failed"})
            return JSONResponse(
                create_mcp_response(
                    None,
                    error={"code": -32000, "message": "Request could not be completed"}
                ),
                status_code=500
            )

    # Handle GET requests (SSE streaming for server-to-client messages)
    elif request.method == "GET":
        # Check for Last-Event-ID header for resumability
        last_event_id = request.headers.get("Last-Event-ID")

        if last_event_id:
            logger.debug("MCP stream resume requested", extra={"event": "mcp.stream_resume_requested"})

        # Session is required for GET (unless resuming with Last-Event-ID)
        if not mcp_session_id and not last_event_id:
            return JSONResponse(
                {"error": "MCP-Session-Id header required"},
                status_code=400
            )

        # Validate session if provided
        if mcp_session_id:
            session = await sessions.get_session(mcp_session_id)
            if not session or session["project_slug"] != project_slug:
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
                logger.info("MCP stream cancelled", extra={"event": "mcp.stream_cancelled"})
            except Exception:
                logger.error("MCP stream failed", extra={"event": "mcp.stream_failed"})

        return EventSourceResponse(sse_stream())


@app.delete("/mcp/{project_slug}")
@app.delete("/v2/{project_slug}/mcp")
async def delete_session_v2(
    project_slug: str,
    request: Request,
    mcp_session_id: Optional[str] = Header(None, alias="MCP-Session-Id")
):
    """V2: Delete/terminate a session."""
    logger.debug("MCP termination requested", extra={"event": "mcp.termination_requested"})

    if not mcp_session_id:
        return JSONResponse(
            {"error": "MCP-Session-Id header required"},
            status_code=400
        )

    sessions: SessionManager = request.app.state.session_manager
    session = await sessions.get_session(mcp_session_id)
    if session and session["project_slug"] == project_slug and await sessions.delete_session(mcp_session_id):
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


class MCPLogFormatter(logging.Formatter):
    """Render standalone diagnostic fields without credential or payload data."""

    converter = time.gmtime

    def format(self, record):
        data = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%SZ"),
            "level": record.levelname, "logger": record.name,
            "message": record.getMessage(),
        }
        for field in ("event", "error_kind", "duration_ms", "upstream_status",
                      "upstream_request_id", "has_continuation", "read_timeout_seconds", "mode"):
            if hasattr(record, field):
                data[field] = getattr(record, field)
        return json.dumps(data)


def main() -> None:
    """Run the standalone MCP server."""
    handler = logging.StreamHandler()
    handler.setFormatter(MCPLogFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    # HTTPX's info records include the complete search query string.
    logging.getLogger("httpx").setLevel(logging.WARNING)
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
