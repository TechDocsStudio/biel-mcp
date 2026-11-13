"""Biel API requests, public errors and tool result formatting."""

import logging
import re
import time
from contextlib import asynccontextmanager
from typing import Any

import httpx
from starlette.requests import Request

from .config import (
    API_READ_TIMEOUT_SECONDS,
    BIEL_API_PATH_TEMPLATE,
    BIEL_DOCUMENT_PATH_TEMPLATE,
    BIEL_SEARCH_PATH_TEMPLATE,
    CLIENT_IDENTITY_MAX_LENGTH,
    DEFAULT_BASE_URL,
    PROJECT_SLUG_PATTERN,
    REQUEST_TIMEOUT,
)

logger = logging.getLogger("biel-mcp")


@asynccontextmanager
async def api_client(client: httpx.AsyncClient | None = None):
    """Borrow injected clients; close only resources owned by this call."""
    if client is not None:
        yield client
    else:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT, trust_env=False) as owned:
            yield owned


def normalize_base_url(base_url: str) -> str:
    """Compare base URLs without tripping over a trailing slash."""
    return base_url.strip().rstrip("/")


def resolve_base_url(
    requested: str | None, allowed: frozenset, default: str
) -> str | None:
    """Resolve only backend origins authorized by the host."""
    if not requested:
        return default
    candidate = normalize_base_url(requested)
    if candidate in {normalize_base_url(url) for url in allowed}:
        return candidate
    return None


def create_error_response(message: str) -> dict[str, Any]:
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
    """Bound a client-supplied value to one Latin-1 HTTP header."""
    if not isinstance(value, str):
        return ""
    collapsed = " ".join(value.split())
    encodable = collapsed.encode("latin-1", "ignore").decode("latin-1")
    return encodable[:CLIENT_IDENTITY_MAX_LENGTH]


def get_user_agent(request: Request) -> str:
    """The client's User-Agent — identity for transports with no handshake."""
    return header_safe(request.headers.get("user-agent", ""))


def extract_client_info(data: dict[str, Any]) -> tuple[str, str]:
    """Read the sanitized identity from a legacy initialize request."""
    if not isinstance(data, dict):
        return "", ""
    params = data.get("params")
    client_info = params.get("clientInfo") if isinstance(params, dict) else None
    if not isinstance(client_info, dict):
        return "", ""
    return header_safe(client_info.get("name")), header_safe(client_info.get("version"))


def create_success_response(text: str) -> dict[str, Any]:
    """Create a standardized success response."""
    return {"content": [{"type": "text", "text": text}], "isError": False}


def validate_biel_request(arguments: dict[str, Any]) -> str | None:
    """Validate Biel.ai request arguments. Returns error message if invalid, None if valid."""
    mode = arguments.get("mode", "answer")
    if mode != "get" and (
        not isinstance(arguments.get("message"), str)
        or not arguments["message"].strip()
    ):
        return "Message cannot be empty"

    if mode not in ("answer", "search", "get"):
        return "Mode must be 'answer' or 'search'"
    if mode == "search":
        limit = arguments.get("limit", 5)
        if type(limit) is not int or not 1 <= limit <= 20:
            return "Search limit must be an integer from 1 to 20"
    if mode == "get":
        document_id = arguments.get("document_id")
        if not isinstance(document_id, str) or not re.fullmatch(
            r"[A-Za-z0-9_.:-]{1,512}", document_id
        ):
            return "A document_id from search is required"
        cursor = arguments.get("cursor", "")
        if not isinstance(cursor, str) or len(cursor) > 512:
            return "Invalid document cursor"
        limit = arguments.get("limit", 20)
        if type(limit) is not int or not 1 <= limit <= 20:
            return "Document limit must be an integer from 1 to 20"

    project_slug = arguments.get("project_slug", "")
    if not isinstance(project_slug, str):
        return "Project slug must be a string"
    project_slug = project_slug.strip()
    if not project_slug:
        return "Project slug is required"

    if not PROJECT_SLUG_PATTERN.match(project_slug):
        return "Invalid project slug"

    return None


def chat_recovery_hint(response) -> str:
    if response.status_code not in (400, 409, 503):
        return ""
    try:
        code = response.json().get("code")
    except (ValueError, AttributeError):
        return ""
    return (
        {
            "context_limit": "The current request still exceeds the model context after preparing the conversation. Ask about a smaller section; existing messages are preserved.",
            "compaction_failed": "Earlier conversation could not be prepared right now. Retry in this chat; existing messages are preserved.",
            "message_too_long": "The message is too long. Use at most 10,000 characters.",
            "output_limit": "The model reached its output limit before producing an answer. Try a shorter request.",
            "generation_busy": "This conversation is already generating a response. Wait before retrying.",
        }.get(code, "")
        if isinstance(code, str)
        else ""
    )


def format_biel_response(data: dict[str, Any]) -> str:
    """Format the response from Biel.ai API into a readable string."""
    ai_message = data.get("ai_message", {})
    ai_response = ai_message.get("message", "No response received")
    sources = ai_message.get("sources", [])

    response_parts = [f"🤖 **Biel.ai responds:**\n\n{ai_response}"]

    if ai_message.get("incomplete"):
        reason = (ai_message.get("generation_metadata") or {}).get("stop_reason")
        if reason in ("length", "max_tokens"):
            response_parts.append(
                "\n\n**Incomplete answer:** The output limit was reached after automatic continuation. Ask for a shorter answer or a specific section."
            )
        else:
            response_parts.append(
                "\n\n**Incomplete answer:** Generation was interrupted. Ask again with the relevant details."
            )

    if sources:
        response_parts.append("\n\n📚 **Sources consulted:**")
        for source in sources:
            response_parts.append(f"• [{source['title']}]({source['url']})")

    return "\n".join(response_parts)


def format_search_response(data: dict[str, Any], limit: int = 5) -> str:
    """Return ranked source excerpts for the client to synthesize its own answer."""
    results = data["results"]
    if not isinstance(results, list):
        raise ValueError("Invalid search response")  # noqa: TRY004
    if not results:
        return "No matching documentation found. Try different search terms."
    parts = [
        (
            f"Documentation search results: showing {min(limit, len(results))} of {len(results)} "
            "matches (source excerpts, no generated answer)."
        )
    ]
    for number, result in enumerate(results[:limit], 1):
        location = [
            result.get("page_title") or result.get("title") or "Untitled source"
        ]
        if result.get("page"):
            location.append(f"page {result['page']}")
        if result.get("sheet"):
            location.append(f"sheet {result['sheet']}")
        reference = (
            f"URL: {result['url']}"
            if result.get("url")
            else "Reference: " + ", ".join(location)
        )
        document_id = (
            f"\nDocument ID: {result['document_id']}"
            if result.get("document_id")
            else ""
        )
        parts.append(
            f"{number}. {result.get('title', '')} ({result.get('source_type', 'page')})\n"
            f"{reference}{document_id}\n"
            f"{result.get('content') or result.get('fragment', '')}"
        )
    return "\n\n".join(parts)


def format_document_response(data: dict[str, Any]) -> str:
    chunks = data["chunks"]
    total = data["total_chunks"]
    if (
        not isinstance(chunks, list)
        or not chunks
        or type(total) is not int
        or total < len(chunks)
    ):
        raise ValueError("Invalid document response")
    reference = (
        data.get("url")
        or data.get("file_path")
        or data.get("title")
        or "Untitled document"
    )
    parts = [
        f"Indexed document: {data.get('title', '')} ({data['source_type']})",
        (
            f"Reference: {reference}\nDocument ID: {data['document_id']}\n"
            f"Returned {len(chunks)} of {total} indexed chunks in this response (no generated answer)."
        ),
    ]
    for chunk in chunks:
        if not isinstance(chunk.get("content"), str):
            raise ValueError("Invalid document chunk")  # noqa: TRY004
        location = [f"Chunk {chunk['chunk_index']}"]
        if chunk.get("page"):
            location.append(f"page {chunk['page']}")
        if chunk.get("sheet"):
            location.append(f"sheet {chunk['sheet']}")
        parts.append(", ".join(location) + "\n" + chunk["content"])
    if data.get("next_cursor"):
        parts.append(
            f"Next cursor: {data['next_cursor']}\nCall biel_get_document with this cursor and the same document_id and limit to continue."
        )
    else:
        parts.append("End of indexed document.")
    return "\n\n".join(parts)


async def query_biel_document(
    arguments: dict[str, Any],
    defaults: dict[str, str] | None = None,
    *,
    client: httpx.AsyncClient | None = None,
):
    if not isinstance(arguments, dict):
        return create_error_response("Tool arguments must be an object."), None, None
    return await query_biel_api({**arguments, "mode": "get"}, defaults, client=client)


async def query_biel_ai(
    arguments: dict[str, Any],
    defaults: dict[str, str] | None = None,
    *,
    client: httpx.AsyncClient | None = None,
) -> tuple[dict[str, Any], str | None, str | None]:
    if not isinstance(arguments, dict):
        return create_error_response("Tool arguments must be an object."), None, None
    return await query_biel_api(
        {**arguments, "mode": "answer"}, defaults, client=client
    )


async def query_biel_search(
    arguments: dict[str, Any],
    defaults: dict[str, str] | None = None,
    *,
    client: httpx.AsyncClient | None = None,
) -> tuple[dict[str, Any], str | None, str | None]:
    if not isinstance(arguments, dict):
        return create_error_response("Tool arguments must be an object."), None, None
    return await query_biel_api(
        {**arguments, "message": arguments.get("query"), "mode": "search"},
        defaults,
        client=client,
    )


async def query_biel_api(
    arguments: dict[str, Any],
    defaults: dict[str, str] | None = None,
    *,
    client: httpx.AsyncClient | None = None,
) -> tuple[dict[str, Any], str | None, str | None]:
    """Relay an API request and return its result plus any conversation credentials."""
    if not isinstance(arguments, dict):
        return create_error_response("Tool arguments must be an object."), None, None
    arguments = dict(arguments)
    mode = arguments.get("mode", "answer")
    if mode not in ("answer", "search", "get"):
        return create_error_response("Mode must be 'answer' or 'search'"), None, None
    session_project = (defaults or {}).get("project_slug")
    if session_project and arguments.get("project_slug") not in (
        None,
        "",
        session_project,
    ):
        return (
            create_error_response("The project is fixed by this connection."),
            None,
            None,
        )
    requested_chat = arguments.get("chat_uuid")
    session_chat = (defaults or {}).get("chat_uuid")
    session_token = (defaults or {}).get("restore_token")
    if (
        mode == "answer"
        and requested_chat
        and (requested_chat != session_chat or not session_token)
    ):
        logger.warning(
            "MCP continuation unavailable",
            extra={
                "event": "mcp.continuation_rejected",
                "error_kind": "session_mismatch",
            },
        )
        return (
            create_error_response(
                "This conversation cannot be resumed by this connection. "
                "Reconnect to start a new conversation; do not supply a chat UUID."
            ),
            None,
            None,
        )
    if mode == "answer" and session_chat and not session_token:
        logger.warning(
            "MCP continuation unavailable",
            extra={
                "event": "mcp.continuation_rejected",
                "error_kind": "missing_capability",
            },
        )
        return (
            create_error_response(
                "This connection cannot resume its conversation. Reconnect to start a new conversation."
            ),
            None,
            None,
        )
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

    # Only host-vetted defaults choose the upstream backend.
    message = arguments.get("message", "")
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
        "metadata": metadata,
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
    # Handshake identity survives stateless requests through application state.
    if client_name:
        headers["X-MCP-Client-Name"] = client_name
    if client_version:
        headers["X-MCP-Client-Version"] = client_version
    if user_agent:
        headers["X-MCP-Client-UA"] = user_agent

    path_template = (
        BIEL_SEARCH_PATH_TEMPLATE if mode == "search" else BIEL_API_PATH_TEMPLATE
    )
    if mode == "get":
        path_template = BIEL_DOCUMENT_PATH_TEMPLATE
    full_url = f"{base_url.rstrip('/')}{path_template.format(project_slug=project_slug, document_id=arguments.get('document_id', ''))}"

    started = time.monotonic()
    outcome = "upstream_error"
    upstream_status = None
    upstream_request_id = ""
    try:
        async with api_client(client) as upstream:
            if mode == "search":
                response = await upstream.get(
                    full_url,
                    params={
                        "q": message,
                        "url": domain or base_url,
                        "source_types": "all",
                        "search_type": "hybrid",
                    },
                    headers=headers,
                )
            elif mode == "get":
                response = await upstream.get(
                    full_url,
                    params={
                        "url": domain or base_url,
                        "cursor": arguments.get("cursor", ""),
                        "limit": arguments.get("limit", 20),
                    },
                    headers=headers,
                )
            else:
                response = await upstream.post(full_url, json=payload, headers=headers)
            upstream_status = response.status_code
            upstream_request_id = header_safe(response.headers.get("x-request-id", ""))

            if response.status_code in (200, 201):
                data = response.json()
                if mode == "search":
                    formatted_response = format_search_response(
                        data, arguments.get("limit", 5)
                    )
                elif mode == "get":
                    if data.get("document_id") != arguments["document_id"]:
                        raise ValueError("Document reference mismatch")
                    formatted_response = format_document_response(data)
                else:
                    formatted_response = format_biel_response(data)
                outcome = "success"
                if mode != "answer":
                    return create_success_response(formatted_response), None, None
                # Persist the pair together: the token is bound to this chat.
                return (
                    create_success_response(formatted_response),
                    data.get("chat_uuid"),
                    data.get("restore_token"),
                )
            else:
                error_msg = (
                    chat_recovery_hint(response)
                    or f"Biel.ai API returned HTTP {response.status_code}."
                )
                if response.status_code == 403:
                    error_msg += (
                        " Check project search access and quota."
                        if mode != "answer"
                        else " Check project access or reconnect to start a new conversation."
                    )
                elif response.status_code == 404:
                    error_msg += (
                        " The project or document is unavailable; search again for a current document reference."
                        if mode == "get"
                        else " The project is unavailable; check the connection."
                        if mode == "search"
                        else " The project or conversation is unavailable; check the connection and reconnect."
                    )
                elif response.status_code == 409 and mode == "get":
                    error_msg += (
                        " Recrawl this source to enable complete document retrieval."
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
    except Exception:  # noqa: BLE001 - sanitize unexpected upstream response failures.
        outcome = "transport_error"
        return (
            create_error_response(
                "Biel.ai could not complete the request. Please try again."
            ),
            None,
            None,
        )
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
