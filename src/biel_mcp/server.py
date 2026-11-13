"""Public package entrypoint and standalone ASGI application."""

import json
import logging
import os
import time

import uvicorn
from mcp.types.version import LATEST_HANDSHAKE_VERSION

from .api import (
    chat_recovery_hint,
    create_error_response,
    create_success_response,
    extract_client_info,
    format_biel_response,
    format_document_response,
    format_search_response,
    get_client_ip,
    get_user_agent,
    header_safe,
    normalize_base_url,
    query_biel_ai,
    query_biel_api,
    query_biel_document,
    query_biel_search,
    resolve_base_url,
    validate_biel_request,
)
from .config import (
    API_READ_TIMEOUT_SECONDS,
    CLIENT_IDENTITY_MAX_LENGTH,
    DEFAULT_BASE_URL,
    DEFAULT_PORT,
    REQUEST_TIMEOUT,
    SERVER_NAME,
    SERVER_VERSION,
    SESSION_TIMEOUT,
)
from .http import create_app
from .sessions import InMemorySessionStore, SessionManager, SessionStore
from .tools import create_server

__all__ = [
    "API_READ_TIMEOUT_SECONDS",
    "CLIENT_IDENTITY_MAX_LENGTH",
    "DEFAULT_BASE_URL",
    "DEFAULT_PORT",
    "MCP_PROTOCOL_VERSION_V2",
    "REQUEST_TIMEOUT",
    "SERVER_NAME",
    "SERVER_VERSION",
    "SESSION_TIMEOUT",
    "InMemorySessionStore",
    "MCPLogFormatter",
    "SessionManager",
    "SessionStore",
    "app",
    "chat_recovery_hint",
    "create_app",
    "create_error_response",
    "create_server",
    "create_success_response",
    "extract_client_info",
    "format_biel_response",
    "format_document_response",
    "format_search_response",
    "get_client_ip",
    "get_user_agent",
    "header_safe",
    "main",
    "normalize_base_url",
    "query_biel_ai",
    "query_biel_api",
    "query_biel_document",
    "query_biel_search",
    "resolve_base_url",
    "validate_biel_request",
]

MCP_PROTOCOL_VERSION_V2 = LATEST_HANDSHAKE_VERSION
app = create_app()
logger = logging.getLogger("biel-mcp")


class MCPLogFormatter(logging.Formatter):
    """Render standalone diagnostic fields without credential or payload data."""

    converter = time.gmtime

    def format(self, record):
        data = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%SZ"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in (
            "event",
            "error_kind",
            "duration_ms",
            "upstream_status",
            "upstream_request_id",
            "has_continuation",
            "read_timeout_seconds",
            "mode",
        ):
            if hasattr(record, field):
                data[field] = getattr(record, field)
        return json.dumps(data)


def main() -> None:
    """Run one standalone process with Streamable HTTP and legacy SSE."""
    handler = logging.StreamHandler()
    handler.setFormatter(MCPLogFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpx2").setLevel(logging.WARNING)
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
    logger.info("Starting %s v%s on port %s", SERVER_NAME, SERVER_VERSION, DEFAULT_PORT)
    uvicorn.run(app, host="0.0.0.0", port=DEFAULT_PORT, workers=1, access_log=False)


if __name__ == "__main__":
    main()
