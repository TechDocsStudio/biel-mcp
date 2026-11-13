"""SDK transports with Biel-owned, optionally shared application sessions."""

import re
from contextlib import asynccontextmanager
from urllib.parse import parse_qs, quote, urlsplit

from mcp.server.transport_security import TransportSecuritySettings
from mcp.shared.exceptions import MCPError
from mcp.types import ErrorData, JSONRPCError
from mcp.types.version import HANDSHAKE_PROTOCOL_VERSIONS
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from .api import (
    extract_client_info,
    get_client_ip,
    get_user_agent,
    normalize_base_url,
    resolve_base_url,
)
from .config import DEFAULT_BASE_URL, SERVER_NAME, SERVER_VERSION
from .context import RequestContext, current_request
from .sessions import SessionManager
from .tools import create_server

ENDPOINT = re.compile(r"(?:/mcp/([A-Za-z0-9_-]+)|/v2/([A-Za-z0-9_-]+)/mcp)")


def session_error(message="Session unavailable. Initialize a new connection."):
    error = JSONRPCError(
        jsonrpc="2.0", id=None, error=ErrorData(code=-32000, message=message)
    )
    return JSONResponse(
        error.model_dump(by_alias=True, exclude_unset=True), status_code=404
    )


class MCPHTTPBoundary:
    """Bind trusted connection configuration before entering SDK handlers."""

    def __init__(self, owner, http_app, sse_app=None):
        self.owner = owner
        self.http_app = http_app
        self.sse_app = sse_app
        self.sse_connections = {}

    async def initialize(self, ctx, call_next):
        context = current_request.get()
        scope = ctx.request.scope if ctx.request is not None else {}
        context = scope.get("biel_mcp_context", context)
        if context is None or not scope.get("biel_mcp_legacy_http"):
            return await call_next(ctx)
        if ctx.method != "initialize":
            if not context.session_id and not ctx.method.startswith("notifications/"):
                scope["biel_mcp_response_status"] = 400
                raise MCPError(code=-32000, message="MCP-Session-Id header required")
            return await call_next(ctx)

        result = await call_next(ctx)
        name, version = extract_client_info({"params": ctx.params})
        if context.session_id:
            await context.sessions.record_client_info(context.session_id, name, version)
        else:
            fields = {
                name: context.defaults.get(name, "")
                for name in (
                    "project_slug",
                    "api_key",
                    "base_url",
                    "domain",
                    "metadata",
                )
            }
            context.session_id = await context.sessions.create_session(
                **fields, client_name=name, client_version=version
            )
        scope["biel_mcp_response_session_id"] = context.session_id
        return result

    async def __call__(self, scope, receive, send):
        root_path = scope.get("root_path", "").rstrip("/")
        path = scope["path"]
        if root_path and (path == root_path or path.startswith(root_path + "/")):
            path = path[len(root_path) :]

        if self.sse_app is not None and path == "/messages/":
            request = Request(scope, receive)
            context = self.sse_connections.get(request.query_params.get("session_id"))
            if context is not None:
                context = RequestContext(
                    defaults={
                        **context.defaults,
                        "client_ip": get_client_ip(request),
                        "user_agent": get_user_agent(request),
                    },
                    sessions=context.sessions,
                    conversations=context.conversations,
                )
                scope = {**scope, "biel_mcp_context": context}
            token = current_request.set(context)
            try:
                return await self.sse_app(scope, receive, send)
            finally:
                current_request.reset(token)
        is_sse = self.sse_app is not None and path == "/sse"
        match = ENDPOINT.fullmatch(path)
        if not match and not is_sse:
            return await JSONResponse({"error": "Not found"}, 404)(scope, receive, send)

        request = Request(scope, receive)
        if request.method == "OPTIONS":
            return await Response(status_code=204)(scope, receive, send)
        if is_sse and request.method != "GET":
            return await Response(status_code=405, headers={"Allow": "GET"})(
                scope, receive, send
            )
        state = self.owner.state
        base_url = resolve_base_url(
            request.query_params.get("base_url"),
            state.allowed_base_urls,
            state.api_base_url,
        )
        if base_url is None:
            return await JSONResponse(
                {"error": "base_url is not an allowed Biel.ai instance"}, 400
            )(scope, receive, send)

        project_slug = (
            request.query_params.get("project_slug", "")
            if is_sse
            else next(value for value in match.groups() if value is not None)
        )
        if not project_slug or not re.fullmatch(r"[A-Za-z0-9_-]+", project_slug):
            return await JSONResponse({"error": "Invalid project slug"}, 400)(
                scope, receive, send
            )
        defaults = {
            "project_slug": project_slug,
            "api_key": request.query_params.get("api_key", ""),
            "base_url": base_url,
            "domain": request.query_params.get("domain", ""),
            "metadata": request.query_params.get("metadata", ""),
            "client_ip": get_client_ip(request),
            "user_agent": get_user_agent(request),
        }
        protocol = request.headers.get("mcp-protocol-version")
        legacy = not is_sse and (
            protocol is None or protocol in HANDSHAKE_PROTOCOL_VERSIONS
        )
        session_id = request.headers.get("mcp-session-id") if legacy else None
        await state.session_manager.cleanup_expired_sessions()
        await state.conversation_manager.cleanup_expired_sessions()
        if session_id:
            session = await state.session_manager.get_session(session_id)
            if session is None or session.get("project_slug") != project_slug:
                return await session_error()(scope, receive, send)
            if (
                normalize_base_url(session.get("base_url", ""))
                not in state.allowed_base_urls
            ):
                return await session_error()(scope, receive, send)
            defaults.update(
                {key: value for key, value in session.items() if isinstance(value, str)}
            )
            defaults["client_ip"] = get_client_ip(request)
            defaults["user_agent"] = get_user_agent(request)
        if not is_sse and request.method == "DELETE":
            if not session_id:
                return await JSONResponse(
                    {"error": "MCP-Session-Id header required"}, 400
                )(scope, receive, send)
            await state.session_manager.delete_session(session_id)
            return await JSONResponse(
                {"message": "Session terminated"}, status_code=200
            )(scope, receive, send)

        context = RequestContext(
            defaults=defaults,
            session_id=session_id,
            sessions=state.session_manager,
            conversations=state.conversation_manager,
        )
        rewritten = {
            **scope,
            "biel_mcp_context": context,
            "biel_mcp_legacy_http": legacy,
        }
        target = self.sse_app if is_sse else self.http_app
        if not is_sse:
            sdk_path = f"{root_path}/mcp"
            rewritten.update(
                path=sdk_path, raw_path=quote(sdk_path, safe="/").encode("ascii")
            )

        endpoint_buffer = bytearray()
        sse_identifier = None

        async def application_send(message):
            nonlocal sse_identifier
            if (
                is_sse
                and sse_identifier is None
                and len(endpoint_buffer) < 4096
                and message["type"] == "http.response.body"
            ):
                # Observe only the SDK's routing announcement, never tool payloads.
                endpoint_buffer.extend(
                    message.get("body", b"")[: 4096 - len(endpoint_buffer)]
                )
                frame = bytes(endpoint_buffer).replace(b"\r\n", b"\n").split(b"\n\n", 1)
                if len(frame) == 2 and frame[0].startswith(b"event: endpoint\n"):
                    data = next(
                        (
                            line[5:].strip()
                            for line in frame[0].splitlines()
                            if line.startswith(b"data:")
                        ),
                        b"",
                    )
                    values = parse_qs(urlsplit(data.decode("utf-8")).query)
                    identifiers = values.get("session_id", [])
                    if len(identifiers) == 1:
                        sse_identifier = identifiers[0]
                        self.sse_connections[sse_identifier] = context
                    endpoint_buffer.clear()
            if message["type"] == "http.response.start":
                if rewritten.get("biel_mcp_response_status"):
                    message = {
                        **message,
                        "status": rewritten["biel_mcp_response_status"],
                    }
                if identifier := rewritten.get("biel_mcp_response_session_id"):
                    message = {
                        **message,
                        "headers": [
                            *message.get("headers", []),
                            (b"mcp-session-id", identifier.encode("ascii")),
                        ],
                    }
            await send(message)

        token = current_request.set(context)
        try:
            return await target(rewritten, receive, application_send)
        finally:
            current_request.reset(token)
            if sse_identifier is not None:
                self.sse_connections.pop(sse_identifier, None)


def create_app(
    *,
    store=None,
    conversation_store=None,
    api_base_url=DEFAULT_BASE_URL,
    allowed_base_urls=None,
    client=None,
    enable_sse=True,
):
    """Create an independent app; the ASGI host must run its lifespan."""
    app = Starlette()
    app.state.session_manager = SessionManager(store=store)
    app.state.conversation_manager = SessionManager(store=conversation_store)
    app.state.api_base_url = normalize_base_url(api_base_url)
    app.state.allowed_base_urls = frozenset(
        normalize_base_url(url)
        for url in (
            allowed_base_urls if allowed_base_urls is not None else {api_base_url}
        )
    )
    boundary = MCPHTTPBoundary(app, None)
    server = create_server(client=client, middleware=[boundary.initialize])
    # Public endpoints accept non-cookie clients from any origin; hosting owns Host validation.
    security = TransportSecuritySettings(enable_dns_rebinding_protection=False)
    boundary.http_app = server.streamable_http_app(
        stateless_http=True,
        json_response=True,
        transport_security=security,
    )
    if enable_sse:
        sse_server = create_server(client=client)
        boundary.sse_app = sse_server.sse_app(transport_security=security)
        app.state.sse_server = sse_server
    app.state.mcp_server = server

    @asynccontextmanager
    async def lifespan(application):
        async with server.session_manager.run():
            yield

    async def health(request):
        return JSONResponse(
            {"status": "ok", "service": SERVER_NAME, "version": SERVER_VERSION}
        )

    app.router.lifespan_context = lifespan
    app.router.routes.extend(
        [
            Route("/", endpoint=health),
            Route("/health", endpoint=health),
            Route(
                "/{path:path}",
                endpoint=boundary,
                methods=["GET", "POST", "DELETE", "OPTIONS"],
            ),
        ]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["Mcp-Session-Id"],
    )
    return app
