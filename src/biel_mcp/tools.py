"""SDK-managed tools with application-owned conversation continuity."""

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from hmac import compare_digest
from inspect import signature
from typing import Annotated, Any
from uuid import UUID

import httpx
from mcp.server.mcpserver import Context, MCPServer
from mcp.types import CallToolResult, ToolAnnotations
from pydantic import Field

from . import api
from .config import DEFAULT_BASE_URL, SERVER_NAME, SERVER_VERSION
from .context import RequestContext, current_request
from .sessions import SessionManager

AI_DESCRIPTION = (
    "Ask Biel.ai to generate an answer grounded in this project's indexed "
    "product documentation, APIs, SDKs and knowledge base. Supports conversational "
    "context within this connection. Use when the user explicitly requests a "
    "Biel.ai-generated response; otherwise ask the user first. For source retrieval "
    "and your own answer, use biel_search and biel_get_document. "
    "Do not fall back to biel_ai automatically on a retrieval error. "
    "Without a transport session, reuse the returned conversation_id for follow-up "
    "questions; omit it to start a new conversation."
)
SEARCH_DESCRIPTION = (
    "Search this project's indexed documentation using hybrid keyword and semantic "
    "retrieval. Use for product setup, API and SDK usage, configuration and "
    "troubleshooting. Searches web pages, uploaded files, repositories and OpenAPI "
    "sources without generating a Biel.ai answer. Start here for documentation "
    "questions; natural-language questions or concise search terms both work. "
    "Results include ranked text chunks, source references and document IDs. "
    "Use biel_get_document with relevant IDs when excerpts need more context, "
    "then write your own answer and cite the sources. No automatic fallback to "
    "biel_ai; ask the user before requesting a Biel.ai-generated answer."
)
DOCUMENT_DESCRIPTION = (
    "Retrieve complete indexed document text using a document_id from biel_search. "
    "Reads indexed web pages, uploaded files, repository files and OpenAPI content "
    "without visiting the source site or generating an answer. Large documents return "
    "a next_cursor: repeat with that cursor and the same document_id and limit until "
    "next_cursor is absent. References belong to this connection's project and active "
    "index; search again if a document is unavailable."
)

ApiKey = Annotated[
    str, Field(strict=True, description="API key for authentication (optional)")
]
Domain = Annotated[
    str,
    Field(
        strict=True, description="Domain URL if Allowed domains is enabled (optional)"
    ),
]
Limit = Annotated[int, Field(strict=True, ge=1, le=20)]


def request_context(ctx: Context) -> RequestContext:
    bound = current_request.get()
    if bound is not None:
        return bound
    request = getattr(ctx.request_context, "request", None)
    scope = getattr(request, "scope", {})
    return scope.get("biel_mcp_context") or RequestContext()


def request_defaults(ctx: Context, bound: RequestContext) -> dict[str, str]:
    defaults = dict(bound.defaults)
    params = ctx.session.client_params
    info = params.client_info if params else None
    # Stateless legacy requests recover identity from Biel's shared record.
    if not bound.session_id and info is not None:
        defaults["client_name"] = api.header_safe(info.name)
        defaults["client_version"] = api.header_safe(info.version)
    return defaults


def result_from_api(result: dict[str, Any]) -> CallToolResult:
    return CallToolResult.model_validate(result)


def conversation_error() -> CallToolResult:
    return result_from_api(
        api.create_error_response(
            "Conversation unavailable for this connection. Start a new conversation."
        )
    )


def is_conversation_id(identifier: str) -> bool:
    try:
        parsed = UUID(identifier)
        return parsed.version == 4 and str(parsed) == identifier
    except (TypeError, ValueError, AttributeError):
        return False


def same_authority(
    record: dict[str, Any], defaults: dict[str, str], api_key: str
) -> bool:
    return (
        record.get("project_slug") == defaults.get("project_slug")
        and api.normalize_base_url(record.get("base_url", ""))
        == api.normalize_base_url(defaults.get("base_url", DEFAULT_BASE_URL))
        and compare_digest(
            str(record.get("api_key", "")).encode("utf-8"), api_key.encode("utf-8")
        )
    )


def create_server(
    *, client: httpx.AsyncClient | None = None, middleware: Sequence[Any] = ()
) -> MCPServer:
    allowed: dict[str, set[str]] = {}

    async def validate_arguments(ctx, call_next):
        if ctx.method == "tools/call" and isinstance(ctx.params, dict):
            name = ctx.params.get("name")
            arguments = ctx.params.get("arguments", {})
            if (
                isinstance(name, str)
                and name in allowed
                and isinstance(arguments, dict)
                and set(arguments) - allowed[name]
            ):
                return result_from_api(
                    api.create_error_response("Unexpected tool arguments.")
                )
        return await call_next(ctx)

    @asynccontextmanager
    async def lifespan(server: MCPServer) -> AsyncIterator[httpx.AsyncClient]:
        async with api.api_client(client) as upstream:
            yield upstream

    server = MCPServer(
        SERVER_NAME,
        version=SERVER_VERSION,
        lifespan=lifespan,
        middleware=[*middleware, validate_arguments],
    )

    def register(description, annotations=None):
        def tool(function):
            allowed[function.__name__] = set(signature(function).parameters) - {"ctx"}
            return server.tool(
                description=description,
                annotations=annotations,
                structured_output=False,
            )(function)

        return tool

    @register(AI_DESCRIPTION)
    async def biel_ai(
        ctx: Context,
        message: Annotated[
            str, Field(strict=True, description="Question for Biel.ai to answer")
        ],
        api_key: ApiKey = "",
        domain: Domain = "",
        metadata: Annotated[
            str,
            Field(
                strict=True,
                description="Metadata to tag the conversation source (optional)",
            ),
        ] = "",
        conversation_id: Annotated[
            str | None,
            Field(
                description="Opaque conversation ID returned by biel_ai; omit for a new chat"
            ),
        ] = None,
    ) -> CallToolResult:
        bound = request_context(ctx)
        defaults = request_defaults(ctx, bound)
        effective_key = api_key or defaults.get("api_key", "")
        arguments = {
            "message": message,
            "api_key": api_key,
            "domain": domain,
            "metadata": metadata,
        }
        manager: SessionManager | None
        identifier: str | None
        modern = not bound.session_id
        if bound.session_id:
            manager, identifier = bound.sessions, bound.session_id
            if conversation_id is not None and conversation_id != identifier:
                return conversation_error()
        else:
            manager, identifier = bound.conversations, conversation_id
            if manager is None:
                return result_from_api(
                    api.create_error_response("Conversation storage unavailable.")
                )
            if identifier is not None:
                if not is_conversation_id(identifier):
                    return conversation_error()
                record = await manager.get_session(identifier)
                if record is None or not same_authority(
                    record, defaults, effective_key
                ):
                    return conversation_error()
                defaults.update(
                    {
                        "chat_uuid": record.get("chat_uuid", ""),
                        "restore_token": record.get("restore_token", ""),
                        "domain": record.get("domain", ""),
                        "metadata": record.get("metadata", ""),
                    }
                )

        result, chat_uuid, restore_token = await api.query_biel_ai(
            arguments, defaults, client=ctx.request_context.lifespan_context
        )
        if result.get("isError"):
            return result_from_api(result)
        credentials_valid = all(
            isinstance(value, str) and value for value in (chat_uuid, restore_token)
        )
        if modern and not credentials_valid:
            return result_from_api(
                api.create_error_response(
                    "Biel.ai returned incomplete conversation credentials. Please try again."
                )
            )
        if credentials_valid and manager is not None:
            if identifier is None:
                identifier = await manager.create_session(
                    project_slug=defaults["project_slug"],
                    api_key=effective_key,
                    base_url=defaults.get("base_url", DEFAULT_BASE_URL),
                    domain=domain or defaults.get("domain", ""),
                    metadata=metadata or defaults.get("metadata", ""),
                    client_name=defaults.get("client_name", ""),
                    client_version=defaults.get("client_version", ""),
                )
            await manager.record_chat_credentials(identifier, chat_uuid, restore_token)
        if modern and identifier is not None and chat_uuid and restore_token:
            result = {
                **result,
                "content": [
                    *result["content"],
                    {"type": "text", "text": f"Conversation ID: {identifier}"},
                ],
                "structuredContent": {"conversation_id": identifier},
            }
        return result_from_api(result)

    @register(SEARCH_DESCRIPTION, ToolAnnotations(read_only_hint=True))
    async def biel_search(
        ctx: Context,
        query: Annotated[
            str,
            Field(strict=True, description="Documentation question or search terms"),
        ],
        limit: Limit = 5,
        api_key: ApiKey = "",
        domain: Domain = "",
    ) -> CallToolResult:
        defaults = request_defaults(ctx, request_context(ctx))
        result, _, _ = await api.query_biel_search(
            {"query": query, "limit": limit, "api_key": api_key, "domain": domain},
            defaults,
            client=ctx.request_context.lifespan_context,
        )
        return result_from_api(result)

    @register(DOCUMENT_DESCRIPTION, ToolAnnotations(read_only_hint=True))
    async def biel_get_document(
        ctx: Context,
        document_id: Annotated[
            str,
            Field(
                strict=True,
                pattern=r"^[A-Za-z0-9_.:-]{1,512}$",
                max_length=512,
                description="Document ID returned by search",
            ),
        ],
        cursor: Annotated[
            str,
            Field(
                strict=True,
                max_length=512,
                description="Continuation cursor from the previous get response",
            ),
        ] = "",
        limit: Limit = 20,
        api_key: ApiKey = "",
        domain: Domain = "",
    ) -> CallToolResult:
        defaults = request_defaults(ctx, request_context(ctx))
        result, _, _ = await api.query_biel_document(
            {
                "document_id": document_id,
                "cursor": cursor,
                "limit": limit,
                "api_key": api_key,
                "domain": domain,
            },
            defaults,
            client=ctx.request_context.lifespan_context,
        )
        return result_from_api(result)

    return server
