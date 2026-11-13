"""Real SDK clients and independent ASGI applications exercise public transports."""

import asyncio
import json
import socket
from contextlib import AsyncExitStack
from datetime import datetime, timedelta
from unittest import IsolatedAsyncioTestCase

import httpx
import httpx2
import uvicorn
from asgi_lifespan import LifespanManager
from mcp import Client
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import (
    CLIENT_CAPABILITIES_META_KEY,
    PROTOCOL_VERSION_META_KEY,
    Implementation,
)
from mcp.types.jsonrpc import jsonrpc_message_adapter

from biel_mcp.config import SESSION_TIMEOUT
from biel_mcp.http import create_app
from biel_mcp.sessions import InMemorySessionStore

HEADERS = {
    "Accept": "application/json, text/event-stream",
    "MCP-Protocol-Version": "2025-11-25",
}


def initialize(name="sdk-test", version="1"):
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": {"name": name, "version": version},
        },
    }


def call(name="biel_ai", **arguments):
    return {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }


class Backend:
    def __init__(self):
        self.requests = []
        self.chat_count = 0

    def __call__(self, request):
        self.requests.append(request)
        if request.url.path.endswith("/chats/"):
            payload = json.loads(request.content)
            if not payload.get("chat_uuid"):
                self.chat_count += 1
            chat = payload.get("chat_uuid") or f"backend-chat-{self.chat_count}"
            return httpx.Response(
                200,
                json={
                    "chat_uuid": chat,
                    "restore_token": f"secret-restore-{chat}",
                    "ai_message": {"message": "answer", "sources": []},
                },
            )
        if request.url.path.endswith("/search/"):
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "title": "README.md",
                            "content": "Indexed full text",
                            "file_path": "client/README.md",
                            "source_type": "repository",
                            "document_id": "123:signed-reference",
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "document_id": "123:signed-reference",
                "title": "README.md",
                "source_type": "repository",
                "file_path": "client/README.md",
                "total_chunks": 2,
                "chunks": [{"content": "Indexed document", "chunk_index": 0}],
                "next_cursor": "next-reference",
            },
        )


class HTTPTest(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.stack = AsyncExitStack()
        await self.stack.__aenter__()
        self.addAsyncCleanup(self.stack.aclose)
        self.backend = Backend()
        self.upstream = await self.stack.enter_async_context(
            httpx.AsyncClient(transport=httpx.MockTransport(self.backend))
        )
        self.sessions = InMemorySessionStore()
        self.conversations = InMemorySessionStore()
        self.app = await self.new_app()
        self.client = await self.http(self.app)

    async def new_app(self, **kwargs):
        app = create_app(
            store=self.sessions,
            conversation_store=self.conversations,
            client=self.upstream,
            **kwargs,
        )
        await self.stack.enter_async_context(LifespanManager(app))
        return app

    async def http(self, app):
        return await self.stack.enter_async_context(
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://testserver",
                headers=HEADERS,
            )
        )

    async def established(self, path="/mcp/docs?api_key=private-key", client=None):
        client = client or self.client
        initialized = await client.post(path, json=initialize())
        self.assertEqual(initialized.status_code, 200, initialized.text)
        identifier = initialized.headers["MCP-Session-Id"]
        headers = {"MCP-Session-Id": identifier}
        response = await client.post(
            path,
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
            headers=headers,
        )
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.content, b"")
        return identifier, headers

    async def modern(self, body, *, path="/mcp/docs?api_key=private-key", client=None):
        body = {
            **body,
            "params": {
                **body.get("params", {}),
                "_meta": {
                    PROTOCOL_VERSION_META_KEY: "2026-07-28",
                    CLIENT_CAPABILITIES_META_KEY: {},
                },
            },
        }
        return await (client or self.client).post(
            path,
            json=body,
            headers={
                "MCP-Protocol-Version": "2026-07-28",
                "Mcp-Method": body["method"],
                "Mcp-Name": body["params"].get("name", ""),
            },
        )

    async def test_real_sdk_client_discovers_and_calls_all_three_tools_on_both_aliases(
        self,
    ):
        for path in ("/mcp/docs", "/v2/docs/mcp"):
            async with (
                httpx2.AsyncClient(
                    transport=httpx2.ASGITransport(app=self.app)
                ) as wire,
                Client(
                    streamable_http_client(
                        f"http://testserver{path}?api_key=private-key", http_client=wire
                    ),
                    mode="legacy",
                    client_info=Implementation(name="real-client", version="3"),
                ) as client,
            ):
                tools = (await client.list_tools()).tools
                self.assertEqual(
                    [tool.name for tool in tools],
                    ["biel_ai", "biel_search", "biel_get_document"],
                )
                self.assertIn("conversation_id", tools[0].input_schema["properties"])
                for name, arguments, text in (
                    ("biel_ai", {"message": "hello"}, "answer"),
                    ("biel_search", {"query": "setup"}, "Indexed full text"),
                    (
                        "biel_get_document",
                        {"document_id": "123:signed-reference"},
                        "Indexed document",
                    ),
                ):
                    result = await client.call_tool(name, arguments)
                    self.assertFalse(result.is_error, result)
                    self.assertIn(text, result.content[0].text)
            self.assertEqual(
                self.backend.requests[-1].headers["X-MCP-Client-Name"], "real-client"
            )
        self.assertFalse(self.upstream.is_closed)

    async def test_real_modern_sdk_client_creates_and_continues_without_initialize(
        self,
    ):
        async with (
            httpx2.AsyncClient(transport=httpx2.ASGITransport(app=self.app)) as wire,
            Client(
                streamable_http_client(
                    "http://testserver/mcp/docs?api_key=private-key", http_client=wire
                ),
                mode="2026-07-28",
                client_info=Implementation(name="modern-client", version="2"),
            ) as client,
        ):
            result = await client.call_tool("biel_ai", {"message": "first"})
            identifier = result.structured_content["conversation_id"]
            following = await client.call_tool(
                "biel_ai", {"message": "next", "conversation_id": identifier}
            )
            self.assertEqual(
                following.structured_content["conversation_id"], identifier
            )
        self.assertEqual(
            json.loads(self.backend.requests[-1].content)["chat_uuid"], "backend-chat-1"
        )
        self.assertEqual(
            self.backend.requests[-1].headers["X-MCP-Client-Name"], "modern-client"
        )
        self.assertEqual(self.sessions.sessions, {})

    async def test_legacy_continuation_identity_and_search_survive_an_instance_hop(
        self,
    ):
        other = await self.http(await self.new_app())
        path = "/mcp/docs?api_key=private-key&domain=https%3A%2F%2Fdocs.example.com&metadata=docs-mcp"
        identifier, headers = await self.established(path)
        first = await self.client.post(
            path, json=call(message="first"), headers=headers
        )
        search = await other.post(
            "/mcp/docs", json=call("biel_search", query="search"), headers=headers
        )
        self.assertFalse(search.json()["result"]["isError"])
        self.assertNotIn("X-Chat-Restore-Token", self.backend.requests[-1].headers)
        following = await other.post(
            "/mcp/docs", json=call(message="follow-up"), headers=headers
        )
        self.assertFalse(first.json()["result"]["isError"])
        self.assertFalse(following.json()["result"]["isError"])
        self.assertNotIn("structuredContent", following.json()["result"])
        relay = self.backend.requests[-1]
        self.assertEqual(json.loads(relay.content)["chat_uuid"], "backend-chat-1")
        self.assertEqual(json.loads(relay.content)["metadata"], "docs-mcp")
        self.assertEqual(relay.headers["Authorization"], "Api-Key private-key")
        self.assertEqual(relay.headers["X-MCP-Client-Name"], "sdk-test")
        self.assertEqual(
            relay.headers["X-Chat-Restore-Token"], "secret-restore-backend-chat-1"
        )
        self.assertEqual(
            (await self.app.state.session_manager.get_session(identifier))["chat_uuid"],
            "backend-chat-1",
        )

    async def test_pre_refactor_session_records_work_without_reinitialization(self):
        identifier = await self.app.state.session_manager.create_session(
            project_slug="docs",
            api_key="private-key",
            client_name="old-client",
            client_version="1",
        )
        await self.app.state.session_manager.record_chat_credentials(
            identifier, "prior-chat", "prior-restore"
        )
        response = await self.client.post(
            "/mcp/docs",
            json=call(message="next"),
            headers={"MCP-Session-Id": identifier},
        )
        self.assertFalse(response.json()["result"]["isError"])
        self.assertEqual(
            json.loads(self.backend.requests[-1].content)["chat_uuid"], "prior-chat"
        )
        self.assertEqual(
            self.backend.requests[-1].headers["X-Chat-Restore-Token"], "prior-restore"
        )

    async def test_expiry_deletion_and_project_mismatch_are_visible_on_other_instances(
        self,
    ):
        other = await self.http(await self.new_app())
        identifier, headers = await self.established()
        for method in (other.post, other.get, other.delete):
            kwargs = {"json": call(message="forbidden")} if method == other.post else {}
            response = await method("/mcp/foreign", headers=headers, **kwargs)
            self.assertEqual(response.status_code, 404)
            self.assertIsNone(response.json()["id"])
            jsonrpc_message_adapter.validate_python(response.json(), by_name=False)
        response = await other.delete("/mcp/docs", headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            (
                await self.client.post(
                    "/mcp/docs", json=call(message="gone"), headers=headers
                )
            ).status_code,
            404,
        )
        identifier, headers = await self.established()
        self.sessions.sessions[identifier]["last_active"] = datetime.now() - timedelta(
            seconds=SESSION_TIMEOUT + 1
        )
        self.assertEqual(
            (
                await other.post(
                    "/mcp/docs", json=call(message="expired"), headers=headers
                )
            ).status_code,
            404,
        )
        self.assertEqual(self.backend.requests, [])

    async def test_legacy_explicit_conversation_id_must_match_implicit_session(self):
        identifier, headers = await self.established()
        response = await self.client.post(
            "/mcp/docs",
            json=call(message="bad", conversation_id="other"),
            headers=headers,
        )
        self.assertTrue(response.json()["result"]["isError"])
        self.assertEqual(self.backend.requests, [])
        response = await self.client.post(
            "/mcp/docs",
            json=call(message="ok", conversation_id=identifier),
            headers=headers,
        )
        self.assertFalse(response.json()["result"]["isError"])

    async def test_modern_conversation_id_is_scoped_shared_and_never_a_backend_capability(
        self,
    ):
        response = await self.modern(call(message="first"))
        self.assertEqual(response.status_code, 200, response.text)
        identifier = response.json()["result"]["structuredContent"]["conversation_id"]
        self.assertNotIn("MCP-Session-Id", response.headers)
        self.assertNotIn("secret-restore", response.text)
        self.assertNotIn("backend-chat", response.text)
        other = await self.http(await self.new_app())
        following = await self.modern(
            call(message="next", conversation_id=identifier), client=other
        )
        self.assertEqual(
            following.json()["result"]["structuredContent"]["conversation_id"],
            identifier,
        )
        self.assertEqual(
            json.loads(self.backend.requests[-1].content)["chat_uuid"], "backend-chat-1"
        )
        count = len(self.backend.requests)
        for path, handle in (
            ("/mcp/foreign?api_key=private-key", identifier),
            ("/mcp/docs?api_key=foreign-key", identifier),
            ("/mcp/docs?api_key=private-key", "unknown"),
        ):
            result = await self.modern(
                call(message="bad", conversation_id=handle), path=path
            )
            self.assertTrue(result.json()["result"]["isError"])
        self.assertEqual(len(self.backend.requests), count)
        different_backend = await self.http(
            await self.new_app(api_base_url="https://other.example")
        )
        result = await self.modern(
            call(message="wrong backend", conversation_id=identifier),
            client=different_backend,
        )
        self.assertTrue(result.json()["result"]["isError"])
        self.assertEqual(len(self.backend.requests), count)
        self.conversations.sessions[identifier]["last_active"] = (
            datetime.now() - timedelta(seconds=SESSION_TIMEOUT + 1)
        )
        expired = await self.modern(call(message="expired", conversation_id=identifier))
        self.assertTrue(expired.json()["result"]["isError"])

    async def test_modern_no_id_starts_new_chats_and_retrieval_creates_no_conversation(
        self,
    ):
        for name, arguments in (
            ("biel_search", {"query": "setup"}),
            ("biel_get_document", {"document_id": "123:signed-reference"}),
        ):
            result = await self.modern(call(name, **arguments))
            self.assertFalse(result.json()["result"]["isError"])
        self.assertEqual(self.conversations.sessions, {})
        first = await self.modern(call(message="one"))
        second = await self.modern(call(message="two"))
        self.assertNotEqual(
            first.json()["result"]["structuredContent"],
            second.json()["result"]["structuredContent"],
        )
        self.assertEqual(self.backend.chat_count, 2)

    async def test_invalid_bodies_never_create_sessions_or_call_backend(self):
        for body in (
            [],
            None,
            "text",
            {},
            {"method": []},
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": []},
        ):
            response = await self.client.post("/mcp/docs", json=body)
            self.assertEqual(response.status_code, 400, response.text)
            self.assertNotIn("Mcp-Session-Id", response.headers)
        for body in (b"{", b"\xff"):
            response = await self.client.post(
                "/mcp/docs", content=body, headers={"Content-Type": "application/json"}
            )
            self.assertEqual(response.status_code, 400)
            self.assertEqual(response.json()["error"]["code"], -32700)
        self.assertEqual(self.sessions.sessions, {})
        self.assertEqual(self.backend.requests, [])

    async def test_failed_initialize_does_not_issue_application_session_header(self):
        for params in (
            {},
            {"protocolVersion": "2025-11-25", "capabilities": []},
            {"protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": []},
        ):
            body = initialize()
            body["params"] = params
            response = await self.client.post("/mcp/docs", json=body)
            self.assertIn("error", response.json())
            self.assertNotIn("Mcp-Session-Id", response.headers)
        self.assertEqual(self.sessions.sessions, {})

    async def test_tool_validation_and_backend_origin_gate_prevent_upstream_calls(self):
        _, headers = await self.established()
        for name, arguments in (
            ("biel_ai", {"message": "hi", "chat_uuid": "foreign"}),
            ("biel_ai", {"message": "hi", "restore_token": "foreign"}),
            ("biel_ai", {"message": "hi", "project_slug": "foreign"}),
            ("biel_search", {"query": "q", "limit": True}),
            ("biel_get_document", {"document_id": "../private"}),
        ):
            response = await self.client.post(
                "/mcp/docs", json=call(name, **arguments), headers=headers
            )
            self.assertTrue(
                response.json().get("error") or response.json()["result"]["isError"]
            )
        rejected = await self.client.post(
            "/mcp/docs?base_url=http://db:5432", json=initialize()
        )
        self.assertEqual(rejected.status_code, 400)
        self.assertEqual(self.backend.requests, [])

    async def test_get_uses_sdk_semantics_and_cors_exposes_legacy_header(self):
        _, headers = await self.established()
        disconnected = asyncio.Event()
        started = asyncio.Event()
        messages = []
        request_delivered = False

        async def receive():
            nonlocal request_delivered
            if not request_delivered:
                request_delivered = True
                return {"type": "http.request", "body": b"", "more_body": False}
            await disconnected.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            messages.append(message)
            if message["type"] == "http.response.start":
                started.set()
                disconnected.set()

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/mcp/docs",
            "raw_path": b"/mcp/docs",
            "root_path": "",
            "query_string": b"",
            "server": ("testserver", 80),
            "client": ("127.0.0.1", 123),
            "headers": [
                (b"host", b"testserver"),
                (b"accept", b"text/event-stream"),
                (b"mcp-session-id", headers["MCP-Session-Id"].encode()),
                (b"mcp-protocol-version", b"2025-11-25"),
            ],
        }
        task = asyncio.create_task(self.app(scope, receive, send))
        await asyncio.wait_for(started.wait(), timeout=3)
        await asyncio.wait_for(task, timeout=3)
        self.assertEqual(messages[0]["status"], 200)
        self.assertIn(
            b"text/event-stream", dict(messages[0]["headers"])[b"content-type"]
        )
        response = await self.client.post(
            "/mcp/docs",
            json=initialize(),
            headers={"Origin": "https://docs.example.com"},
        )
        self.assertEqual(response.headers["Access-Control-Allow-Origin"], "*")
        self.assertIn(
            "Mcp-Session-Id", response.headers["Access-Control-Expose-Headers"]
        )
        self.assertNotIn("Access-Control-Allow-Credentials", response.headers)

    async def test_concurrent_requests_keep_project_and_api_key_context_isolated(self):
        async def perform(slug, key):
            path = f"/mcp/{slug}?api_key={key}"
            _, headers = await self.established(path)
            return await self.client.post(
                path, json=call(message=slug), headers=headers
            )

        responses = await asyncio.gather(
            perform("alpha", "key-alpha"), perform("beta", "key-beta")
        )
        self.assertTrue(
            all(not response.json()["result"]["isError"] for response in responses)
        )
        self.assertEqual(
            {
                request.url.path.split("/")[4]: request.headers["Authorization"]
                for request in self.backend.requests
            },
            {"alpha": "Api-Key key-alpha", "beta": "Api-Key key-beta"},
        )

    async def test_sse_sdk_client_preserves_get_configuration_across_message_posts(
        self,
    ):
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.setblocking(False)
        port = listener.getsockname()[1]
        application = create_app(client=self.upstream)
        server = uvicorn.Server(
            uvicorn.Config(application, log_level="error", lifespan="on")
        )
        server.install_signal_handlers = lambda: None
        task = asyncio.create_task(server.serve(sockets=[listener]))
        try:
            for _ in range(200):
                if server.started:
                    break
                if task.done():
                    await task
                await asyncio.sleep(0.01)
            self.assertTrue(server.started)
            url = f"http://127.0.0.1:{port}/sse?project_slug=docs&api_key=sse-key&domain=https%3A%2F%2Fdocs.example.com&metadata=sse-docs"
            async with Client(
                sse_client(url),
                mode="legacy",
                client_info=Implementation(name="sse-client", version="1"),
            ) as client:
                tools = await client.list_tools()
                self.assertEqual(len(tools.tools), 3)
                response = await client.call_tool("biel_ai", {"message": "from SSE"})
                self.assertFalse(response.is_error, response)
                identifier = response.structured_content["conversation_id"]
                response = await client.call_tool(
                    "biel_ai",
                    {"message": "SSE follow-up", "conversation_id": identifier},
                )
                self.assertFalse(response.is_error, response)
                self.assertEqual(
                    response.structured_content["conversation_id"], identifier
                )
            request = self.backend.requests[-1]
            self.assertEqual(request.headers["Authorization"], "Api-Key sse-key")
            self.assertEqual(request.headers["X-MCP-Client-Name"], "sse-client")
            self.assertEqual(
                json.loads(request.content)["url"], "https://docs.example.com"
            )
            self.assertEqual(json.loads(request.content)["metadata"], "sse-docs")
            self.assertEqual(json.loads(request.content)["chat_uuid"], "backend-chat-1")
            self.assertEqual(
                request.headers["X-Chat-Restore-Token"], "secret-restore-backend-chat-1"
            )
        finally:
            server.should_exit = True
            await asyncio.wait_for(task, timeout=5)
            listener.close()
