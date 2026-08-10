import json
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import patch

import httpx

from biel_mcp.server import (
    CLIENT_IDENTITY_MAX_LENGTH,
    MCP_PROTOCOL_VERSION_V2,
    InMemorySessionStore,
    SessionManager,
    app,
    extract_client_info,
    header_safe,
    normalize_base_url,
    query_biel_ai,
    resolve_base_url,
    validate_biel_request,
)

# Bound before any patching: the stand-in below replaces the attribute on the
# httpx module itself, which is the same object these tests drive the app with.
RealAsyncClient = httpx.AsyncClient


class RecordingAsyncClient:
    """Stands in for ``httpx.AsyncClient``, capturing what would be relayed."""

    calls = []

    def __init__(self, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def post(self, url, json=None, headers=None):
        type(self).calls.append({"url": url, "json": json, "headers": headers})
        return SimpleNamespace(
            status_code=200,
            json=lambda: {
                "chat_uuid": "chat-1",
                "ai_message": {"message": "an answer", "sources": []},
            },
        )


class ConfigurationTest(TestCase):
    def test_normalize_base_url_strips_whitespace_and_trailing_slash(self):
        self.assertEqual(normalize_base_url(" https://app.biel.ai/ "), "https://app.biel.ai")

    def test_requested_base_url_must_be_allowed(self):
        allowed = frozenset({"https://app.biel.ai"})

        self.assertEqual(
            resolve_base_url("https://app.biel.ai/", allowed, "https://default.example"),
            "https://app.biel.ai",
        )
        self.assertIsNone(
            resolve_base_url("https://untrusted.example", allowed, "https://default.example")
        )

    def test_project_slug_cannot_reshape_the_api_path(self):
        self.assertEqual(
            validate_biel_request({"message": "hello", "project_slug": "../admin"}),
            "Invalid project slug",
        )


class SessionManagerTest(IsolatedAsyncioTestCase):
    async def test_session_roundtrip(self):
        sessions = SessionManager()
        session_id = await sessions.create_session(project_slug="docs")

        session = await sessions.get_session(session_id)

        self.assertEqual(session["project_slug"], "docs")


class ApplicationTest(IsolatedAsyncioTestCase):
    async def test_initialize_creates_a_session(self):
        app.state.session_manager = SessionManager()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/mcp/docs",
                content=json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "initialize",
                        "params": {},
                    }
                ),
                headers={"MCP-Protocol-Version": MCP_PROTOCOL_VERSION_V2},
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["MCP-Session-Id"])
        self.assertEqual(response.json()["result"]["serverInfo"]["name"], "biel-ai-mcp")

    async def test_tool_call_records_the_conversation_on_its_session(self):
        sessions = SessionManager()
        app.state.session_manager = sessions
        session_id = await sessions.create_session(project_slug="docs")
        transport = httpx.ASGITransport(app=app)
        with patch(
            "biel_mcp.server.query_biel_ai",
            return_value=({"type": "text", "text": "answer"}, "chat-1"),
        ):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                response = await client.post(
                    "/mcp/docs",
                    content=json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "id": 2,
                            "method": "tools/call",
                            "params": {
                                "name": "biel_ai",
                                "arguments": {"message": "hello"},
                            },
                        }
                    ),
                    headers={
                        "MCP-Protocol-Version": MCP_PROTOCOL_VERSION_V2,
                        "MCP-Session-Id": session_id,
                    },
                )

        self.assertEqual(response.status_code, 200)
        session = await sessions.get_session(session_id)
        self.assertEqual(session["chat_uuid"], "chat-1")


class HeaderSafeTest(TestCase):
    def test_plain_value_survives(self):
        self.assertEqual(header_safe("claude-code"), "claude-code")

    def test_newlines_cannot_smuggle_a_second_header(self):
        """The failure this exists to prevent: a client naming itself such that
        the relayed request grows headers of its own."""
        self.assertEqual(
            header_safe("evil\r\nX-Client-IP: 10.0.0.1"), "evil X-Client-IP: 10.0.0.1"
        )

    def test_non_latin1_is_dropped_rather_than_failing_the_query(self):
        self.assertEqual(header_safe("cursor✨"), "cursor")

    def test_value_is_capped(self):
        self.assertEqual(len(header_safe("x" * 500)), CLIENT_IDENTITY_MAX_LENGTH)

    def test_non_strings_are_no_identity_at_all(self):
        for value in (None, 42, {"name": "x"}, ["x"]):
            with self.subTest(value=value):
                self.assertEqual(header_safe(value), "")


class ExtractClientInfoTest(TestCase):
    def test_reads_the_handshakes_name_and_version(self):
        self.assertEqual(
            extract_client_info(
                {"params": {"clientInfo": {"name": "claude-code", "version": "2.1"}}}
            ),
            ("claude-code", "2.1"),
        )

    def test_sanitizes_what_it_reads(self):
        self.assertEqual(
            extract_client_info({"params": {"clientInfo": {"name": "a\nb"}}}),
            ("a b", ""),
        )

    def test_a_handshake_without_client_info_names_nobody(self):
        for data in (
            {},
            {"params": {}},
            {"params": None},
            {"params": {"clientInfo": None}},
            {"params": {"clientInfo": "claude-code"}},
            None,
        ):
            with self.subTest(data=data):
                self.assertEqual(extract_client_info(data), ("", ""))


class SessionClientIdentityTest(IsolatedAsyncioTestCase):
    def setUp(self):
        self.manager = SessionManager()

    async def test_identity_given_at_creation_is_readable_back(self):
        session_id = await self.manager.create_session(
            project_slug="abc123", client_name="claude-code", client_version="2.1"
        )

        session = await self.manager.get_session(session_id)

        self.assertEqual(session["client_name"], "claude-code")
        self.assertEqual(session["client_version"], "2.1")

    async def test_a_session_created_without_a_handshake_names_nobody(self):
        session_id = await self.manager.create_session(project_slug="abc123")

        session = await self.manager.get_session(session_id)

        self.assertEqual(session["client_name"], "")
        self.assertEqual(session["client_version"], "")

    async def test_reinitializing_attaches_identity_to_a_live_session(self):
        session_id = await self.manager.create_session(project_slug="abc123")

        await self.manager.record_client_info(session_id, "copilot", "1.4")

        session = await self.manager.get_session(session_id)
        self.assertEqual(session["client_name"], "copilot")
        self.assertEqual(session["client_version"], "1.4")

    async def test_recording_identity_leaves_the_conversation_threaded(self):
        """A whole-record write at initialize must not drop the ``chat_uuid``
        a tool call already claimed."""
        session_id = await self.manager.create_session(project_slug="abc123")
        await self.manager.record_chat_uuid(session_id, "chat-1")

        await self.manager.record_client_info(session_id, "copilot", "1.4")

        session = await self.manager.get_session(session_id)
        self.assertEqual(session["chat_uuid"], "chat-1")

    async def test_an_anonymous_handshake_does_not_erase_a_known_client(self):
        session_id = await self.manager.create_session(
            project_slug="abc123", client_name="claude-code", client_version="2.1"
        )

        await self.manager.record_client_info(session_id, "", "")

        session = await self.manager.get_session(session_id)
        self.assertEqual(session["client_name"], "claude-code")

    async def test_an_unchanged_identity_is_not_written_back(self):
        store = InMemorySessionStore()
        manager = SessionManager(store=store)
        session_id = await manager.create_session(
            project_slug="abc123", client_name="claude-code", client_version="2.1"
        )
        with patch.object(
            store, "set", side_effect=AssertionError("rewrote an unchanged record")
        ):
            await manager.record_client_info(session_id, "claude-code", "2.1")

    async def test_recording_against_an_unknown_session_records_nothing(self):
        await self.manager.record_client_info("nope", "claude-code", "2.1")

        self.assertIsNone(await self.manager.get_session("nope"))


class RelayedHeadersTest(IsolatedAsyncioTestCase):
    def setUp(self):
        RecordingAsyncClient.calls = []
        patcher = patch("biel_mcp.server.httpx.AsyncClient", RecordingAsyncClient)
        patcher.start()
        self.addCleanup(patcher.stop)

    async def relay(self, defaults):
        await query_biel_ai(
            {"message": "hi", "project_slug": "abc123"},
            {"base_url": "https://app.biel.ai", **defaults},
        )
        return RecordingAsyncClient.calls[-1]["headers"]

    async def test_handshake_identity_labels_the_query(self):
        headers = await self.relay(
            {"client_name": "claude-code", "client_version": "2.1"}
        )

        self.assertEqual(headers["X-MCP-Client-Name"], "claude-code")
        self.assertEqual(headers["X-MCP-Client-Version"], "2.1")

    async def test_user_agent_travels_as_the_fallback_signal(self):
        headers = await self.relay({"user_agent": "node/20"})

        self.assertEqual(headers["X-MCP-Client-UA"], "node/20")

    async def test_absent_identity_sends_no_empty_headers(self):
        headers = await self.relay({})

        for header in (
            "X-MCP-Client-Name",
            "X-MCP-Client-Version",
            "X-MCP-Client-UA",
        ):
            with self.subTest(header=header):
                self.assertNotIn(header, headers)

    async def test_the_relay_still_declares_itself_as_mcp(self):
        headers = await self.relay({"client_name": "claude-code"})

        self.assertEqual(headers["X-Biel-Source"], "mcp")


class ClientIdentityOverTheTransportTest(IsolatedAsyncioTestCase):
    """The whole path: handshake, then a tool call carrying what it announced."""

    def setUp(self):
        app.state.session_manager = SessionManager()
        RecordingAsyncClient.calls = []
        patcher = patch("biel_mcp.server.httpx.AsyncClient", RecordingAsyncClient)
        patcher.start()
        self.addCleanup(patcher.stop)

    async def post_mcp(self, client, body, session_id=None, user_agent=None):
        headers = {
            "Content-Type": "application/json",
            "MCP-Protocol-Version": MCP_PROTOCOL_VERSION_V2,
        }
        if session_id:
            headers["MCP-Session-Id"] = session_id
        if user_agent:
            headers["User-Agent"] = user_agent
        return await client.post(
            "/mcp/abc123", content=json.dumps(body), headers=headers
        )

    def initialize_body(self, name="claude-code", version="2.1"):
        return {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": MCP_PROTOCOL_VERSION_V2,
                "capabilities": {},
                "clientInfo": {"name": name, "version": version},
            },
        }

    def call_body(self):
        return {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "biel_ai", "arguments": {"message": "hi"}},
        }

    async def test_the_client_named_at_initialize_labels_a_later_tool_call(self):
        transport = httpx.ASGITransport(app=app)
        async with RealAsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            initialized = await self.post_mcp(
                client, self.initialize_body(), user_agent="claude-code/2.1"
            )
            session_id = initialized.headers["MCP-Session-Id"]

            await self.post_mcp(
                client,
                self.call_body(),
                session_id=session_id,
                user_agent="claude-code/2.1",
            )

        headers = RecordingAsyncClient.calls[-1]["headers"]
        self.assertEqual(headers["X-MCP-Client-Name"], "claude-code")
        self.assertEqual(headers["X-MCP-Client-Version"], "2.1")
        self.assertEqual(headers["X-MCP-Client-UA"], "claude-code/2.1")

    async def test_a_client_that_names_nobody_still_gets_its_query_relayed(self):
        transport = httpx.ASGITransport(app=app)
        async with RealAsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            body = self.initialize_body()
            del body["params"]["clientInfo"]
            initialized = await self.post_mcp(client, body)
            session_id = initialized.headers["MCP-Session-Id"]

            response = await self.post_mcp(
                client, self.call_body(), session_id=session_id
            )

        self.assertEqual(response.status_code, 200)
        headers = RecordingAsyncClient.calls[-1]["headers"]
        self.assertNotIn("X-MCP-Client-Name", headers)
