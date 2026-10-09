import json
import logging
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import patch

import httpx
from biel_mcp.server import (
    CLIENT_IDENTITY_MAX_LENGTH,
    MCP_PROTOCOL_VERSION_V2,
    REQUEST_TIMEOUT,
    TOOLS,
    InMemorySessionStore,
    MCPLogFormatter,
    SessionManager,
    app,
    extract_client_info,
    format_biel_response,
    format_search_response,
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
            headers={},
            json=lambda: {
                "chat_uuid": "chat-1",
                "restore_token": "restore-1",
                "ai_message": {"message": "an answer", "sources": []},
            },
        )

    async def get(self, url, params=None, headers=None):
        type(self).calls.append({"url": url, "params": params, "headers": headers})
        return SimpleNamespace(
            status_code=200, headers={},
            json=lambda: {"results": [{"title": "Setup", "url": "https://docs.example.com/setup", "fragment": "Install the SDK."}]},
        )


class ConfigurationTest(TestCase):
    def test_tool_advertises_answer_and_search_modes(self):
        properties = TOOLS[0]["inputSchema"]["properties"]
        self.assertEqual(properties["mode"]["enum"], ["answer", "search"])
        self.assertEqual(properties["mode"]["default"], "search")

    def test_search_format_bounds_results_and_reports_empty_matches(self):
        data = {"results": [
            {"title": f"Result {n}", "url": f"https://example.com/{n}", "fragment": f"Snippet {n}"}
            for n in range(6)
        ]}
        text = format_search_response(data, 2)
        self.assertIn("showing 2 of 6", text)
        self.assertIn("Snippet 1", text)
        self.assertIn("https://example.com/1", text)
        self.assertNotIn("Snippet 2", text)
        self.assertIn("No matching documentation", format_search_response({"results": []}))

    def test_document_search_returns_full_text_and_citable_location(self):
        text = format_search_response({"results": [{
            "title": "manual.pdf", "source_type": "file", "url": None,
            "page": 3, "sheet": "Revenue", "fragment": "preview",
            "content": "The full instructions used to answer the question.",
        }]})
        self.assertIn("manual.pdf (file)", text)
        self.assertIn("page 3", text)
        self.assertIn("sheet Revenue", text)
        self.assertIn("The full instructions", text)
        self.assertNotIn("URL:", text)
    def test_standalone_logs_preserve_safe_upstream_diagnostics(self):
        record = logging.makeLogRecord({
            "msg": "MCP upstream request completed", "levelname": "ERROR",
            "event": "mcp.upstream_completed", "upstream_status": 404,
            "duration_ms": 27, "upstream_request_id": "request-1",
            "restore_token": "private-token", "chat_uuid": "private-chat",
        })
        data = json.loads(MCPLogFormatter().format(record))
        self.assertEqual(data["duration_ms"], 27)
        self.assertEqual(data["upstream_status"], 404)
        self.assertEqual(data["upstream_request_id"], "request-1")
        self.assertNotIn("private", json.dumps(data))

    def test_read_timeout_has_headroom_without_lengthening_connection_waits(self):
        self.assertEqual(REQUEST_TIMEOUT.read, 60)
        self.assertEqual(REQUEST_TIMEOUT.connect, 5)
        self.assertEqual(REQUEST_TIMEOUT.write, 10)
        self.assertEqual(REQUEST_TIMEOUT.pool, 5)

    def test_tool_does_not_advertise_unsecured_manual_continuation(self):
        self.assertNotIn("chat_uuid", TOOLS[0]["inputSchema"]["properties"])
        answer = format_biel_response({"chat_uuid": "chat-1", "ai_message": {"message": "answer"}})
        self.assertNotIn("chat-1", answer)

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
            return_value=(
                {"content": [{"type": "text", "text": "answer"}], "isError": False},
                "chat-1",
                "restore-1",
            ),
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
        self.assertEqual(session["restore_token"], "restore-1")


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
        await self.manager.record_chat_credentials(
            session_id, "chat-1", "restore-1"
        )

        await self.manager.record_client_info(session_id, "copilot", "1.4")

        session = await self.manager.get_session(session_id)
        self.assertEqual(session["chat_uuid"], "chat-1")
        self.assertEqual(session["restore_token"], "restore-1")

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
            {"message": "hi", "project_slug": "abc123", "mode": "answer"},
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

    async def test_session_restore_capability_authorizes_the_matching_chat(self):
        headers = await self.relay(
            {"chat_uuid": "chat-1", "restore_token": "restore-1"}
        )

        self.assertEqual(headers["X-Chat-Restore-Token"], "restore-1")
        self.assertEqual(RecordingAsyncClient.calls[-1]["json"]["chat_uuid"], "chat-1")

    async def test_uuid_only_session_requires_explicit_reconnection(self):
        result, chat, token = await query_biel_ai(
            {"message": "hi", "project_slug": "abc123", "mode": "answer"}, {"chat_uuid": "legacy-chat"}
        )
        self.assertTrue(result["isError"])
        self.assertEqual(RecordingAsyncClient.calls, [])
        self.assertIsNone(chat)
        self.assertIsNone(token)

    async def test_capability_is_not_relayed_for_a_caller_supplied_chat(self):
        result, _, _ = await query_biel_ai(
            {
                "message": "hi",
                "mode": "answer",
                "project_slug": "abc123",
                "chat_uuid": "other-chat",
            },
            {
                "base_url": "https://app.biel.ai",
                "chat_uuid": "chat-1",
                "restore_token": "restore-1",
            },
        )

        self.assertTrue(result["isError"])
        self.assertEqual(RecordingAsyncClient.calls, [])

    async def test_matching_explicit_uuid_keeps_existing_clients_authorized(self):
        result, _, _ = await query_biel_ai(
            {"message": "hi", "project_slug": "abc123", "chat_uuid": "chat-1", "mode": "answer"},
            {"chat_uuid": "chat-1", "restore_token": "restore-1"},
        )
        self.assertFalse(result["isError"])
        self.assertEqual(RecordingAsyncClient.calls[-1]["headers"]["X-Chat-Restore-Token"], "restore-1")


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
            "params": {"name": "biel_ai", "arguments": {"message": "hi", "mode": "answer"}},
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

    async def test_existing_client_continues_without_supplying_a_restore_token(self):
        """The capability stays internal; old MCP configs need no changes."""
        transport = httpx.ASGITransport(app=app)
        async with RealAsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            initialized = await self.post_mcp(client, self.initialize_body())
            session_id = initialized.headers["MCP-Session-Id"]

            await self.post_mcp(client, self.call_body(), session_id=session_id)
            await self.post_mcp(client, self.call_body(), session_id=session_id)

        second_relay = RecordingAsyncClient.calls[-1]
        self.assertEqual(second_relay["json"]["chat_uuid"], "chat-1")
        self.assertEqual(
            second_relay["headers"]["X-Chat-Restore-Token"], "restore-1"
        )

    async def test_search_between_answers_preserves_the_conversation(self):
        async with RealAsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            initialized = await self.post_mcp(client, self.initialize_body())
            session_id = initialized.headers["MCP-Session-Id"]
            await self.post_mcp(client, self.call_body(), session_id)
            search = self.call_body()
            search["params"]["arguments"]["mode"] = "search"
            response = await self.post_mcp(client, search, session_id)
            request = RecordingAsyncClient.calls[-1]
            self.assertTrue(request["url"].endswith("/search/"))
            self.assertNotIn("X-Chat-Restore-Token", request["headers"])
            self.assertNotIn("chat_uuid", request["params"])
            self.assertEqual(request["params"]["q"], search["params"]["arguments"]["message"])
            self.assertFalse(response.json()["result"]["isError"])
            self.assertIn("Install the SDK", response.json()["result"]["content"][0]["text"])
            await self.post_mcp(client, self.call_body(), session_id)
        self.assertEqual(RecordingAsyncClient.calls[-1]["json"]["chat_uuid"], "chat-1")
        self.assertEqual(RecordingAsyncClient.calls[-1]["headers"]["X-Chat-Restore-Token"], "restore-1")

    async def test_initialized_notification_is_accepted_without_a_response_body(self):
        async with RealAsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            initialized = await self.post_mcp(client, self.initialize_body())
            response = await self.post_mcp(client, {"jsonrpc": "2.0", "method": "notifications/initialized"}, initialized.headers["MCP-Session-Id"])
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.content, b"")

    async def test_expired_session_returns_404_without_creating_a_conversation(self):
        session = await app.state.session_manager.create_session(project_slug="abc123")
        await app.state.session_manager.delete_session(session)
        async with RealAsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await self.post_mcp(client, self.call_body(), session)
        self.assertEqual(response.status_code, 404)
        self.assertEqual(RecordingAsyncClient.calls, [])
        self.assertIsNone(await app.state.session_manager.get_session(session))

    async def test_session_cannot_be_used_at_another_project_path(self):
        session = await app.state.session_manager.create_session(project_slug="different-project")
        async with RealAsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            for body in (self.initialize_body(), self.call_body()):
                response = await self.post_mcp(client, body, session)
                self.assertEqual(response.status_code, 404)
            headers = {"MCP-Session-Id": session}
            for method in (client.get, client.delete):
                response = await method("/mcp/abc123", headers=headers)
                self.assertEqual(response.status_code, 404)
        self.assertEqual(RecordingAsyncClient.calls, [])
        self.assertIsNotNone(await app.state.session_manager.get_session(session))

    async def test_tool_cannot_override_connection_project(self):
        session = await app.state.session_manager.create_session(project_slug="abc123")
        async with RealAsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            body = self.call_body()
            body["params"]["arguments"]["project_slug"] = "another-project"
            response = await self.post_mcp(client, body, session)
        self.assertTrue(response.json()["result"]["isError"])
        self.assertEqual(RecordingAsyncClient.calls, [])
        stored = await app.state.session_manager.get_session(session)
        self.assertFalse(stored.get("chat_uuid"))

    async def test_reconnected_client_cannot_continue_with_a_previous_uuid(self):
        async with RealAsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            initialized = await self.post_mcp(client, self.initialize_body())
            body = self.call_body()
            body["params"]["arguments"]["chat_uuid"] = "chat-1"
            response = await self.post_mcp(client, body, initialized.headers["MCP-Session-Id"])
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["result"]["isError"])
        self.assertEqual(RecordingAsyncClient.calls, [])


class SearchModeTest(IsolatedAsyncioTestCase):
    async def test_omitting_mode_searches_without_generating_an_answer(self):
        RecordingAsyncClient.calls = []
        with patch("biel_mcp.server.httpx.AsyncClient", RecordingAsyncClient):
            result, chat, token = await query_biel_ai({"message": "SDK auth"}, {"project_slug": "docs"})
        self.assertFalse(result["isError"])
        self.assertIsNone(chat)
        self.assertIsNone(token)
        self.assertTrue(RecordingAsyncClient.calls[0]["url"].endswith("/search/"))

    async def test_search_uses_get_with_project_credentials_without_chat_capabilities(self):
        RecordingAsyncClient.calls = []
        with patch("biel_mcp.server.httpx.AsyncClient", RecordingAsyncClient):
            result, chat, token = await query_biel_ai(
                {"message": "SDK auth", "mode": "search", "limit": 2},
                {"project_slug": "docs", "api_key": "private-key", "domain": "https://docs.example.com", "chat_uuid": "old-chat"},
            )
        self.assertFalse(result["isError"])
        self.assertIsNone(chat)
        self.assertIsNone(token)
        request = RecordingAsyncClient.calls[0]
        self.assertTrue(request["url"].endswith("/projects/docs/search/"))
        self.assertEqual(request["params"], {"q": "SDK auth", "url": "https://docs.example.com", "content_scope": "all"})
        self.assertEqual(request["headers"]["Authorization"], "Api-Key private-key")
        self.assertEqual(request["headers"]["X-Biel-Source"], "mcp")
        self.assertNotIn("X-Chat-Restore-Token", request["headers"])

    async def test_invalid_mode_or_search_limit_fails_before_any_api_call(self):
        RecordingAsyncClient.calls = []
        for options in ({"mode": "unknown"}, {"mode": None}, {"mode": "search", "limit": 0}, {"mode": "search", "limit": 21}, {"mode": "search", "limit": True}, {"mode": "search", "limit": "5"}):
            with self.subTest(options=options), patch("biel_mcp.server.httpx.AsyncClient", RecordingAsyncClient):
                result, _, _ = await query_biel_ai({"message": "setup", "project_slug": "docs", **options})
            self.assertTrue(result["isError"])
        self.assertEqual(RecordingAsyncClient.calls, [])


class UpstreamFailuresTest(IsolatedAsyncioTestCase):
    async def query(self, transport, mode="answer"):
        client = RealAsyncClient(transport=transport)
        with patch("biel_mcp.server.httpx.AsyncClient", return_value=client):
            return await query_biel_ai({"message": "private question", "project_slug": "docs", "mode": mode})

    async def test_search_http_errors_and_timeouts_are_marked_as_tool_failures(self):
        for code in (403, 404, 429, 500):
            with self.subTest(code=code):
                result, _, _ = await self.query(
                    httpx.MockTransport(lambda request, status=code: httpx.Response(status, text="private-detail")),
                    mode="search",
                )
                self.assertTrue(result["isError"])
                self.assertNotIn("private-detail", str(result))
        with self.assertLogs("biel-mcp", level="ERROR") as captured:
            result, _, _ = await self.query(httpx.MockTransport(raise_read_timeout), mode="search")
        self.assertTrue(result["isError"])
        self.assertEqual(captured.records[-1].mode, "search")
        self.assertEqual(captured.records[-1].error_kind, "timeout")

    async def test_malformed_search_payload_is_not_presented_as_a_successful_answer(self):
        result, _, _ = await self.query(
            httpx.MockTransport(lambda request: httpx.Response(200, json={"ai_message": {"message": "wrong-contract"}})),
            mode="search",
        )
        self.assertTrue(result["isError"])
        self.assertNotIn("wrong-contract", str(result))

    async def test_http_errors_are_tool_errors_and_do_not_echo_backend_payloads(self):
        for status in (403, 404, 429, 500):
            with self.subTest(status=status):
                transport = httpx.MockTransport(lambda request, code=status: httpx.Response(code, text="secret-backend-payload", headers={"x-request-id": "upstream-id"}))
                with self.assertLogs("biel-mcp", level="ERROR") as captured:
                    result, chat, token = await self.query(transport)
                self.assertTrue(result["isError"])
                self.assertIsNone(chat)
                self.assertIsNone(token)
                self.assertNotIn("secret-backend-payload", str(result))
                self.assertNotIn("private question", str(captured.output))
                record = captured.records[-1]
                self.assertEqual(record.upstream_status, status)
                self.assertEqual(record.upstream_request_id, "upstream-id")
                self.assertEqual(record.error_kind, "upstream_error")
                self.assertGreaterEqual(record.duration_ms, 0)

    async def test_read_timeout_is_reported_separately_from_http_rejections(self):
        with self.assertLogs("biel-mcp", level="ERROR") as captured:
            result, _, _ = await self.query(httpx.MockTransport(raise_read_timeout))
        self.assertTrue(result["isError"])
        self.assertEqual(captured.records[-1].error_kind, "timeout")
        self.assertIsNone(captured.records[-1].upstream_status)
        self.assertNotIn("secret-backend-payload", str(result))


def raise_read_timeout(request):
    raise httpx.ReadTimeout("secret-backend-payload")
