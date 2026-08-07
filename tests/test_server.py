import json
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import patch

import httpx

from biel_mcp.server import (
    MCP_PROTOCOL_VERSION_V2,
    SessionManager,
    app,
    normalize_base_url,
    resolve_base_url,
    validate_biel_request,
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
