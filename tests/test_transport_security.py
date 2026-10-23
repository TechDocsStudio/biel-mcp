import os
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import patch

import httpx

from biel_mcp.server import InMemorySessionStore, SessionManager, app, main
from biel_mcp.transport_security import TransportPolicy, normalize_origin


class OriginConfigurationTest(TestCase):
    def test_origin_normalization_and_malformed_values(self):
        self.assertEqual(
            normalize_origin("https://DOCS.example:443"), ("https", "docs.example", 443)
        )
        for value in (
            "null",
            "*",
            "https://docs.example/path",
            "https://user@docs.example",
            "https://docs.example?x=1",
            "https://docs.example#x",
            "https://docs.example:invalid",
            "https://docs.example:0",
            "https://*.example",
            "https://docs.example https://evil.example",
            "https://docs.example\\evil",
            "\x00https://docs.example",
            "https://docs.example\x7f",
            "https://docs.example?",
            "https://docs.example#",
            "https://docs.example:",
        ):
            with self.subTest(value=value):
                self.assertIsNone(normalize_origin(value))

    def test_wildcard_hosts_and_origins_cannot_disable_protection(self):
        for hosts, origins in (
            ({"*"}, set()),
            ({".example.com"}, set()),
            ({"localhost:8000"}, set()),
            ({"localhost"}, {"*"}),
            ({"localhost"}, {"null"}),
        ):
            with self.subTest(hosts=hosts, origins=origins):
                with self.assertRaises(ValueError):
                    TransportPolicy(frozenset(hosts), frozenset(origins))

    def test_loopback_bind_default_and_explicit_remote_bind(self):
        saved = app.state.transport_policy
        self.addCleanup(setattr, app.state, "transport_policy", saved)
        for env, expected in (
            ({}, "127.0.0.1"),
            (
                {
                    "BIEL_BIND_HOST": "0.0.0.0",
                    "BIEL_ALLOWED_HOSTS": "mcp.example,localhost",
                    "BIEL_ALLOWED_ORIGINS": "https://docs.example",
                },
                "0.0.0.0",
            ),
        ):
            with (
                self.subTest(env=env),
                patch.dict(os.environ, env, clear=True),
                patch("biel_mcp.server.uvicorn.run") as run,
            ):
                main()
                self.assertEqual(run.call_args.kwargs["host"], expected)


class TransportSecurityTest(IsolatedAsyncioTestCase):
    def setUp(self):
        self.policy = app.state.transport_policy
        self.sessions = app.state.session_manager
        self.allowed = app.state.allowed_base_urls
        app.state.transport_policy = TransportPolicy(
            frozenset({"localhost", "mcp.example", "::1"}),
            frozenset({"https://docs.example"}),
        )
        self.store = InMemorySessionStore()
        app.state.session_manager = SessionManager(store=self.store)
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://localhost"
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        app.state.transport_policy = self.policy
        app.state.session_manager = self.sessions
        app.state.allowed_base_urls = self.allowed

    async def initialize(self, path="/mcp/docs", headers=None):
        return await self.client.post(
            path,
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {},
                    "clientInfo": {"name": "test-client", "version": "1"},
                },
            },
        )

    async def test_disallowed_origin_rejected_before_session_creation_every_alias(self):
        for path in ("/mcp/docs", "/v2/docs/mcp", "/sse?project_slug=docs"):
            with self.subTest(path=path):
                response = await self.initialize(
                    path, {"Origin": "https://evil.example"}
                )
                self.assertEqual(response.status_code, 403)
                self.assertNotIn("MCP-Session-Id", response.headers)
        self.assertEqual(self.store.sessions, {})

    async def test_allowed_origin_and_nonbrowser_client_initialize(self):
        for headers in (
            {},
            {"Origin": "https://docs.example"},
            {"Origin": "https://DOCS.example:443"},
        ):
            with self.subTest(headers=headers):
                response = await self.initialize(headers=headers)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertIn("MCP-Session-Id", response.headers)

    async def test_malicious_host_cannot_initialize_even_without_origin(self):
        for host in (
            "evil.example",
            "localhost.evil.example",
            "localhost@evil.example",
            "localhost,evil.example",
            "localhost/path",
        ):
            with self.subTest(host=host):
                self.assertEqual(
                    (await self.initialize(headers={"Host": host})).status_code, 403
                )
        self.assertEqual(self.store.sessions, {})

    async def test_ipv6_loopback_host_and_explicit_browser_origin(self):
        app.state.transport_policy = TransportPolicy(
            frozenset({"::1"}), frozenset({"https://[::1]"})
        )
        response = await self.initialize(
            headers={"Host": "[::1]:7832", "Origin": "https://[::1]:443"}
        )
        self.assertEqual(response.status_code, 200, response.text)

    async def test_duplicate_host_or_origin_headers_are_rejected(self):
        for headers in (
            [("Host", "localhost"), ("Host", "evil.example")],
            [("Origin", "https://docs.example"), ("Origin", "https://evil.example")],
        ):
            with self.subTest(headers=headers):
                self.assertEqual(
                    (await self.initialize(headers=headers)).status_code, 403
                )

    async def test_preflight_origin_policy_precedes_cors(self):
        for origin, expected in (
            ("https://evil.example", 403),
            ("https://docs.example", 200),
        ):
            response = await self.client.options(
                "/mcp/docs",
                headers={"Origin": origin, "Access-Control-Request-Method": "POST"},
            )
            self.assertEqual(response.status_code, expected)

    async def test_get_and_session_delete_also_check_origin(self):
        response = await self.initialize()
        session = response.headers["MCP-Session-Id"]
        for method in ("GET", "DELETE"):
            response = await self.client.request(
                method,
                "/mcp/docs",
                headers={"Origin": "https://evil.example", "MCP-Session-Id": session},
            )
            self.assertEqual(response.status_code, 403)
        self.assertIsNotNone(await app.state.session_manager.get_session(session))

    async def test_upstream_allowlist_is_separate_from_browser_origin(self):
        response = await self.initialize(
            "/mcp/docs?base_url=https://docs.example",
            {"Origin": "https://docs.example"},
        )
        self.assertEqual(response.status_code, 400)

    async def test_invalid_origin_value_is_not_accepted(self):
        for origin in (
            "null",
            "",
            "https://docs.example/path",
            "https://user@docs.example",
        ):
            with self.subTest(origin=origin):
                self.assertEqual(
                    (await self.initialize(headers={"Origin": origin})).status_code, 403
                )
