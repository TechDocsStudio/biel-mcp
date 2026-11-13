"""Conversation policy exercised through the real SDK tool dispatcher."""

import asyncio
import base64
import json
from datetime import datetime, timedelta
from unittest import IsolatedAsyncioTestCase

import httpx
from mcp import Client
from mcp.types import Implementation

from biel_mcp import api
from biel_mcp.config import SESSION_TIMEOUT
from biel_mcp.context import RequestContext, current_request
from biel_mcp.sessions import InMemorySessionStore, SessionManager
from biel_mcp.tools import create_server


def restore_token(timestamp):
    return base64.urlsafe_b64encode(
        b"\x80" + timestamp.to_bytes(8, "big") + b"synthetic-token"
    ).decode()


class ApplicationStoreTest(IsolatedAsyncioTestCase):
    async def test_same_chat_renews_only_to_a_newer_capability(self):
        manager = SessionManager()
        identifier = await manager.create_session("docs")
        await manager.record_chat_credentials(identifier, "chat", restore_token(100))
        await manager.record_chat_credentials(identifier, "chat", restore_token(200))
        await manager.record_chat_credentials(identifier, "chat", restore_token(150))
        refused = await manager.record_chat_credentials(
            identifier, "other", restore_token(300)
        )
        self.assertEqual(refused, ("chat", restore_token(200)))
        self.assertEqual(
            (await manager.get_session(identifier))["restore_token"], restore_token(200)
        )

    async def test_get_does_not_revive_an_expired_session(self):
        store = InMemorySessionStore()
        manager = SessionManager(store=store)
        identifier = await manager.create_session("docs")
        store.sessions[identifier]["last_active"] = datetime.now() - timedelta(
            seconds=SESSION_TIMEOUT + 1
        )
        self.assertIsNone(await manager.get_session(identifier))
        self.assertNotIn(identifier, store.sessions)

    async def test_simultaneous_claims_keep_one_complete_pair(self):
        manager = SessionManager()
        identifier = await manager.create_session("docs")
        pairs = await asyncio.gather(
            manager.record_chat_credentials(identifier, "chat-a", "token-a"),
            manager.record_chat_credentials(identifier, "chat-b", "token-b"),
        )
        self.assertEqual(pairs, [("chat-a", "token-a"), ("chat-a", "token-a")])


class SdkToolTest(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.requests = []
        self.manager = SessionManager()
        self.response = {
            "chat_uuid": "chat-1",
            "restore_token": restore_token(100),
            "ai_message": {"message": "Answer", "sources": []},
        }

        def upstream(request):
            self.requests.append(request)
            if request.url.path.endswith("/search/"):
                return httpx.Response(200, json={"results": []})
            return httpx.Response(200, json=self.response)

        self.upstream = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        self.bound = RequestContext(
            defaults={
                "project_slug": "docs",
                "base_url": "https://app.biel.ai",
                "api_key": "key-a",
            },
            conversations=self.manager,
        )
        self.token = current_request.set(self.bound)

    async def asyncTearDown(self):
        current_request.reset(self.token)
        await self.upstream.aclose()

    def client(self):
        return Client(
            create_server(client=self.upstream),
            client_info=Implementation(name="Test editor", version="1.2"),
        )

    async def test_modern_follow_up_crosses_servers_without_exposing_restore_token(
        self,
    ):
        async with self.client() as client:
            first = await client.call_tool("biel_ai", {"message": "First"})
        identifier = first.structured_content["conversation_id"]
        self.assertIn(identifier, first.content[-1].text)
        self.assertNotIn(self.response["restore_token"], first.model_dump_json())
        self.assertNotIn("chat-1", first.model_dump_json())
        async with self.client() as client:
            follow = await client.call_tool(
                "biel_ai", {"message": "Next", "conversation_id": identifier}
            )
        self.assertFalse(follow.is_error)
        self.assertEqual(follow.structured_content["conversation_id"], identifier)
        request = self.requests[-1]
        self.assertEqual(json.loads(request.content)["chat_uuid"], "chat-1")
        self.assertEqual(request.headers["X-Chat-Restore-Token"], restore_token(100))
        self.assertEqual(request.headers["X-MCP-Client-Name"], "Test editor")
        self.assertFalse(self.upstream.is_closed)

    async def test_modern_handle_rejects_another_authority_and_expiry(self):
        async with self.client() as client:
            first = await client.call_tool("biel_ai", {"message": "First"})
            identifier = first.structured_content["conversation_id"]
            for changed in (
                {"api_key": "key-b"},
                {"project_slug": "other"},
                {"base_url": "https://other.example"},
            ):
                before = dict(self.bound.defaults)
                self.bound.defaults.update(changed)
                rejected = await client.call_tool(
                    "biel_ai", {"message": "Next", "conversation_id": identifier}
                )
                self.assertTrue(rejected.is_error)
                self.bound.defaults = before
            self.assertEqual(len(self.requests), 1)
            await self.manager.delete_session(identifier)
            expired = await client.call_tool(
                "biel_ai", {"message": "Next", "conversation_id": identifier}
            )
            self.assertTrue(expired.is_error)
            self.assertEqual(len(self.requests), 1)

    async def test_modern_handle_restores_domain_and_metadata_for_follow_up(self):
        async with self.client() as client:
            first = await client.call_tool(
                "biel_ai",
                {
                    "message": "First",
                    "domain": "https://docs.example.com",
                    "metadata": "original-source",
                },
            )
            identifier = first.structured_content["conversation_id"]
        async with self.client() as client:
            follow = await client.call_tool(
                "biel_ai",
                {
                    "message": "Next",
                    "conversation_id": identifier,
                },
            )
            explicit = await client.call_tool(
                "biel_ai",
                {
                    "message": "Override",
                    "conversation_id": identifier,
                    "domain": "https://other-docs.example.com",
                    "metadata": "override-source",
                },
            )
        self.assertFalse(follow.is_error)
        resumed = json.loads(self.requests[1].content)
        self.assertEqual(resumed["url"], "https://docs.example.com")
        self.assertEqual(resumed["metadata"], "original-source")
        self.assertFalse(explicit.is_error)
        overridden = json.loads(self.requests[2].content)
        self.assertEqual(overridden["url"], "https://other-docs.example.com")
        self.assertEqual(overridden["metadata"], "override-source")

    async def test_malformed_handle_is_rejected_before_store_access(self):
        class UnavailableManager(SessionManager):
            async def get_session(self, identifier):
                raise AssertionError("Malformed handle reached storage")

        self.bound.conversations = UnavailableManager()
        async with self.client() as client:
            result = await client.call_tool(
                "biel_ai", {"message": "Next", "conversation_id": "bad\ncache key"}
            )
        self.assertTrue(result.is_error)
        self.assertEqual(self.requests, [])

    async def test_modern_missing_capability_is_not_a_successful_unthreaded_chat(self):
        self.response.pop("restore_token")
        async with self.client() as client:
            result = await client.call_tool("biel_ai", {"message": "First"})
        self.assertTrue(result.is_error)
        self.assertIsNone(result.structured_content)
        self.assertEqual(self.manager._store.sessions, {})

    async def test_implicit_session_preserves_content_and_recorded_identity(self):
        identifier = await self.manager.create_session(
            "docs", api_key="key-a", client_name="Original editor", client_version="2"
        )
        self.bound.session_id = identifier
        self.bound.sessions = self.manager
        self.bound.defaults.update(
            {"client_name": "Original editor", "client_version": "2"}
        )
        async with self.client() as client:
            first = await client.call_tool("biel_ai", {"message": "First"})
            self.bound.defaults.update(await self.manager.get_session(identifier))
            second = await client.call_tool(
                "biel_ai", {"message": "Next", "conversation_id": identifier}
            )
            wrong = await client.call_tool(
                "biel_ai", {"message": "Next", "conversation_id": "different"}
            )
        self.assertFalse(first.is_error)
        self.assertEqual(len(first.content), 1)
        self.assertIsNone(first.structured_content)
        self.assertFalse(second.is_error)
        self.assertTrue(wrong.is_error)
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(
            self.requests[-1].headers["X-MCP-Client-Name"], "Original editor"
        )
        self.assertEqual(json.loads(self.requests[-1].content)["chat_uuid"], "chat-1")

    async def test_search_never_creates_a_conversation_and_strict_schema_rejects_bool_limit(
        self,
    ):
        async with self.client() as client:
            tools = (await client.list_tools()).tools
            self.assertEqual(
                [tool.name for tool in tools],
                ["biel_ai", "biel_search", "biel_get_document"],
            )
            self.assertTrue(tools[1].annotations.read_only_hint)
            self.assertNotIn("conversation_id", tools[1].input_schema["properties"])
            search = await client.call_tool("biel_search", {"query": "Setup"})
            rejected = await client.call_tool(
                "biel_search", {"query": "Setup", "limit": True}
            )
            unsupported = await client.call_tool(
                "biel_ai", {"message": "First", "restore_token": "untrusted"}
            )
        self.assertFalse(search.is_error)
        self.assertTrue(rejected.is_error)
        self.assertTrue(unsupported.is_error)
        self.assertEqual(len(self.requests), 1)
        self.assertNotIn("X-Chat-Restore-Token", self.requests[0].headers)
        self.assertEqual(self.manager._store.sessions, {})

    async def test_upstream_cancellation_propagates_and_keeps_borrowed_client_open(
        self,
    ):
        started = asyncio.Event()

        async def slow(request):
            started.set()
            await asyncio.Event().wait()

        async with httpx.AsyncClient(transport=httpx.MockTransport(slow)) as upstream:
            task = asyncio.create_task(
                api.query_biel_ai(
                    {"message": "Question"}, self.bound.defaults, client=upstream
                )
            )
            await started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertFalse(upstream.is_closed)
