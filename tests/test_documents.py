from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import patch

import httpx

from biel_mcp.server import (
    TOOLS,
    format_document_response,
    format_search_response,
    handle_mcp_request,
    query_biel_document,
)

RealAsyncClient = httpx.AsyncClient


class DocumentUpstream:
    def __init__(self, *, status=200, data=None):
        self.requests = []
        self.status = status
        self.data = (
            data
            if data is not None
            else {
                "document_id": "123:signed-reference",
                "title": "README.md",
                "source_type": "repository",
                "url": None,
                "file_path": "client/README.md",
                "total_chunks": 3,
                "chunks": [
                    {"content": "Full setup instructions.", "chunk_index": 0, "page": None, "sheet": ""},
                    {"content": "Additional configuration details.", "chunk_index": 1, "page": None, "sheet": ""},
                ],
                "next_cursor": "next-reference",
            }
        )

    def __call__(self, request):
        self.requests.append(request)
        return httpx.Response(self.status, json=self.data)


class DocumentFormatTest(TestCase):
    def test_get_tool_explains_pagination_and_search_first_workflow(self):
        tool = next(t for t in TOOLS if t["name"] == "biel_get_document")
        self.assertEqual(tool["inputSchema"]["required"], ["document_id"])
        self.assertTrue(tool["annotations"]["readOnlyHint"])
        self.assertIn("next_cursor", tool["description"])
        self.assertIn("explicitly requests", TOOLS[0]["description"])
        self.assertIn("ask the user first", TOOLS[0]["description"])
        self.assertIn("do not fall back to answer automatically", TOOLS[0]["description"])

    def test_search_exposes_document_id_for_get(self):
        text = format_search_response(
            {
                "results": [
                    {
                        "title": "README.md",
                        "document_id": "123:signed-reference",
                        "content": "Search excerpt",
                    }
                ]
            }
        )
        self.assertIn("Document ID: 123:signed-reference", text)

    def test_get_preserves_full_text_and_exposes_continuation(self):
        text = format_document_response(DocumentUpstream().data)
        self.assertIn("Full setup instructions.", text)
        self.assertIn("Additional configuration details.", text)
        self.assertIn("client/README.md", text)
        self.assertIn("Next cursor: next-reference", text)
        self.assertNotIn("End of indexed document", text)

    def test_final_pdf_chunks_include_page_and_sheet_references(self):
        data = DocumentUpstream().data
        data.update({"source_type": "file", "next_cursor": None})
        data["chunks"][0].update({"page": 7, "sheet": "Revenue"})
        text = format_document_response(data)
        self.assertIn("page 7, sheet Revenue", text)
        self.assertIn("End of indexed document", text)


class DocumentToolTest(IsolatedAsyncioTestCase):
    async def query(self, upstream, arguments=None, defaults=None):
        client = RealAsyncClient(transport=httpx.MockTransport(upstream))
        with patch("biel_mcp.server.httpx.AsyncClient", return_value=client):
            return await query_biel_document(
                arguments if arguments is not None else {"document_id": "123:signed-reference"},
                defaults or {"project_slug": "docs"},
            )

    async def test_get_uses_project_api_and_credentials_without_answer_generation(self):
        upstream = DocumentUpstream()
        result, chat, token = await self.query(
            upstream,
            {
                "document_id": "123:signed-reference",
                "cursor": "next",
                "limit": 2,
            },
            {
                "project_slug": "docs",
                "api_key": "private-key",
                "domain": "https://docs.example.com",
                "chat_uuid": "old-chat",
                "restore_token": "old-restore",
            },
        )
        self.assertFalse(result["isError"])
        self.assertIsNone(chat)
        self.assertIsNone(token)
        self.assertEqual(len(upstream.requests), 1)
        request = upstream.requests[0]
        self.assertEqual(request.method, "GET")
        self.assertEqual(request.url.path, "/api/v2/projects/docs/documents/123:signed-reference/")
        self.assertEqual(dict(request.url.params), {"url": "https://docs.example.com", "cursor": "next", "limit": "2"})
        self.assertEqual(request.headers["Authorization"], "Api-Key private-key")
        self.assertNotIn("X-Chat-Restore-Token", request.headers)

    async def test_router_dispatches_get_without_recording_a_chat(self):
        upstream = DocumentUpstream()
        client = RealAsyncClient(transport=httpx.MockTransport(upstream))
        with patch("biel_mcp.server.httpx.AsyncClient", return_value=client):
            response = await handle_mcp_request(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": "biel_get_document", "arguments": {"document_id": "123:signed-reference"}},
                },
                {"project_slug": "docs", "chat_uuid": "old-chat"},
            )
        self.assertFalse(response["result"]["isError"])
        self.assertIn("Additional configuration", response["result"]["content"][0]["text"])

    async def test_invalid_arguments_and_project_override_fail_before_requests(self):
        invalid = [None, {}, {"document_id": "../evil"}, {"document_id": "https://evil.example"}]
        for extra in ({"limit": 0}, {"limit": 21}, {"limit": True}, {"cursor": None}, {"project_slug": "foreign"}):
            invalid.append({"document_id": "123:signed-reference", **extra})
        for arguments in invalid:
            upstream = DocumentUpstream()
            with self.subTest(arguments=arguments):
                # Explicit None must reach the validator rather than the query helper's default.
                if arguments is None:
                    result, _, _ = await query_biel_document(None, {"project_slug": "docs"})
                else:
                    result, _, _ = await self.query(upstream, arguments)
                self.assertTrue(result["isError"])
                self.assertEqual(upstream.requests, [])

    async def test_api_failures_and_malformed_payloads_are_tool_errors(self):
        for upstream in (
            DocumentUpstream(status=403),
            DocumentUpstream(status=404),
            DocumentUpstream(status=500),
            DocumentUpstream(data={"ai_message": {"message": "unexpected answer"}}),
            DocumentUpstream(data={**DocumentUpstream().data, "document_id": "different-reference"}),
        ):
            with self.subTest(status=upstream.status):
                result, chat, token = await self.query(upstream)
                self.assertTrue(result["isError"])
                self.assertIsNone(chat)
                self.assertIsNone(token)

    async def test_legacy_document_recrawl_requirement_is_explained(self):
        result, _, _ = await self.query(DocumentUpstream(status=409))
        self.assertTrue(result["isError"])
        self.assertIn("Recrawl this source", result["content"][0]["text"])
