# Biel.ai MCP server installation guide

This guide is written for AI agents (like Cline) installing the Biel.ai MCP server on a user's behalf. Biel.ai is a hosted service: there is nothing to build or run locally. You only need the user's project details.

## What this server does

Connects AI tools to a product's documentation indexed by Biel.ai. Three tools are exposed: `biel_search` retrieves matching indexed chunks, `biel_get_document` reads complete indexed documents, and `biel_ai` requests a Biel-generated answer. Prefer search and document retrieval so the calling assistant can answer from the source text.

## Prerequisites

Ask the user for:

1. Their **Biel.ai project slug** (found in the dashboard at https://app.biel.ai under the project). If they do not have an account, they can start a trial at https://app.biel.ai/accounts/signup/ and add their documentation URL as a source.
2. The **complete MCP endpoint** from **Settings** > **MCP**, including an allowed `domain` when the project restricts domains. Enable MCP and save before copying it.
3. An **API key**, only if the project is private (public projects do not need a key).

## Installation (recommended: Streamable HTTP)

For Cline, add this to the MCP settings file:

```json
{
  "mcpServers": {
    "biel-ai": {
      "type": "streamableHttp",
      "url": "https://app.biel.ai/mcp/PROJECT_SLUG"
    }
  }
}
```

Replace `PROJECT_SLUG` with the user's project slug, or use the full endpoint they copied. Configuration formats differ between clients: use the [Claude Code](https://docs.biel.ai/integrations/mcp-server#connect-technical-documentation-to-claude-code) or [Codex](https://docs.biel.ai/integrations/mcp-server#connect-technical-documentation-to-codex) guide for those clients.

For clients without native Streamable HTTP support, use mcp-remote:

```json
{
  "mcpServers": {
    "biel-ai": {
      "command": "npx",
      "args": [
        "mcp-remote",
        "https://app.biel.ai/mcp/PROJECT_SLUG"
      ]
    }
  }
}
```

For private projects, append a URL-encoded API key with `search` permission for retrieval and `chats_create` for generated answers: `https://app.biel.ai/mcp/PROJECT_SLUG?api_key=API_KEY`.

Keep URLs containing keys in private client configuration. Do not put them in shared files, public listings, or widget markup. Authorization headers and OAuth login do not replace Biel.ai’s `api_key` query parameter.

## Verify the installation

Confirm all three tools are discovered, then ask a question that the user's indexed documentation answers, for example:

```
Use biel_search to find API authentication instructions, then use
biel_get_document to read relevant results and answer with source links.
```

Check the retrieved text and references against the user's documentation. References can be URLs or document titles with page or sheet locations.

Search combines keyword and semantic retrieval. Use `biel_get_document` when search excerpts need more context. Follow any returned cursor with the same document ID and limit until the document is complete. Ask before calling `biel_ai`, unless the user already explicitly requested a Biel-generated answer. Retrieval errors must not trigger answer generation.

MCP returns indexed text, not live pages or original uploaded files. After source changes, refresh sources in Biel.ai and search again for current document IDs.

## Troubleshooting

- **404**: verify the endpoint and project slug. If an existing session has expired, reconnect.
- **Connection refused**: check that the client can reach the endpoint host.
- **403**: check project access, private-project key permissions, allowed domains, and usage limits. Do not assume every access error means the project is private.
- **Empty or unhelpful answers**: the project may not have finished indexing. Check the project's sync status in the dashboard.

Full documentation: https://docs.biel.ai/integrations/mcp-server
