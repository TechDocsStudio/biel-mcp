# Biel.ai MCP server installation guide

This guide is written for AI agents (like Cline) installing the Biel.ai MCP server on a user's behalf. Biel.ai is a hosted service: there is nothing to build or run locally. You only need the user's project details.

## What this server does

Connects AI tools to a product's documentation indexed by Biel.ai. One tool is exposed: ask questions and get answers generated from the indexed docs, with citations to the source pages.

## Prerequisites

Ask the user for:

1. Their **Biel.ai project slug** (found in the dashboard at https://app.biel.ai under the project). If they do not have an account, they can start a trial at https://app.biel.ai/accounts/signup/ and add their documentation URL as a source.
2. Their **documentation site URL** (only needed for the legacy SSE endpoint).
3. An **API key**, only if the project is private (most are public; skip unless the connection is rejected).

## Installation (recommended: Streamable HTTP)

Add this to the MCP settings file:

```json
{
  "mcpServers": {
    "biel-ai": {
      "type": "streamableHttp",
      "url": "https://mcp.biel.ai/v2/PROJECT_SLUG/mcp"
    }
  }
}
```

Replace `PROJECT_SLUG` with the user's project slug.

For clients without native Streamable HTTP support, use mcp-remote:

```json
{
  "mcpServers": {
    "biel-ai": {
      "command": "npx",
      "args": [
        "mcp-remote",
        "https://mcp.biel.ai/v2/PROJECT_SLUG/mcp"
      ]
    }
  }
}
```

For private projects, append the API key: `https://mcp.biel.ai/v2/PROJECT_SLUG/mcp?api_key=API_KEY`.

## Verify the installation

Ask a question that the user's documentation answers, prefixed to route to the tool, for example:

```
Using biel_ai, what authentication does the API use?
```

A correct installation returns an answer with source links to the user's documentation pages.

## Troubleshooting

- **404 or connection refused**: the project slug is wrong. Confirm it in the dashboard.
- **Authorization error**: the project is private; ask the user for an API key and append it as the `api_key` query parameter.
- **Empty or unhelpful answers**: the project may not have finished indexing. Check the project's sync status in the dashboard.

Full documentation: https://docs.biel.ai/integrations/mcp-server
