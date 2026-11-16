"""Shared package configuration without transport initialization."""

import math
import os
import re

import httpx

SERVER_VERSION = "2.1.1"
SERVER_NAME = "biel-ai-mcp"
DEFAULT_PORT = 7832
DEFAULT_BASE_URL = "https://app.biel.ai"
BIEL_API_PATH_TEMPLATE = "/api/v2/projects/{project_slug}/chats/"
BIEL_SEARCH_PATH_TEMPLATE = "/api/v2/projects/{project_slug}/search/"
BIEL_DOCUMENT_PATH_TEMPLATE = "/api/v2/projects/{project_slug}/documents/{document_id}/"
API_READ_TIMEOUT_SECONDS = float(os.environ.get("BIEL_MCP_READ_TIMEOUT_SECONDS", "600"))
if not math.isfinite(API_READ_TIMEOUT_SECONDS) or API_READ_TIMEOUT_SECONDS <= 0:
    raise ValueError("BIEL_MCP_READ_TIMEOUT_SECONDS must be a positive finite number")
REQUEST_TIMEOUT = httpx.Timeout(
    API_READ_TIMEOUT_SECONDS, connect=5.0, write=10.0, pool=5.0
)
# Keep restore credentials alive beyond the longest upstream read.
SESSION_TIMEOUT = math.ceil(API_READ_TIMEOUT_SECONDS) + 300
KEEPALIVE_INTERVAL = 30
PROJECT_SLUG_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
CLIENT_IDENTITY_MAX_LENGTH = 128
