"""Per-request application context shared by HTTP and SDK tool handlers."""

from contextvars import ContextVar
from dataclasses import dataclass, field

from .sessions import SessionManager


@dataclass
class RequestContext:
    defaults: dict[str, str] = field(default_factory=dict)
    session_id: str | None = None
    sessions: SessionManager | None = None
    conversations: SessionManager | None = None


current_request: ContextVar[RequestContext | None] = ContextVar(
    "biel_mcp_request", default=None
)
