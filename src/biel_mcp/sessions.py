"""Application conversation state independent of SDK transport sessions."""

import base64
import binascii
import logging
import uuid
from datetime import datetime
from typing import Any, Protocol

from .config import DEFAULT_BASE_URL, SESSION_TIMEOUT

logger = logging.getLogger("biel-mcp")


def restore_token_issued_at(token: str) -> int | None:
    """Read issuance ordering only; Biel's API authenticates each capability."""
    try:
        decoded = base64.urlsafe_b64decode(token)
        if decoded[0] == 0x80 and len(decoded) >= 9:
            return int.from_bytes(decoded[1:9], "big")
    except (TypeError, ValueError, IndexError, binascii.Error):
        pass
    return None


class SessionStore(Protocol):
    """Store conversation identity and its matching restore capability together."""

    async def get(self, session_id: str) -> dict[str, Any] | None: ...

    async def set(self, session_id: str, session: dict[str, Any]) -> None: ...

    async def delete(self, session_id: str) -> bool: ...

    async def touch(self, session_id: str) -> None: ...

    async def claim_chat_credentials(
        self, session_id: str, chat_uuid: str, restore_token: str
    ) -> tuple[str, str]: ...

    async def purge_expired(self) -> None: ...


class InMemorySessionStore:
    """Application session storage for a single-process standalone server."""

    def __init__(self):
        self.sessions: dict[str, dict[str, Any]] = {}

    async def get(self, session_id: str) -> dict[str, Any] | None:
        session = self.sessions.get(session_id)
        if (
            session is not None
            and (datetime.now() - session["last_active"]).total_seconds()
            > SESSION_TIMEOUT
        ):
            del self.sessions[session_id]
            return None
        return session

    async def set(self, session_id: str, session: dict[str, Any]) -> None:
        self.sessions[session_id] = session

    async def delete(self, session_id: str) -> bool:
        return self.sessions.pop(session_id, None) is not None

    async def touch(self, session_id: str) -> None:
        session = self.sessions.get(session_id)
        if session is not None:
            session["last_active"] = datetime.now()

    async def claim_chat_credentials(
        self, session_id: str, chat_uuid: str, restore_token: str
    ) -> tuple[str, str]:
        session = self.sessions.get(session_id)
        if session is None:
            return "", ""
        # Nothing is awaited between the read and the write, so the event loop
        # cannot interleave a competing claim.
        if not session["chat_uuid"]:
            session["chat_uuid"] = chat_uuid
            session["restore_token"] = restore_token
        elif session["chat_uuid"] == chat_uuid:
            previous_time = restore_token_issued_at(session.get("restore_token", ""))
            candidate_time = restore_token_issued_at(restore_token)
            if candidate_time is not None and (
                previous_time is None or candidate_time > previous_time
            ):
                session["restore_token"] = restore_token
        return session["chat_uuid"], session["restore_token"]

    async def purge_expired(self) -> None:
        """Drop sessions idle past the timeout."""
        now = datetime.now()
        expired = [
            sid
            for sid, session in self.sessions.items()
            if (now - session["last_active"]).total_seconds() > SESSION_TIMEOUT
        ]
        for sid in expired:
            del self.sessions[sid]
            logger.info("MCP session expired", extra={"event": "mcp.session_expired"})


class SessionManager:
    """Bind application sessions to one conversation across transports."""

    def __init__(self, *, store: SessionStore | None = None):
        self._store = store or InMemorySessionStore()

    async def create_session(
        self,
        project_slug: str,
        api_key: str = "",
        base_url: str = DEFAULT_BASE_URL,
        domain: str = "",
        metadata: str = "",
        client_name: str = "",
        client_version: str = "",
    ) -> str:
        """Create a new session and return session ID."""
        session_id = str(uuid.uuid4())
        await self._store.set(
            session_id,
            {
                "id": session_id,
                "project_slug": project_slug,
                "api_key": api_key,
                "base_url": base_url,
                "domain": domain,
                "metadata": metadata,
                "chat_uuid": "",
                "restore_token": "",
                "client_name": client_name,
                "client_version": client_version,
                "created_at": datetime.now(),
                "last_active": datetime.now(),
            },
        )
        logger.info("MCP session created", extra={"event": "mcp.session_created"})
        return session_id

    async def record_client_info(
        self, session_id: str, client_name: str, client_version: str
    ) -> None:
        """Attach handshake identity without changing conversation credentials."""
        if not client_name:
            return
        session = await self._store.get(session_id)
        if session is None:
            return
        if (
            session.get("client_name") == client_name
            and session.get("client_version") == client_version
        ):
            return
        session["client_name"] = client_name
        session["client_version"] = client_version
        await self._store.set(session_id, session)

    async def record_chat_credentials(
        self, session_id: str, chat_uuid: str, restore_token: str
    ) -> tuple[str, str]:
        """Atomically claim a UUID and matching capability; the first chat wins."""
        return await self._store.claim_chat_credentials(
            session_id, chat_uuid, restore_token
        )

    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Refresh idle expiry without writing a stale snapshot over credentials."""
        session = await self._store.get(session_id)
        if session is None:
            return None
        await self._store.touch(session_id)
        return session

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if await self._store.delete(session_id):
            logger.info("MCP session deleted", extra={"event": "mcp.session_deleted"})
            return True
        return False

    async def cleanup_expired_sessions(self) -> None:
        """Remove sessions that have been inactive for too long."""
        await self._store.purge_expired()
