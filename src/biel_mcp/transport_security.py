"""HTTP authority and browser-origin policy, independent of upstream destinations."""

from dataclasses import dataclass
from ipaddress import ip_address
import re
from urllib.parse import urlsplit

from starlette.responses import JSONResponse


def normalize_origin(value):
    if (
        any(ord(char) < 33 or ord(char) == 127 for char in value)
        or "?" in value
        or "#" in value
    ):
        return None
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        return None
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path
        or parsed.query
        or parsed.fragment
        or any(char.isspace() for char in value)
        or "*" in parsed.netloc
        or "," in parsed.netloc
        or port == 0
        or parsed.netloc.endswith(":")
    ):
        return None
    host = parsed.hostname.lower()
    if ":" in host:
        try:
            ip_address(host)
        except ValueError:
            return None
    elif not re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?", host):
        return None
    return (
        parsed.scheme,
        parsed.hostname.lower(),
        port or (443 if parsed.scheme == "https" else 80),
    )


def normalize_host(value):
    origin = normalize_origin("http://" + value)
    return origin[1] if origin else None


@dataclass(frozen=True)
class TransportPolicy:
    allowed_hosts: frozenset
    allowed_origins: frozenset = frozenset()

    def __post_init__(self):
        if not self.allowed_hosts or any(
            normalize_host(f"[{host}]" if ":" in host else host) != host.lower()
            for host in self.allowed_hosts
        ):
            raise ValueError(
                "MCP allowed hosts must be explicit hostnames, without wildcards or ports"
            )
        if any(normalize_origin(origin) is None for origin in self.allowed_origins):
            raise ValueError("MCP allowed origins must be explicit HTTP(S) origins")


class TransportSecurityMiddleware:
    def __init__(self, app, *, state):
        self.app = app
        self.state = state

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        policy = self.state.transport_policy
        headers = scope.get("headers", [])
        hosts = [
            value.decode("latin-1") for key, value in headers if key.lower() == b"host"
        ]
        origins = [
            value.decode("latin-1")
            for key, value in headers
            if key.lower() == b"origin"
        ]
        host = normalize_host(hosts[0]) if len(hosts) == 1 else None
        allowed_hosts = {value.lower() for value in policy.allowed_hosts}
        if host not in allowed_hosts:
            await JSONResponse({"error": "Host is not allowed"}, status_code=403)(
                scope, receive, send
            )
            return
        allowed_origins = {normalize_origin(value) for value in policy.allowed_origins}
        if origins and (
            len(origins) != 1 or normalize_origin(origins[0]) not in allowed_origins
        ):
            await JSONResponse({"error": "Origin is not allowed"}, status_code=403)(
                scope, receive, send
            )
            return
        await self.app(scope, receive, send)
