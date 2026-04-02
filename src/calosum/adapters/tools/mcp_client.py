from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request
from uuid import uuid4


logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class McpServerEndpoint:
    name: str
    url: str
    timeout_seconds: float = 12.0
    auth_token: str | None = None


class HttpMcpClientAdapter:
    """
    Cliente MCP via HTTP/JSON-RPC para integração opcional com servidores externos.
    """

    def __init__(
        self,
        servers: dict[str, McpServerEndpoint] | None = None,
        allowlisted_servers: set[str] | None = None,
    ) -> None:
        self.servers = servers or {}
        self.allowlisted_servers = allowlisted_servers

    def list_servers(self) -> list[str]:
        return sorted(self.servers.keys())

    def call_tool(
        self,
        *,
        server: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        endpoint = self._resolve_endpoint(server)
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid4()),
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if endpoint.auth_token:
            headers["Authorization"] = f"Bearer {endpoint.auth_token}"
        req = urllib_request.Request(
            endpoint.url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib_request.urlopen(req, timeout=endpoint.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except urllib_error.URLError as exc:
            raise RuntimeError(f"MCP transport error for server '{server}': {exc}") from exc
        except TimeoutError as exc:
            raise RuntimeError(f"MCP timeout for server '{server}'") from exc

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"MCP invalid JSON response from '{server}'") from exc

        if not isinstance(parsed, dict):
            raise RuntimeError(f"MCP malformed response from '{server}'")
        if "error" in parsed:
            raise RuntimeError(f"MCP server '{server}' returned error: {parsed['error']}")

        return {
            "server": server,
            "tool_name": tool_name,
            "result": parsed.get("result"),
            "response_id": parsed.get("id"),
        }

    def _resolve_endpoint(self, server: str) -> McpServerEndpoint:
        if self.allowlisted_servers and server not in self.allowlisted_servers:
            raise RuntimeError(f"MCP server '{server}' is not allowlisted")
        endpoint = self.servers.get(server)
        if endpoint is None:
            raise RuntimeError(f"MCP server '{server}' is not configured")
        return endpoint
