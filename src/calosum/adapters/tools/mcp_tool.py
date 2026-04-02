from __future__ import annotations

import json
from typing import Any

from calosum.shared.utils.tools import ToolSchema


class McpTool:
    def __init__(self, mcp_client: Any | None = None) -> None:
        self.mcp_client = mcp_client
        self.schema = ToolSchema(
            name="call_mcp_tool",
            description="Call an external MCP server tool by name.",
            parameters={
                "server": "string",
                "tool_name": "string",
                "arguments": "dict",
            },
            required_permissions=["network"],
            needs_approval=True,
        )

    async def execute(self, payload: dict[str, Any], **_: Any) -> str:
        if self.mcp_client is None:
            return "MCP is not enabled in this environment."
        server = str(payload.get("server", "")).strip()
        tool_name = str(payload.get("tool_name", "")).strip()
        arguments = payload.get("arguments", {})
        if not server:
            return "MCP call rejected: 'server' is required."
        if not tool_name:
            return "MCP call rejected: 'tool_name' is required."
        if not isinstance(arguments, dict):
            return "MCP call rejected: 'arguments' must be an object."

        try:
            result = self.mcp_client.call_tool(
                server=server,
                tool_name=tool_name,
                arguments=arguments,
            )
        except Exception as exc:
            return f"MCP call failed: {exc}"
        return json.dumps(result, ensure_ascii=False)
