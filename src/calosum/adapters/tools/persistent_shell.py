from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

from calosum.shared.utils.tools import ToolSchema
from calosum.adapters.execution.docker_sandbox import DockerToolSandbox


logger = logging.getLogger(__name__)


class PersistentShellTool:
    """
    Shell persistente isolado via Docker por sessão.
    """

    def __init__(self, sandbox: DockerToolSandbox | None = None) -> None:
        self.sandbox = sandbox or DockerToolSandbox()
        self.schema = ToolSchema(
            name="execute_bash_persistent",
            description="Execute bash command in a persistent Docker container. Maintains state (cd, env vars) between calls.",
            parameters={"command": "string"},
            required_permissions=["shell"],
            needs_approval=True,
        )

    async def execute(self, payload: dict[str, Any], session_id: str | None = None, **_: Any) -> str:
        if not session_id:
            return "Error: session_id is required for persistent shell."
        command = str(payload.get("command", "")).strip()
        if not command:
            return "No command provided."
            
        response = await self.sandbox.execute_command(command, session_id=session_id)
        return json.dumps(response, ensure_ascii=False)
