from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from calosum.domain.agent.multiagent import MultiAgentWorkflow
from calosum.shared.utils.tools import ToolSchema


class SubordinateAgentTool:
    def __init__(self, agent_accessor: Any = None) -> None:
        self.agent_accessor = agent_accessor
        self.schema = ToolSchema(
            name="spawn_subordinate",
            description="Delegate a subtask to an isolated subordinate workflow.",
            parameters={"task": "string"},
            required_permissions=[],
            needs_approval=False,
        )

    async def execute(self, payload: dict[str, Any], **_: Any) -> str:
        task = str(payload.get("task", "")).strip()
        if not task:
            return "Subordinate delegation rejected: 'task' is required."

        timeout_seconds = float(payload.get("timeout_seconds", 8.0))
        try:
            outcome = await MultiAgentWorkflow().aorchestrate(
                json.dumps({"task": task, "delegation_id": str(uuid4())}, ensure_ascii=False),
                timeout_seconds=timeout_seconds,
            )
        except TimeoutError:
            return "Subordinate delegation timed out."

        response = {
            "status": "completed",
            "task": task,
            "verification": outcome,
        }
        return json.dumps(response, ensure_ascii=False)
