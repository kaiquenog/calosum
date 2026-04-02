from __future__ import annotations

import asyncio
import json
from typing import Any
from uuid import uuid4

from calosum.domain.infrastructure.event_bus import CognitiveEvent, InternalEventBus
from calosum.domain.agent.multiagent import ExecutorRole, PlannerRole, VerifierRole
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
        event_bus = InternalEventBus()
        PlannerRole("planner", event_bus)
        ExecutorRole("executor", event_bus)
        VerifierRole("verifier", event_bus)

        completed = asyncio.Event()
        outcome: dict[str, Any] = {}

        async def _on_verification(event: CognitiveEvent) -> None:
            nonlocal outcome
            outcome = event.payload if isinstance(event.payload, dict) else {"payload": event.payload}
            completed.set()

        event_bus.subscribe("VerificationCompletedEvent", _on_verification)
        await event_bus.publish(
            CognitiveEvent(
                "TaskAssignedEvent",
                {"task": task, "delegation_id": str(uuid4())},
                turn_id=str(uuid4()),
            )
        )
        try:
            await asyncio.wait_for(completed.wait(), timeout=timeout_seconds)
        except TimeoutError:
            return "Subordinate delegation timed out."
        finally:
            await event_bus.stop()

        response = {
            "status": "completed",
            "task": task,
            "verification": outcome,
        }
        return json.dumps(response, ensure_ascii=False)
