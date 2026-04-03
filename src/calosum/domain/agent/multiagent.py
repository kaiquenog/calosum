from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any

from calosum.domain.infrastructure.event_bus import CognitiveEvent, InternalEventBus
from calosum.shared.models.types import UserTurn

logger = logging.getLogger(__name__)


class AgentRole(ABC):
    """
    Papel mínimo de um agente dentro da orquestração multiagente.
    Os agentes se comunicam exclusivamente via InternalEventBus.
    """

    def __init__(self, name: str, event_bus: InternalEventBus):
        self.name = name
        self.event_bus = event_bus
        self._register_handlers()

    @abstractmethod
    def _register_handlers(self) -> None:
        """Registra os handlers no event_bus."""
        pass


class PlannerRole(AgentRole):
    def _register_handlers(self) -> None:
        self.event_bus.subscribe("TaskAssignedEvent", self.on_task_assigned)

    async def on_task_assigned(self, event: CognitiveEvent) -> None:
        logger.info(f"[Planner] Received task: {event.payload}")
        # Simulando o planejamento
        plan = {"steps": ["step 1", "step 2"], "original_task": event.payload}
        self.event_bus.publish(CognitiveEvent("PlanCreatedEvent", plan, event.turn_id))


class ExecutorRole(AgentRole):
    def _register_handlers(self) -> None:
        self.event_bus.subscribe("PlanCreatedEvent", self.on_plan_created)

    async def on_plan_created(self, event: CognitiveEvent) -> None:
        logger.info(f"[Executor] Executing plan: {event.payload}")
        # Simulando a execução
        result = {"status": "success", "executed_steps": event.payload.get("steps", [])}
        self.event_bus.publish(CognitiveEvent("ExecutionCompletedEvent", result, event.turn_id))


class VerifierRole(AgentRole):
    def _register_handlers(self) -> None:
        self.event_bus.subscribe("ExecutionCompletedEvent", self.on_execution_completed)

    async def on_execution_completed(self, event: CognitiveEvent) -> None:
        logger.info(f"[Verifier] Verifying execution: {event.payload}")
        verdict = {
            "is_valid": event.payload.get("status") == "success",
            "notes": "Looks good" if event.payload.get("status") == "success" else "Execution failed",
            "executed_steps": event.payload.get("executed_steps", []),
            "original_task": event.payload.get("original_task"),
        }
        self.event_bus.publish(CognitiveEvent("VerificationCompletedEvent", verdict, event.turn_id))


class MultiAgentWorkflow:
    """Orquestra planner, executor e verifier sobre o barramento interno."""

    def __init__(self, event_bus: InternalEventBus | None = None) -> None:
        self.event_bus = event_bus or InternalEventBus()
        self.planner = PlannerRole("planner", self.event_bus)
        self.executor = ExecutorRole("executor", self.event_bus)
        self.verifier = VerifierRole("verifier", self.event_bus)

    async def aorchestrate(self, task: str, timeout_seconds: float = 8.0) -> dict[str, Any]:
        completed: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()

        async def _on_verification(event: CognitiveEvent) -> None:
            payload = event.payload if isinstance(event.payload, dict) else {"payload": event.payload}
            if not completed.done():
                completed.set_result(payload)

        self.event_bus.subscribe("VerificationCompletedEvent", _on_verification)
        self.event_bus.publish(CognitiveEvent("TaskAssignedEvent", {"task": task}, turn_id=task))
        return await asyncio.wait_for(completed, timeout=timeout_seconds)

    def orchestrate(self, task: str, timeout_seconds: float = 8.0) -> dict[str, Any]:
        return asyncio.run(self.aorchestrate(task, timeout_seconds=timeout_seconds))
