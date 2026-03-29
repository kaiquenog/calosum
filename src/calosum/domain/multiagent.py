from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from calosum.domain.event_bus import CognitiveEvent, InternalEventBus
from calosum.shared.types import UserTurn

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
        await self.event_bus.publish(CognitiveEvent("PlanCreatedEvent", plan, event.turn_id))


class ExecutorRole(AgentRole):
    def _register_handlers(self) -> None:
        self.event_bus.subscribe("PlanCreatedEvent", self.on_plan_created)

    async def on_plan_created(self, event: CognitiveEvent) -> None:
        logger.info(f"[Executor] Executing plan: {event.payload}")
        # Simulando a execução
        result = {"status": "success", "executed_steps": event.payload.get("steps", [])}
        await self.event_bus.publish(CognitiveEvent("ExecutionCompletedEvent", result, event.turn_id))


class VerifierRole(AgentRole):
    def _register_handlers(self) -> None:
        self.event_bus.subscribe("ExecutionCompletedEvent", self.on_execution_completed)

    async def on_execution_completed(self, event: CognitiveEvent) -> None:
        logger.info(f"[Verifier] Verifying execution: {event.payload}")
        # Simulando a verificação
        verdict = {"is_valid": True, "notes": "Looks good"}
        await self.event_bus.publish(CognitiveEvent("VerificationCompletedEvent", verdict, event.turn_id))