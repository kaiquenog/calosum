from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from calosum.domain.event_bus import CognitiveEvent, InternalEventBus


logger = logging.getLogger(__name__)


class CognitiveInterceptor(Protocol):
    async def on_stage(self, stage: str, payload: dict[str, Any]) -> None: ...


@dataclass(slots=True)
class AuditLogInterceptor:
    max_events: int = 200
    events: list[dict[str, Any]] = field(default_factory=list)

    async def on_stage(self, stage: str, payload: dict[str, Any]) -> None:
        self.events.append({"stage": stage, "payload": payload})
        if len(self.events) > self.max_events:
            del self.events[0 : len(self.events) - self.max_events]


class InterceptorManager:
    def __init__(self, interceptors: list[CognitiveInterceptor] | None = None) -> None:
        self.interceptors = interceptors or []

    async def aemit(self, stage: str, payload: dict[str, Any]) -> None:
        for interceptor in self.interceptors:
            try:
                await interceptor.on_stage(stage, payload)
            except Exception:
                logger.exception("Interceptor failed on stage '%s'", stage)

    def attach_event_bus(self, event_bus: InternalEventBus) -> None:
        event_bus.subscribe("UserTurnEvent", self._on_user_turn)
        event_bus.subscribe("PerceptionEvent", self._on_perception)
        event_bus.subscribe("ExecutionEvent", self._on_execution)

    async def _on_user_turn(self, event: CognitiveEvent) -> None:
        await self.aemit(
            "message_loop_start",
            {"turn_id": event.turn_id, "event_type": event.event_type},
        )

    async def _on_perception(self, event: CognitiveEvent) -> None:
        await self.aemit(
            "after_perception",
            {"turn_id": event.turn_id, "event_type": event.event_type},
        )

    async def _on_execution(self, event: CognitiveEvent) -> None:
        await self.aemit(
            "after_turn_execution",
            {"turn_id": event.turn_id, "event_type": event.event_type},
        )
