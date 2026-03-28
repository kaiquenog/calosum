from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from .types import AgentTurnResult, utc_now


@dataclass(slots=True)
class TelemetryEvent:
    channel: str
    session_id: str
    turn_id: str
    recorded_at: str
    payload: dict[str, Any]


class TelemetrySink(Protocol):
    def emit(self, event: TelemetryEvent) -> None: ...


@dataclass(slots=True)
class InMemoryTelemetrySink:
    events: list[TelemetryEvent] = field(default_factory=list)

    def emit(self, event: TelemetryEvent) -> None:
        self.events.append(event)

    def query(
        self,
        session_id: str | None = None,
        channel: str | None = None,
    ) -> list[TelemetryEvent]:
        filtered = self.events
        if session_id is not None:
            filtered = [event for event in filtered if event.session_id == session_id]
        if channel is not None:
            filtered = [event for event in filtered if event.channel == channel]
        return filtered


class CognitiveTelemetryBus:
    """
    Barramento de observabilidade que separa a telemetria por camada cognitiva.
    """

    def __init__(self, sink: TelemetrySink | None = None) -> None:
        self.sink = sink or InMemoryTelemetrySink()

    def record_turn(self, result: AgentTurnResult) -> None:
        session_id = result.user_turn.session_id
        turn_id = result.user_turn.turn_id
        timestamp = utc_now().isoformat()

        self.sink.emit(
            TelemetryEvent(
                channel="felt",
                session_id=session_id,
                turn_id=turn_id,
                recorded_at=timestamp,
                payload=result.telemetry.felt,
            )
        )
        self.sink.emit(
            TelemetryEvent(
                channel="thought",
                session_id=session_id,
                turn_id=turn_id,
                recorded_at=timestamp,
                payload=result.telemetry.thought,
            )
        )
        self.sink.emit(
            TelemetryEvent(
                channel="decision",
                session_id=session_id,
                turn_id=turn_id,
                recorded_at=timestamp,
                payload=result.telemetry.decision,
            )
        )
        self.sink.emit(
            TelemetryEvent(
                channel="execution",
                session_id=session_id,
                turn_id=turn_id,
                recorded_at=timestamp,
                payload={
                    "results": [asdict(item) for item in result.execution_results],
                },
            )
        )

    def record_reflection(
        self,
        session_id: str,
        turn_id: str,
        payload: dict[str, Any],
    ) -> None:
        self.sink.emit(
            TelemetryEvent(
                channel="reflection",
                session_id=session_id,
                turn_id=turn_id,
                recorded_at=utc_now().isoformat(),
                payload=payload,
            )
        )

    def dashboard_for_session(self, session_id: str) -> dict[str, list[dict[str, Any]]]:
        if not hasattr(self.sink, "query"):
            raise TypeError("dashboard_for_session requires a queryable telemetry sink")
        channels = ("felt", "thought", "decision", "execution", "reflection")
        dashboard: dict[str, list[dict[str, Any]]] = {}
        for channel in channels:
            dashboard[channel] = [
                event.payload for event in self.sink.query(session_id=session_id, channel=channel)
            ]
        return dashboard
