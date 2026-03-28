from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from calosum.shared.types import AgentTurnResult, utc_now


@dataclass(slots=True)
class TelemetryEvent:
    channel: str
    session_id: str
    turn_id: str
    recorded_at: str
    payload: dict[str, Any]
    trace_id: str = ""
    span_id: str = ""
    metrics: dict[str, float] = field(default_factory=dict)


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


@dataclass(slots=True)
class OTLPJsonlTelemetrySink:
    path: Path
    service_name: str = "calosum"
    events: list[TelemetryEvent] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: TelemetryEvent) -> None:
        self.events.append(event)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(self._to_otlp_envelope(event), ensure_ascii=False) + "\n")

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

    def _to_otlp_envelope(self, event: TelemetryEvent) -> dict[str, Any]:
        return {
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": [
                            {"key": "service.name", "value": {"stringValue": self.service_name}},
                            {"key": "calosum.session_id", "value": {"stringValue": event.session_id}},
                        ]
                    },
                    "scopeSpans": [
                        {
                            "scope": {"name": "calosum.telemetry"},
                            "spans": [
                                {
                                    "traceId": event.trace_id,
                                    "spanId": event.span_id,
                                    "name": event.channel,
                                    "attributes": [
                                        {"key": "calosum.turn_id", "value": {"stringValue": event.turn_id}},
                                        {
                                            "key": "calosum.payload",
                                            "value": {"stringValue": json.dumps(event.payload, ensure_ascii=False)},
                                        },
                                    ]
                                    + [
                                        {
                                            "key": f"calosum.metric.{key}",
                                            "value": {"doubleValue": float(value)},
                                        }
                                        for key, value in event.metrics.items()
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ]
        }


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
        trace_id = hashlib.sha256(f"{session_id}:{turn_id}".encode("utf-8")).hexdigest()[:32]

        self.sink.emit(
            TelemetryEvent(
                channel="felt",
                session_id=session_id,
                turn_id=turn_id,
                recorded_at=timestamp,
                payload=result.telemetry.felt,
                trace_id=trace_id,
                span_id=self._span_id(trace_id, "felt"),
                metrics={"latency_ms": result.latency_ms},
            )
        )
        self.sink.emit(
            TelemetryEvent(
                channel="thought",
                session_id=session_id,
                turn_id=turn_id,
                recorded_at=timestamp,
                payload=result.telemetry.thought,
                trace_id=trace_id,
                span_id=self._span_id(trace_id, "thought"),
                metrics={
                    "latency_ms": result.latency_ms,
                    "runtime_retry_count": float(result.runtime_retry_count),
                },
            )
        )
        self.sink.emit(
            TelemetryEvent(
                channel="decision",
                session_id=session_id,
                turn_id=turn_id,
                recorded_at=timestamp,
                payload=result.telemetry.decision,
                trace_id=trace_id,
                span_id=self._span_id(trace_id, "decision"),
                metrics={
                    "latency_ms": result.latency_ms,
                    "runtime_retry_count": float(result.runtime_retry_count),
                },
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
                trace_id=trace_id,
                span_id=self._span_id(trace_id, "execution"),
                metrics={
                    "latency_ms": result.latency_ms,
                    "runtime_retry_count": float(result.runtime_retry_count),
                    "rejected_count": float(
                        sum(1 for item in result.execution_results if item.status == "rejected")
                    ),
                },
            )
        )

    async def arecord_turn(self, result: AgentTurnResult) -> None:
        self.record_turn(result)

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
                trace_id=hashlib.sha256(f"{session_id}:{turn_id}".encode("utf-8")).hexdigest()[:32],
                span_id=self._span_id(turn_id, "reflection"),
            )
        )

    async def arecord_reflection(
        self,
        session_id: str,
        turn_id: str,
        payload: dict[str, Any],
    ) -> None:
        self.record_reflection(session_id, turn_id, payload)

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

    def _span_id(self, seed: str, channel: str) -> str:
        return hashlib.sha256(f"{seed}:{channel}".encode("utf-8")).hexdigest()[:16]
