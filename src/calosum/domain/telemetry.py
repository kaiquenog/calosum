from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from calosum.shared.types import AgentTurnResult, utc_now
from calosum.shared.serialization import to_primitive


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


def event_to_otlp_trace_envelope(
    event: TelemetryEvent,
    *,
    service_name: str = "calosum",
) -> dict[str, Any]:
    timestamp_unix_nano = _iso_to_unix_nano(event.recorded_at)
    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": service_name}},
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
                                "kind": 1,
                                "startTimeUnixNano": timestamp_unix_nano,
                                "endTimeUnixNano": timestamp_unix_nano,
                                "attributes": [
                                    {"key": "calosum.turn_id", "value": {"stringValue": event.turn_id}},
                                    {
                                        "key": "calosum.recorded_at",
                                        "value": {"stringValue": event.recorded_at},
                                    },
                                    {
                                        "key": "calosum.payload",
                                        "value": {"stringValue": json.dumps(event.payload, ensure_ascii=False)},
                                    },
                                ]
                                + [
                                    {
                                        "key": f"calosum.metric.{key}",
                                        "value": {"doubleValue": float(sum(value)) if isinstance(value, (list, tuple)) else float(value)},
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


def _iso_to_unix_nano(recorded_at: str) -> str:
    try:
        dt = datetime.fromisoformat(recorded_at)
    except ValueError:
        dt = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return str(int(dt.timestamp() * 1_000_000_000))


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
class CompositeTelemetrySink:
    sinks: list[TelemetrySink]
    query_sink: Any | None = None

    def emit(self, event: TelemetryEvent) -> None:
        for sink in self.sinks:
            sink.emit(event)

    def query(
        self,
        session_id: str | None = None,
        channel: str | None = None,
    ) -> list[TelemetryEvent]:
        sink = self.query_sink or next((item for item in self.sinks if hasattr(item, "query")), None)
        if sink is None:
            raise TypeError("Composite telemetry sink requires at least one queryable sink")
        return sink.query(session_id=session_id, channel=channel)


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
            handle.write(json.dumps(event_to_otlp_trace_envelope(event, service_name=self.service_name), ensure_ascii=False) + "\n")

    def query(
        self,
        session_id: str | None = None,
        channel: str | None = None,
    ) -> list[TelemetryEvent]:
        filtered = self._read_persisted_events() if self.path.exists() else self.events
        if session_id is not None:
            filtered = [event for event in filtered if event.session_id == session_id]
        if channel is not None:
            filtered = [event for event in filtered if event.channel == channel]
        return filtered

    def _to_otlp_envelope(self, event: TelemetryEvent) -> dict[str, Any]:
        return event_to_otlp_trace_envelope(event, service_name=self.service_name)

    def _read_persisted_events(self) -> list[TelemetryEvent]:
        events: list[TelemetryEvent] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    envelope = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                events.extend(self._events_from_envelope(envelope))
        return events

    def _events_from_envelope(self, envelope: dict[str, Any]) -> list[TelemetryEvent]:
        events: list[TelemetryEvent] = []
        for resource_span in envelope.get("resourceSpans", []):
            session_id = self._attribute_string(
                resource_span.get("resource", {}).get("attributes", []),
                "calosum.session_id",
            )
            for scope_span in resource_span.get("scopeSpans", []):
                for span in scope_span.get("spans", []):
                    attributes = span.get("attributes", [])
                    payload_raw = self._attribute_string(attributes, "calosum.payload")
                    if not session_id or payload_raw is None:
                        continue
                    try:
                        payload = json.loads(payload_raw)
                    except json.JSONDecodeError:
                        continue

                    metrics: dict[str, float] = {}
                    for attribute in attributes:
                        key = attribute.get("key", "")
                        if not key.startswith("calosum.metric."):
                            continue
                        metric_name = key.removeprefix("calosum.metric.")
                        metric_value = self._numeric_attribute_value(attribute.get("value", {}))
                        if metric_value is not None:
                            metrics[metric_name] = metric_value

                    events.append(
                        TelemetryEvent(
                            channel=span.get("name", ""),
                            session_id=session_id,
                            turn_id=self._attribute_string(attributes, "calosum.turn_id") or "",
                            recorded_at=self._attribute_string(attributes, "calosum.recorded_at") or "",
                            payload=payload,
                            trace_id=span.get("traceId", ""),
                            span_id=span.get("spanId", ""),
                            metrics=metrics,
                        )
                    )
        return events

    def _attribute_string(self, attributes: list[dict[str, Any]], key: str) -> str | None:
        for attribute in attributes:
            if attribute.get("key") != key:
                continue
            value = attribute.get("value", {})
            if "stringValue" in value:
                return value["stringValue"]
        return None

    def _numeric_attribute_value(self, value: dict[str, Any]) -> float | None:
        for field_name in ("doubleValue", "intValue"):
            if field_name not in value:
                continue
            try:
                return float(value[field_name])
            except (TypeError, ValueError):
                return None
        return None


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
                payload=to_primitive({
                    **result.telemetry.felt,
                    "expected_free_energy": result.right_state.world_hypotheses.get("expected_free_energy", 0.0),
                    "peer_latents_count": result.telemetry.felt.get("peer_latents_count", 0),
                }),
                trace_id=trace_id,
                span_id=self._span_id(trace_id, "felt"),
                metrics=to_primitive({
                    "latency_ms": result.latency_ms,
                    "expected_free_energy": result.right_state.world_hypotheses.get("expected_free_energy", 0.0),
                    "surprise_score": result.right_state.surprise_score,
                }),
            )
        )
        self.sink.emit(
            TelemetryEvent(
                channel="thought",
                session_id=session_id,
                turn_id=turn_id,
                recorded_at=timestamp,
                payload=to_primitive({
                    **result.telemetry.thought,
                    "bridge_config": result.telemetry.bridge_config,
                    "active_variant": result.telemetry.active_variant,
                }),
                trace_id=trace_id,
                span_id=self._span_id(trace_id, "thought"),
                metrics=to_primitive({
                    "latency_ms": result.latency_ms,
                    "runtime_retry_count": result.runtime_retry_count,
                    "critique_revision_count": result.critique_revision_count,
                }),
            )
        )
        self.sink.emit(
            TelemetryEvent(
                channel="decision",
                session_id=session_id,
                turn_id=turn_id,
                recorded_at=timestamp,
                payload=to_primitive({
                    **result.telemetry.decision,
                    "capabilities": result.telemetry.capabilities,
                }),
                trace_id=trace_id,
                span_id=self._span_id(trace_id, "decision"),
                metrics=to_primitive({
                    "latency_ms": result.latency_ms,
                    "runtime_retry_count": result.runtime_retry_count,
                    "tool_success_rate": result.telemetry.decision.get("tool_success_rate", 1.0),
                }),
            )
        )
        self.sink.emit(
            TelemetryEvent(
                channel="execution",
                session_id=session_id,
                turn_id=turn_id,
                recorded_at=timestamp,
                payload=to_primitive({
                    "results": [asdict(item) for item in result.execution_results],
                }),
                trace_id=trace_id,
                span_id=self._span_id(trace_id, "execution"),
                metrics=to_primitive({
                    "latency_ms": result.latency_ms,
                    "runtime_retry_count": result.runtime_retry_count,
                    "rejected_count": sum(1 for item in result.execution_results if item.status == "rejected"),
                    "tool_success_rate": result.telemetry.decision.get("tool_success_rate", 1.0),
                }),
            )
        )

    async def arecord_turn(self, result: AgentTurnResult) -> None:
        self.record_turn(result)

    def record_reflection(self, session_id: str, turn_id: str, payload: dict[str, Any]) -> None:
        cost_metrics = payload.get("cost_metrics", {})
        self.sink.emit(
            TelemetryEvent(
                channel="reflection", session_id=session_id, turn_id=turn_id,
                recorded_at=utc_now().isoformat(), payload=to_primitive(payload),
                trace_id=hashlib.sha256(f"{turn_id}:reflection".encode("utf-8")).hexdigest()[:32],
                span_id=self._span_id(turn_id, "reflection"),
                metrics=to_primitive({
                    "branch_count": cost_metrics.get("branch_count", 0.0),
                    "total_latency_ms": cost_metrics.get("total_latency_ms", 0.0),
                    "cognitive_dissonance": payload.get("cognitive_dissonance", 0.0),
                }),
            )
        )

    async def arecord_reflection(self, session_id: str, turn_id: str, payload: dict[str, Any]) -> None:
        self.record_reflection(session_id, turn_id, payload)

    def record_awareness(self, session_id: str, diagnostic: Any) -> None:
        self.sink.emit(
            TelemetryEvent(
                channel="awareness",
                session_id=session_id,
                turn_id="system-loop",
                recorded_at=utc_now().isoformat(),
                payload=to_primitive(diagnostic),
                trace_id=hashlib.sha256(f"{session_id}:awareness".encode("utf-8")).hexdigest()[:32],
                span_id=self._span_id("system-loop", "awareness"),
                metrics=to_primitive({
                    "bottlenecks_count": len(diagnostic.bottlenecks),
                    "tool_success_rate": diagnostic.tool_success_rate,
                    "pending_approval_backlog": diagnostic.pending_approval_backlog,
                    "pending_directive_count": diagnostic.pending_directive_count,
                    "surprise_trend": diagnostic.surprise_trend,
                    "dominant_variant_ratio": diagnostic.dominant_variant_ratio,
                }),
            )
        )

    def dashboard_for_session(self, session_id: str | None = None) -> dict[str, list[dict[str, Any]]]:
        if not hasattr(self.sink, "query"): raise TypeError("dashboard_for_session requires a queryable telemetry sink")
        return {c: [{**e.payload, "_session_id": e.session_id, "_recorded_at": e.recorded_at} for e in self.sink.query(session_id=session_id, channel=c)] for c in ("felt", "thought", "decision", "execution", "reflection", "awareness")}

    def _span_id(self, seed: str, channel: str) -> str: return hashlib.sha256(f"{seed}:{channel}".encode("utf-8")).hexdigest()[:16]
