from __future__ import annotations

import logging
from time import monotonic
from dataclasses import dataclass, field

import httpx

from calosum.domain.infrastructure.telemetry import TelemetryEvent, event_to_otlp_trace_envelope

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class OTLPHTTPTraceSink:
    endpoint: str
    service_name: str = "calosum"
    timeout_s: float = 2.0
    base_backoff_s: float = 1.0
    max_backoff_s: float = 30.0
    _warned_unavailable: bool = field(default=False, init=False, repr=False)
    _consecutive_failures: int = field(default=0, init=False, repr=False)
    _next_allowed_emit_at: float = field(default=0.0, init=False, repr=False)

    def emit(self, event: TelemetryEvent) -> None:
        if monotonic() < self._next_allowed_emit_at:
            return
        url = self._trace_url()
        try:
            response = httpx.post(
                url,
                json=event_to_otlp_trace_envelope(event, service_name=self.service_name),
                headers={"Content-Type": "application/json"},
                timeout=self.timeout_s,
            )
            response.raise_for_status()
            self._warned_unavailable = False
            self._consecutive_failures = 0
            self._next_allowed_emit_at = 0.0
        except Exception as exc:
            self._consecutive_failures += 1
            delay = min(self.max_backoff_s, self.base_backoff_s * (2 ** max(0, self._consecutive_failures - 1)))
            self._next_allowed_emit_at = monotonic() + delay
            if not self._warned_unavailable:
                logger.warning("Failed to export OTLP trace to %s: %s", url, exc)
                self._warned_unavailable = True

    def _trace_url(self) -> str:
        base = self.endpoint.rstrip("/")
        if base.endswith("/v1/traces"):
            return base
        return f"{base}/v1/traces"
