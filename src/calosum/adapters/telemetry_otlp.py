from __future__ import annotations

import logging
from dataclasses import dataclass, field

import httpx

from calosum.domain.telemetry import TelemetryEvent, event_to_otlp_trace_envelope

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class OTLPHTTPTraceSink:
    endpoint: str
    service_name: str = "calosum"
    timeout_s: float = 2.0
    _warned_unavailable: bool = field(default=False, init=False, repr=False)

    def emit(self, event: TelemetryEvent) -> None:
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
        except Exception as exc:
            if not self._warned_unavailable:
                logger.warning("Failed to export OTLP trace to %s: %s", url, exc)
                self._warned_unavailable = True

    def _trace_url(self) -> str:
        base = self.endpoint.rstrip("/")
        if base.endswith("/v1/traces"):
            return base
        return f"{base}/v1/traces"
