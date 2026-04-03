from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from calosum.adapters.communication.telemetry_otlp import OTLPHTTPTraceSink
from calosum.domain.infrastructure.telemetry import TelemetryEvent


class OtlpBackoffTests(unittest.TestCase):
    def test_sink_skips_emission_during_backoff_window(self) -> None:
        sink = OTLPHTTPTraceSink("http://collector:4318", base_backoff_s=2.0, max_backoff_s=4.0)
        event = TelemetryEvent(
            channel="decision",
            session_id="session-1",
            turn_id="turn-1",
            recorded_at="2026-03-29T22:50:28.891904+00:00",
            payload={"response_text": "ok"},
            trace_id="553556b9d4880c3db0c28327c1b38fb1",
            span_id="ed810801947e0625",
        )

        with patch("calosum.adapters.communication.telemetry_otlp.monotonic", side_effect=[0.0, 0.0, 1.0]):
            with patch("calosum.adapters.communication.telemetry_otlp.httpx.post", side_effect=RuntimeError("collector down")) as post:
                sink.emit(event)
                sink.emit(event)

        self.assertEqual(post.call_count, 1)


if __name__ == "__main__":
    unittest.main()
