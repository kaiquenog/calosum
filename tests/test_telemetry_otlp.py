from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from calosum.adapters.telemetry_otlp import OTLPHTTPTraceSink
from calosum.domain.telemetry import TelemetryEvent


class OtlpTelemetryAdapterTests(unittest.TestCase):
    def test_http_trace_sink_posts_otlp_trace_payload(self) -> None:
        sink = OTLPHTTPTraceSink("http://collector:4318")
        event = TelemetryEvent(
            channel="decision",
            session_id="session-1",
            turn_id="turn-1",
            recorded_at="2026-03-29T22:50:28.891904+00:00",
            payload={"response_text": "ok"},
            trace_id="553556b9d4880c3db0c28327c1b38fb1",
            span_id="ed810801947e0625",
            metrics={"latency_ms": 12.5},
        )

        response = Mock()
        response.raise_for_status = Mock()

        with patch("calosum.adapters.telemetry_otlp.httpx.post", return_value=response) as post:
            sink.emit(event)

        post.assert_called_once()
        args, kwargs = post.call_args
        self.assertEqual(args[0], "http://collector:4318/v1/traces")
        self.assertEqual(kwargs["headers"]["Content-Type"], "application/json")
        payload = kwargs["json"]
        span = payload["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        self.assertEqual(span["name"], "decision")
        self.assertEqual(span["traceId"], event.trace_id)
        self.assertEqual(span["spanId"], event.span_id)
        self.assertIn("startTimeUnixNano", span)
        self.assertIn("endTimeUnixNano", span)

    def test_http_trace_sink_swallows_export_failures(self) -> None:
        sink = OTLPHTTPTraceSink("http://collector:4318")
        event = TelemetryEvent(
            channel="felt",
            session_id="session-1",
            turn_id="turn-1",
            recorded_at="2026-03-29T22:50:28.891904+00:00",
            payload={"surprise_score": 0.2},
            trace_id="553556b9d4880c3db0c28327c1b38fb1",
            span_id="29a38a4115211474",
        )

        with patch("calosum.adapters.telemetry_otlp.httpx.post", side_effect=RuntimeError("collector down")):
            sink.emit(event)


if __name__ == "__main__":
    unittest.main()
