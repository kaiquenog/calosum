from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from calosum.bootstrap.entry.api import resolve_api_settings
from calosum.bootstrap.entry.cli import _resolve_settings, build_parser
from calosum.bootstrap.infrastructure.settings import InfrastructureProfile, InfrastructureSettings
from calosum.domain.infrastructure.telemetry import OTLPJsonlTelemetrySink, TelemetryEvent


class LocalUiTelemetrySyncTests(unittest.TestCase):
    def test_cli_chat_defaults_to_persistent_local_observability(self) -> None:
        args = build_parser().parse_args(["chat"])

        with patch(
            "calosum.bootstrap.entry.cli.InfrastructureSettings.from_sources",
            return_value=InfrastructureSettings(),
        ):
            with patch.dict(os.environ, {}, clear=True):
                settings = _resolve_settings(args)

        self.assertEqual(settings.profile, InfrastructureProfile.PERSISTENT)
        self.assertEqual(settings.memory_dir, Path(".calosum-runtime/memory"))
        self.assertEqual(settings.otlp_jsonl, Path(".calosum-runtime/telemetry/events.jsonl"))

    def test_api_defaults_to_persistent_local_observability(self) -> None:
        with patch(
            "calosum.bootstrap.entry.api.InfrastructureSettings.from_sources",
            return_value=InfrastructureSettings(),
        ):
            with patch.dict(os.environ, {}, clear=True):
                settings = resolve_api_settings()

        self.assertEqual(settings.profile, InfrastructureProfile.PERSISTENT)
        self.assertEqual(settings.memory_dir, Path(".calosum-runtime/memory"))
        self.assertEqual(settings.otlp_jsonl, Path(".calosum-runtime/telemetry/events.jsonl"))

    def test_otlp_jsonl_sink_rehydrates_events_written_by_another_process(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "telemetry" / "events.jsonl"
            writer = OTLPJsonlTelemetrySink(path)
            writer.emit(
                TelemetryEvent(
                    channel="decision",
                    session_id="terminal-session",
                    turn_id="turn-1",
                    recorded_at="2026-03-28T12:00:00+00:00",
                    payload={"response_text": "OI"},
                    trace_id="abc",
                    span_id="def",
                    metrics={"latency_ms": 42.0},
                )
            )

            reader = OTLPJsonlTelemetrySink(path)
            events = reader.query(session_id="terminal-session", channel="decision")

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].payload["response_text"], "OI")
        self.assertEqual(events[0].recorded_at, "2026-03-28T12:00:00+00:00")
        self.assertEqual(events[0].metrics["latency_ms"], 42.0)


if __name__ == "__main__":
    unittest.main()
