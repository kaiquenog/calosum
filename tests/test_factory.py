from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from calosum import (
    CalosumAgentBuilder,
    InfrastructureProfile,
    InfrastructureSettings,
    PersistentDualMemorySystem,
)
from calosum.domain.telemetry import OTLPJsonlTelemetrySink


class InfrastructureBuilderTests(unittest.TestCase):
    def test_persistent_profile_builds_persistent_backends(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = InfrastructureSettings(
                profile=InfrastructureProfile.PERSISTENT,
                memory_dir=Path(temp_dir) / "memory",
                otlp_jsonl=Path(temp_dir) / "telemetry" / "events.jsonl",
            ).with_profile_defaults()

            builder = CalosumAgentBuilder(settings)
            agent = builder.build()
            description = builder.describe()

            self.assertIsInstance(agent.memory_system, PersistentDualMemorySystem)
            self.assertIsInstance(agent.telemetry_bus.sink, OTLPJsonlTelemetrySink)
            self.assertEqual(description["pattern"], "ports_and_adapters_with_builder_factory")
            self.assertEqual(description["profile"], "persistent")

    def test_docker_profile_resolves_infra_defaults(self) -> None:
        settings = InfrastructureSettings(profile=InfrastructureProfile.DOCKER).with_profile_defaults()
        builder = CalosumAgentBuilder(settings)
        description = builder.describe()

        self.assertEqual(description["memory_backend"], "qdrant_vector_memory")
        self.assertEqual(description["vector_db_url"], "http://qdrant:6333")
        self.assertEqual(description["otel_collector_endpoint"], "http://otel-collector:4318")
        self.assertEqual(description["jaeger_ui_url"], "http://jaeger:16686")


if __name__ == "__main__":
    unittest.main()
