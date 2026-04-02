from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from calosum import (
    ActiveInferenceRightHemisphereAdapter,
    CalosumAgentBuilder,
    InfrastructureProfile,
    InfrastructureSettings,
    PersistentDualMemorySystem,
    RightHemisphereJEPA,
)
from calosum.adapters.contract_wrappers import ContractEnforcedRightHemisphereAdapter
from calosum.domain.infrastructure.telemetry import OTLPJsonlTelemetrySink


class _FakeRightHemisphere:
    def perceive(self, user_turn, memory_context=None):
        raise NotImplementedError

    async def aperceive(self, user_turn, memory_context=None):
        raise NotImplementedError


class InfrastructureBuilderTests(unittest.TestCase):
    def test_persistent_profile_builds_persistent_backends(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = InfrastructureSettings(
                profile=InfrastructureProfile.PERSISTENT,
                memory_dir=Path(temp_dir) / "memory",
                otlp_jsonl=Path(temp_dir) / "telemetry" / "events.jsonl",
            ).with_profile_defaults()

            builder = CalosumAgentBuilder(settings)
            with patch(
                "calosum.adapters.hemisphere.right_hemisphere_hf.HuggingFaceRightHemisphereAdapter",
                return_value=_FakeRightHemisphere(),
            ):
                agent = builder.build()
            description = builder.describe()

            self.assertIsInstance(agent.memory_system, PersistentDualMemorySystem)
            self.assertIsInstance(agent.telemetry_bus.sink, OTLPJsonlTelemetrySink)
            self.assertEqual(description["pattern"], "v3_dual_hemisphere_factory")
            self.assertEqual(description["profile"], "persistent")

    def test_builder_falls_back_when_hf_right_hemisphere_is_unavailable(self) -> None:
        settings = InfrastructureSettings().with_profile_defaults()
        builder = CalosumAgentBuilder(settings)

        with patch(
            "calosum.adapters.hemisphere.right_hemisphere_hf.HuggingFaceRightHemisphereAdapter",
            side_effect=RuntimeError("missing optional model stack"),
        ):
            agent = builder.build()

        description = builder.describe()
        self.assertIsInstance(agent.right_hemisphere, ActiveInferenceRightHemisphereAdapter)
        self.assertIsInstance(agent.right_hemisphere.base_adapter, ContractEnforcedRightHemisphereAdapter)
        self.assertIsInstance(agent.right_hemisphere.base_adapter.provider, RightHemisphereJEPA)
        self.assertEqual(description["right_hemisphere_backend"], "active_inference_heuristic_fallback")
        self.assertEqual(
            getattr(agent.right_hemisphere.base_adapter.provider, "degraded_reason", None),
            "hf_stack_unavailable:RuntimeError",
        )

    def test_docker_profile_resolves_infra_defaults(self) -> None:
        settings = InfrastructureSettings(profile=InfrastructureProfile.DOCKER).with_profile_defaults()
        builder = CalosumAgentBuilder(settings)
        description = builder.describe()

        self.assertEqual(description["memory_backend"], "qdrant_vector_memory")
        self.assertEqual(description["vector_db_url"], "http://qdrant:6333")
        self.assertEqual(description["otel_endpoint"], "http://otel-collector:4318")
        self.assertEqual(description["routing_resolution"]["reflection"]["shared"], True)
        self.assertIn(
            description["knowledge_graph_backend"],
            {"networkx_graph_rag_fallback", "nanorag_compatible_networkx"},
        )

    def test_builder_describes_openai_responses_backend_when_openai_base_url_is_configured(self) -> None:
        settings = InfrastructureSettings(
            left_hemisphere_endpoint="https://api.openai.com/v1",
            left_hemisphere_api_key="sk-test",
            left_hemisphere_model="gpt-5-mini",
        ).with_profile_defaults()
        builder = CalosumAgentBuilder(settings)
        description = builder.describe()

        self.assertEqual(description["left_hemisphere_backend"], "openai_responses_adapter")

    def test_builder_derives_embedding_backend_from_openai_settings_for_qdrant(self) -> None:
        settings = InfrastructureSettings(
            profile=InfrastructureProfile.DOCKER,
            vector_db_url="http://qdrant:6333",
            left_hemisphere_endpoint="https://api.openai.com/v1",
            left_hemisphere_api_key="sk-test",
        ).with_profile_defaults()
        builder = CalosumAgentBuilder(settings)
        description = builder.describe()

        self.assertEqual(description["capabilities"]["embeddings"]["provider"], "openai")
        self.assertEqual(description["capabilities"]["embeddings"]["model_name"], "text-embedding-3-small")
        self.assertEqual(description["embedding_backend"], "openai")

    def test_builder_describes_failover_backend_when_fallback_endpoint_is_configured(self) -> None:
        settings = InfrastructureSettings(
            left_hemisphere_endpoint="http://primary.local/v1/chat/completions",
            left_hemisphere_model="primary-model",
            left_hemisphere_fallback_endpoint="http://secondary.local/v1/chat/completions",
            left_hemisphere_fallback_model="secondary-model",
        ).with_profile_defaults()
        builder = CalosumAgentBuilder(settings)
        description = builder.describe()

        self.assertEqual(description["capabilities"]["left_hemisphere"]["backend"], "resilient_failover_adapter")
        self.assertTrue(description["left_hemisphere_failover"])

    def test_builder_extracts_capability_snapshot(self) -> None:
        settings = InfrastructureSettings(
            left_hemisphere_endpoint="https://api.openai.com/v1",
            left_hemisphere_model="gpt-4o",
            left_hemisphere_provider="openai",
        ).with_profile_defaults()
        builder = CalosumAgentBuilder(settings)
        snapshot = builder.build_capability_snapshot()

        self.assertEqual(snapshot.left_hemisphere.model_name, "gpt-4o")
        self.assertEqual(snapshot.left_hemisphere.provider, "openai")
        self.assertEqual(snapshot.left_hemisphere.backend, "openai_responses_adapter")
        
        self.assertEqual(snapshot.right_hemisphere.model_name, "jepa")
        self.assertEqual(snapshot.knowledge_graph.model_name, "nanorag")

        # The description should contain the snapshot serialized
        description = builder.describe()
        self.assertIn("capabilities", description)
        self.assertEqual(description["capabilities"]["left_hemisphere"]["model_name"], "gpt-4o")

    def test_builder_uses_reason_model_for_runtime_and_describes_real_tools(self) -> None:
        settings = InfrastructureSettings(
            left_hemisphere_endpoint="https://api.openai.com/v1",
            left_hemisphere_model="gpt-4o-mini",
            left_hemisphere_provider="openai",
            reason_model="gpt-4.1-mini",
        ).with_profile_defaults()
        builder = CalosumAgentBuilder(settings)

        with patch(
            "calosum.adapters.hemisphere.right_hemisphere_hf.HuggingFaceRightHemisphereAdapter",
            return_value=_FakeRightHemisphere(),
        ):
            agent = builder.build()

        description = builder.describe(agent)
        self.assertEqual(agent.left_hemisphere.config.model_name, "gpt-4.1-mini")
        self.assertGreater(len(description["capabilities"]["tools"]), 0)
        self.assertEqual(description["routing_resolution"]["reason"]["active"], "gpt-4.1-mini")

    def test_builder_honors_jepa_routing_policy_for_perception(self) -> None:
        settings = InfrastructureSettings(
            perception_model="jepa",
        ).with_profile_defaults()
        builder = CalosumAgentBuilder(settings)

        agent = builder.build()
        description = builder.describe(agent)

        self.assertIsInstance(agent.right_hemisphere.base_adapter, ContractEnforcedRightHemisphereAdapter)
        self.assertIsInstance(agent.right_hemisphere.base_adapter.provider, RightHemisphereJEPA)
        self.assertEqual(description["right_hemisphere_backend"], "active_inference_jepa_policy")
        self.assertEqual(description["routing_resolution"]["perception"]["active"], "jepa")

    def test_factory_turboquant_flag(self) -> None:
        """CALOSUM_VECTOR_QUANTIZATION=turboquant causes _build_codec to return TurboQuantVectorCodec."""
        import os
        from calosum.bootstrap.wiring.factory import _build_codec
        from calosum.adapters.perception.quantized_embeddings import TurboQuantVectorCodec

        settings = InfrastructureSettings.from_sources(
            environ={**os.environ, "CALOSUM_VECTOR_QUANTIZATION": "turboquant", "CALOSUM_TURBOQUANT_BITS": "3"}
        )
        codec = _build_codec(settings)
        self.assertIsNotNone(codec)
        self.assertIsInstance(codec, TurboQuantVectorCodec)
        self.assertEqual(codec.bits_per_dim, 4)  # 3 + 1

    def test_factory_no_codec_when_flag_none(self) -> None:
        """Default settings produce no codec (vector_quantization=none)."""
        from calosum.bootstrap.wiring.factory import _build_codec

        env = {"CALOSUM_VECTOR_QUANTIZATION": "none", "CALOSUM_TURBOQUANT_BITS": "4"}
        settings = InfrastructureSettings.from_sources(environ=env)
        codec = _build_codec(settings)
        self.assertIsNone(codec)

    def test_mcp_settings_are_parsed_and_builder_exposes_client(self) -> None:
        settings = InfrastructureSettings.from_sources(
            environ={
                "CALOSUM_MCP_ENABLED": "true",
                "CALOSUM_MCP_SERVERS": '{"github":"http://localhost:7701/mcp"}',
                "CALOSUM_MCP_ALLOWLIST": "github",
            }
        ).with_profile_defaults()
        builder = CalosumAgentBuilder(settings)
        client = builder.build_mcp_client()
        self.assertIsNotNone(client)
        assert client is not None
        self.assertEqual(client.list_servers(), ["github"])
        self.assertEqual(client.allowlisted_servers, {"github"})

    def test_builder_attaches_interceptor_manager(self) -> None:
        builder = CalosumAgentBuilder(InfrastructureSettings().with_profile_defaults())
        agent = builder.build()
        self.assertTrue(hasattr(agent, "interceptor_manager"))


if __name__ == "__main__":
    unittest.main()
