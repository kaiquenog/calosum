from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from calosum import CalosumAgentBuilder, InfrastructureSettings
from calosum.adapters.infrastructure.contract_wrappers import ContractEnforcedLeftHemisphereAdapter
from calosum.bootstrap.infrastructure.settings import CalosumMode


class FactoryBackends2026Tests(unittest.TestCase):
    def test_builder_selects_vjepa_and_rlm_and_cross_attention(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = InfrastructureSettings(left_hemisphere_endpoint="http://test", 
                mode=CalosumMode.LOCAL,
                right_hemisphere_backend="vjepa21",
                right_model_path=Path(temp_dir),
                left_hemisphere_backend="rlm",
                left_rlm_max_depth=2,
                bridge_backend="cross_attention",
                gea_sharing_enabled=True,
                gea_experience_store_path=Path(temp_dir) / "gea.sqlite",
            ).with_profile_defaults()

            builder = CalosumAgentBuilder(settings)
            agent = builder.build()
            description = builder.describe(agent)

            self.assertEqual(description["right_hemisphere_backend"], "vjepa21_local")
            self.assertEqual(description["left_hemisphere_backend"], "rlm_recursive_adapter")
            self.assertIsInstance(agent.left_hemisphere, ContractEnforcedLeftHemisphereAdapter)
            self.assertEqual(
                agent.left_hemisphere.provider.__class__.__name__,
                "RlmAstLeftHemisphereAdapter"
            )
            self.assertEqual(agent.reflection_controller.__class__.__name__, "GEAReflectionController")

    def test_builder_supports_jepars_backend_selection(self) -> None:
        settings = InfrastructureSettings(left_hemisphere_endpoint="http://test", 
            right_hemisphere_backend="jepars",
            right_jepars_binary="jepa-rs",
        ).with_profile_defaults()

        builder = CalosumAgentBuilder(settings)
        agent = builder.build()
        description = builder.describe(agent)

        self.assertEqual(description["right_hemisphere_backend"], "jepars_local")

    def test_builder_uses_trained_jepa_by_default_in_local_mode_when_available(self) -> None:
        class _FakeTrainedJEPAAdapter:
            def __init__(self):
                self.is_available = True
                self.config = type("Cfg", (), {"model_name": "trained-jepa-v1.0"})()

        settings = InfrastructureSettings(left_hemisphere_endpoint="http://test", 
            mode=CalosumMode.LOCAL,
            right_hemisphere_backend="auto",
        ).with_profile_defaults()
        builder = CalosumAgentBuilder(settings)

        with patch(
            "calosum.bootstrap.wiring.backend_resolvers.TrainedJEPAAdapter",
            _FakeTrainedJEPAAdapter,
        ):
            agent = builder.build()
            description = builder.describe(agent)

        self.assertEqual(description["right_hemisphere_backend"], "predictive_checkpoint")
    def test_builder_falls_back_to_heuristic_when_trained_jepa_backend_is_unavailable(self) -> None:
        class _UnavailableTrainedJEPAAdapter:
            def __init__(self):
                self.is_available = False
                self.degraded_reason = "checkpoint_missing"
                self.config = type("Cfg", (), {"model_name": "trained-jepa-v1.0"})()

        settings = InfrastructureSettings(left_hemisphere_endpoint="http://test", 
            mode=CalosumMode.LOCAL,
            right_hemisphere_backend="trained_jepa",
        ).with_profile_defaults()
        builder = CalosumAgentBuilder(settings)

        with patch(
            "calosum.bootstrap.wiring.backend_resolvers.TrainedJEPAAdapter",
            _UnavailableTrainedJEPAAdapter,
        ):
            agent = builder.build()
            description = builder.describe(agent)

        self.assertEqual(description["right_hemisphere_backend"], "heuristic_literal_fallback")

if __name__ == "__main__":
    unittest.main()
