from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from calosum import CalosumAgentBuilder, InfrastructureSettings
from calosum.adapters.infrastructure.contract_wrappers import ContractEnforcedLeftHemisphereAdapter


class FactoryBackends2026Tests(unittest.TestCase):
    def test_builder_selects_vjepa_and_rlm_and_cross_attention(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = InfrastructureSettings(
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

            self.assertEqual(description["right_hemisphere_backend"], "active_inference_vjepa21")
            self.assertEqual(description["left_hemisphere_backend"], "rlm_recursive_adapter")
            self.assertIsInstance(agent.left_hemisphere, ContractEnforcedLeftHemisphereAdapter)
            self.assertEqual(
                agent.left_hemisphere.provider.__class__.__name__,
                "RlmLeftHemisphereAdapter",
            )
            self.assertEqual(agent.reflection_controller.__class__.__name__, "ExperienceAwareGEAReflectionController")

    def test_builder_supports_jepars_backend_selection(self) -> None:
        settings = InfrastructureSettings(
            right_hemisphere_backend="jepars",
            right_jepars_binary="jepa-rs",
        ).with_profile_defaults()

        builder = CalosumAgentBuilder(settings)
        agent = builder.build()
        description = builder.describe(agent)

        self.assertEqual(description["right_hemisphere_backend"], "active_inference_jepars")


if __name__ == "__main__":
    unittest.main()
