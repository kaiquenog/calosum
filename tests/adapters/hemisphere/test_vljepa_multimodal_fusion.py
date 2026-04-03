from __future__ import annotations

import unittest

from calosum.adapters.hemisphere.input_perception_vljepa import VLJepaConfig, VLJepaRightHemisphereAdapter
from calosum.shared.models.types import Modality, MultimodalSignal, UserTurn


class VLJepaMultimodalFusionTests(unittest.TestCase):
    def test_vljepa_reports_checkpoint_and_multimodal_runtime_flags(self) -> None:
        adapter = VLJepaRightHemisphereAdapter(VLJepaConfig(latent_size=32, hierarchy_levels=3))
        adapter._embedder = False

        state = adapter.perceive(
            UserTurn(
                session_id="fusion",
                user_text="Preciso combinar texto e video para entender o contexto.",
                signals=[
                    MultimodalSignal(
                        modality=Modality.VIDEO,
                        source="capture",
                        payload={"embedding": [0.1] * 32},
                    )
                ],
            )
        )

        self.assertEqual(state.telemetry["right_backend"], "vljepa_local")
        self.assertTrue(state.telemetry["multimodal_active"])
        self.assertIn("checkpoint_loaded", state.telemetry)
        self.assertEqual(state.telemetry["contract_version"], "vljepa-multimodal-v1")
        self.assertGreater(len(state.telemetry["hierarchical_features"]), 0)


if __name__ == "__main__":
    unittest.main()
