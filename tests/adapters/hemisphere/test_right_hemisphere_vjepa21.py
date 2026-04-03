from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from calosum.adapters.hemisphere.input_perception_jepars import JepaRsConfig, JepaRsRightHemisphereAdapter
from calosum.adapters.hemisphere.input_perception_vjepa21 import VJepa21Config, VJepa21RightHemisphereAdapter
from calosum.adapters.hemisphere.input_perception_vljepa import VLJepaConfig, VLJepaRightHemisphereAdapter
from calosum.shared.models.types import PerceptionStatus, UserTurn


class RightHemisphere2026AdaptersTests(unittest.TestCase):
    def test_vjepa21_adapter_emits_predictive_telemetry(self) -> None:
        adapter = VJepa21RightHemisphereAdapter(VJepa21Config(latent_size=32, horizon=3))
        adapter._embedder = False

        user_turn = UserTurn(session_id="s", user_text="Estou ansioso e preciso de um plano.")
        state = adapter.perceive(user_turn)
        state_second = adapter.perceive(user_turn)

        self.assertEqual(state.telemetry["right_backend"], "vjepa21_local")
        self.assertEqual(state.telemetry["right_mode"], "predictive")
        self.assertIn(state.perception_status, (PerceptionStatus.OBSERVED, PerceptionStatus.DEGRADED))
        self.assertGreater(len(state.latent_vector), 0)
        self.assertEqual(state.latent_vector, state_second.latent_vector)
        self.assertGreaterEqual(state.surprise_score, 0.0)
        self.assertLessEqual(state.surprise_score, 1.0)

    def test_vjepa21_predictor_training_skips_when_predictor_is_unavailable(self) -> None:
        adapter = VJepa21RightHemisphereAdapter(VJepa21Config(latent_size=32, horizon=3))
        adapter._predictor = None

        report = adapter.train_predictor_from_records(
            [{"latent_t": [0.1, 0.2], "latent_t1": [0.2, 0.3], "prediction_error": 0.8}]
        )

        self.assertEqual(report["status"], "skipped")
        self.assertEqual(report["reason"], "predictor_unavailable")

    def test_vljepa_adapter_adds_hierarchical_features(self) -> None:
        adapter = VLJepaRightHemisphereAdapter(VLJepaConfig(latent_size=32, hierarchy_levels=2))
        adapter._embedder = False

        state = adapter.perceive(UserTurn(session_id="s", user_text="Contexto multimodal urgente"))

        self.assertEqual(state.telemetry["right_backend"], "vljepa_local")
        self.assertIn("hierarchical_features", state.telemetry)
        self.assertIn("dense_semantic_energy", state.world_hypotheses)

    def test_jepars_adapter_invokes_real_subprocess_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            script = Path(temp_dir) / "fake_jepars.py"
            script.write_text(
                """
#!/usr/bin/env python3
import json, sys
_ = json.loads(sys.stdin.read() or "{}")
print(json.dumps({
    "latent_vector": [0.1, -0.2, 0.3, 0.4],
    "salience": 0.6,
    "confidence": 0.8,
    "surprise_score": 0.2,
    "emotional_labels": ["neutral"]
}))
""".strip(),
                encoding="utf-8",
            )
            os.chmod(script, 0o755)
            adapter = JepaRsRightHemisphereAdapter(
                JepaRsConfig(binary_path=str(script), timeout_seconds=10.0)
            )

            state = adapter.perceive(UserTurn(session_id="s", user_text="teste"))

            self.assertEqual(state.telemetry["right_backend"], "jepa_rs_arrow")
            self.assertEqual(len(state.latent_vector), 4)


if __name__ == "__main__":
    unittest.main()
