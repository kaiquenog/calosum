from __future__ import annotations

import unittest

from calosum.domain.cognition.bridge import ContextCompressor, ContextCompressorConfig
from calosum.shared.models.types import InputPerceptionState


class LatentBridgeStabilityTests(unittest.TestCase):
    def test_high_variance_inputs_produce_conservative_controls(self) -> None:
        bridge = ContextCompressor(
            ContextCompressorConfig(
                base_temperature=0.3,
                top_p_floor=0.35,
                top_p_ceiling=0.95,
            )
        )
        state = InputPerceptionState(
            context_id="ctx",
            latent_vector=[0.9, -0.4, 0.7, -0.2] + ([0.0] * 380),
            salience=0.8,
            emotional_labels=["activated"],
            world_hypotheses={"interaction_complexity": 0.7, "context_novelty": 0.85},
            confidence=0.1,
            surprise_score=0.92,
            latent_mu=[0.0] * 384,
            latent_logvar=[0.8] * 384,
            telemetry={"jepa_uncertainty": 0.95},
        )

        packet = bridge.translate(state)

        self.assertLessEqual(packet.control.target_temperature, 0.2)
        self.assertLessEqual(float(packet.control.annotations["target_top_p"]), 0.8)
        bias = packet.control.annotations["target_logit_bias"]
        self.assertGreater(float(bias["clarify_first"]), float(bias["concise_steps"]))


if __name__ == "__main__":
    unittest.main()
