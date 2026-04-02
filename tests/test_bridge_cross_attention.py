from __future__ import annotations

import unittest

from calosum.adapters.bridge.bridge_cross_attention import CrossAttentionBridgeAdapter
from calosum.domain.bridge import CognitiveTokenizer
from calosum.shared.types import RightHemisphereState


class CrossAttentionBridgeTests(unittest.TestCase):
    def test_cross_attention_fusion_returns_metadata_and_stable_shape(self) -> None:
        adapter = CrossAttentionBridgeAdapter()
        latent = [0.01 * i for i in range(384)]

        fused, meta = adapter.fuse_latent(latent_vector=latent, emotional_labels=["ansioso", "urgente"])

        self.assertEqual(len(fused), 384)
        self.assertIn(meta["fusion_backend"], {"cross_attention_heuristic", "learned_cross_attention"})
        self.assertIn("attention_entropy", meta)

    def test_tokenizer_applies_fusion_when_configured(self) -> None:
        tokenizer = CognitiveTokenizer(fusion=CrossAttentionBridgeAdapter())
        state = RightHemisphereState(
            context_id="c1",
            latent_vector=[0.01 * i for i in range(384)],
            salience=0.6,
            emotional_labels=["ansioso"],
            world_hypotheses={"interaction_complexity": 0.5},
            confidence=0.8,
            surprise_score=0.3,
        )

        packet = tokenizer.translate(state)

        self.assertEqual(len(packet.latent_vector), 384)
        self.assertEqual(packet.control.annotations["neural_active"], tokenizer.use_neural)


if __name__ == "__main__":
    unittest.main()
