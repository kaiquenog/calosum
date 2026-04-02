from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from calosum.adapters.hemisphere.right_hemisphere_trained_jepa import (
    TrainedJEPAAdapter,
    TrainedJEPAConfig,
)
from calosum.shared.models.types import MemoryContext, UserTurn


class TrainedJEPATests(unittest.TestCase):
    def test_adapter_falls_back_to_mean_pooling_when_checkpoint_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = TrainedJEPAAdapter(
                TrainedJEPAConfig(
                    checkpoint_dir=Path(temp_dir),
                    uncertainty_samples=3,
                    encoder_model_name="missing-local-encoder",
                )
            )
            turns = [
                UserTurn(session_id="s", user_text="Quero reorganizar o projeto."),
                UserTurn(session_id="s", user_text="Preciso de um plano com riscos e mitigacoes."),
            ]

            context = asyncio.run(adapter.encode_context(turns))
            prediction = asyncio.run(adapter.predict_response_embedding(context))

            self.assertEqual(len(prediction.predicted_embedding), 384)
            self.assertEqual(prediction.prediction_method, "mean_pooling")
            self.assertEqual(prediction.uncertainty, 1.0)
            self.assertEqual(adapter.degraded_reason, "checkpoint_missing")

    def test_compute_surprise_preserves_jepa_prediction_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = TrainedJEPAAdapter(
                TrainedJEPAConfig(
                    checkpoint_dir=Path(temp_dir),
                    encoder_model_name="missing-local-encoder",
                )
            )
            turns = [
                UserTurn(session_id="s", user_text="Estamos discutindo a entrega de uma feature."),
                UserTurn(session_id="s", user_text="Foco em prazo, riscos e testes."),
            ]
            context = asyncio.run(adapter.encode_context(turns))

            aligned = asyncio.run(adapter.compute_surprise(context, "Plano com testes e mitigacoes de risco."))
            off_topic = asyncio.run(adapter.compute_surprise(context, "Receita de lasanha com molho branco."))

            self.assertLess(aligned.score, off_topic.score)
            self.assertEqual(aligned.source, "jepa_prediction_error")

    def test_perceive_emits_trained_jepa_telemetry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = TrainedJEPAAdapter(
                TrainedJEPAConfig(
                    checkpoint_dir=Path(temp_dir),
                    encoder_model_name="missing-local-encoder",
                )
            )
            turn = UserTurn(session_id="s", user_text="Estou ansioso e preciso de estrutura.")

            state = adapter.perceive(turn, MemoryContext())

            self.assertEqual(len(state.latent_vector), 384)
            self.assertEqual(state.telemetry["right_backend"], "trained_jepa_local")
            self.assertIn("jepa_uncertainty", state.telemetry)
            self.assertIn("surprise_source", state.telemetry)
            self.assertIn("checkpoint_loaded", state.telemetry)


if __name__ == "__main__":
    unittest.main()
