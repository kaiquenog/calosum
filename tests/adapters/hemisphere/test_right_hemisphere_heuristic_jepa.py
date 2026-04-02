from __future__ import annotations

import asyncio
import unittest

from calosum.adapters.hemisphere.right_hemisphere_heuristic_jepa import (
    HeuristicJEPAAdapter,
    HeuristicJEPAConfig,
)
from calosum.shared.models.types import MemoryContext, UserTurn


class HeuristicJEPATests(unittest.TestCase):
    def test_predict_response_embedding_returns_384_dim_with_method(self) -> None:
        adapter = HeuristicJEPAAdapter()
        turns = [
            UserTurn(session_id="s", user_text="Quero reorganizar o projeto."),
            UserTurn(session_id="s", user_text="Preciso de passos claros e objetivos."),
        ]

        context = asyncio.run(adapter.encode_context(turns))
        prediction = asyncio.run(adapter.predict_response_embedding(context))

        self.assertEqual(len(prediction.predicted_embedding), 384)
        self.assertEqual(prediction.prediction_method, "jepa_heuristic")
        self.assertGreaterEqual(prediction.uncertainty, 0.0)
        self.assertLessEqual(prediction.uncertainty, 1.0)

    def test_compute_surprise_uses_prediction_error(self) -> None:
        adapter = HeuristicJEPAAdapter()
        turns = [
            UserTurn(session_id="s", user_text="Estamos definindo um plano tecnico."),
            UserTurn(session_id="s", user_text="Objetivo: reduzir riscos de entrega."),
        ]
        context = asyncio.run(adapter.encode_context(turns))

        aligned = asyncio.run(adapter.compute_surprise(context, "Plano tecnico com riscos e mitigacoes."))
        off_topic = asyncio.run(adapter.compute_surprise(context, "Receita de bolo de chocolate com cobertura."))

        self.assertLess(aligned.score, off_topic.score)
        self.assertEqual(aligned.source, "jepa_prediction_error")
        self.assertEqual(aligned.prediction_method, "jepa_heuristic")

    def test_high_uncertainty_sets_ignore_flag(self) -> None:
        adapter = HeuristicJEPAAdapter(
            HeuristicJEPAConfig(uncertainty_ignore_threshold=0.7)
        )
        turns = [UserTurn(session_id="s", user_text="Mensagem isolada sem contexto.")]
        context = asyncio.run(adapter.encode_context(turns))
        surprise = asyncio.run(adapter.compute_surprise(context, "Resposta qualquer"))

        self.assertTrue(surprise.ignored_due_to_uncertainty)
        self.assertGreaterEqual(surprise.score, 0.0)

    def test_perceive_emits_prediction_and_surprise_telemetry(self) -> None:
        adapter = HeuristicJEPAAdapter()
        turn = UserTurn(session_id="s", user_text="Estou ansioso e preciso de estrutura.")

        state = adapter.perceive(turn, MemoryContext())

        self.assertEqual(len(state.latent_vector), 384)
        self.assertIn("prediction_method", state.telemetry)
        self.assertIn("surprise_source", state.telemetry)
        self.assertIn("jepa_uncertainty", state.telemetry)
        self.assertIn("surprise_band", state.telemetry)
        self.assertEqual(state.telemetry["prediction_method"], "jepa_heuristic")
        self.assertEqual(state.telemetry["surprise_source"], "jepa_prediction_error")


if __name__ == "__main__":
    unittest.main()
