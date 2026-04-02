from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from calosum import CalosumAgent, CognitiveVariantSpec, UserTurn
from calosum.adapters.experience.gea_reflection_experience import (
    LearnedPreferenceGEAReflectionController,
)
from calosum.adapters.experience.variant_preference import (
    VariantPreferenceDatasetStore,
    VariantPreferenceModel,
    VariantTrainingExample,
)


class VariantPreferenceTests(unittest.TestCase):
    def test_rule_based_prefers_empatico_for_high_emotional_intensity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            controller = LearnedPreferenceGEAReflectionController(
                dataset_path=base / "dataset.jsonl",
                model_path=base / "model.joblib",
            )
            agent = CalosumAgent(reflection_controller=controller)
            turn = UserTurn(
                session_id="emotion-session",
                user_text="Estou ansioso e frustrado, preciso de ajuda agora.",
            )
            variants = [
                CognitiveVariantSpec(variant_id="analitico"),
                CognitiveVariantSpec(variant_id="empatico"),
                CognitiveVariantSpec(variant_id="pragmatico"),
            ]

            result = agent.process_group_turn(turn, variants)

            self.assertEqual(result.reflection.selected_by, "rule_based")
            self.assertEqual(result.reflection.selected_variant_id, "empatico")
            self.assertEqual(controller.dataset_store.count(), 1)
            dashboard = agent.cognitive_dashboard(turn.session_id)
            self.assertEqual(dashboard["reflection"][0]["selected_by"], "rule_based")

    def test_legacy_fallback_when_variants_do_not_match_personas(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            controller = LearnedPreferenceGEAReflectionController(
                dataset_path=base / "dataset.jsonl",
                model_path=base / "model.joblib",
            )
            agent = CalosumAgent(reflection_controller=controller)
            turn = UserTurn(session_id="legacy-session", user_text="Pedido factual curto.")
            variants = [
                CognitiveVariantSpec(variant_id="a"),
                CognitiveVariantSpec(variant_id="b"),
                CognitiveVariantSpec(variant_id="c"),
            ]

            result = agent.process_group_turn(turn, variants)

            self.assertEqual(result.reflection.selected_by, "legacy")
            self.assertIn(result.reflection.selected_variant_id, {"a", "b", "c"})

    def test_training_requires_minimum_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = VariantPreferenceDatasetStore(Path(temp_dir) / "dataset.jsonl")
            model = VariantPreferenceModel(
                artifact_path=Path(temp_dir) / "model.joblib",
                min_samples=200,
            )

            for idx in range(10):
                dataset.append(
                    VariantTrainingExample(
                        session_id="s1",
                        turn_id=f"t{idx}",
                        recorded_at="2026-04-02T00:00:00+00:00",
                        variant_scores={"analitico": 1.0, "empatico": 1.0, "pragmatico": 1.0},
                        selected_variant="analitico",
                        response_rating=0.7,
                        context={
                            "intent_type": "technical",
                            "surprise_score": 0.2,
                            "ambiguity_score": 0.3,
                            "session_length": 4,
                            "avg_tool_success_rate": 0.9,
                            "jepa_uncertainty": 0.1,
                        },
                    )
                )

            report = model.train(dataset.read_all())
            self.assertFalse(report.trained)
            self.assertIn("insufficient_samples", report.reason or "")


if __name__ == "__main__":
    unittest.main()
