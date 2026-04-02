from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from calosum import CalosumAgent, CognitiveVariantSpec, UserTurn
from calosum.adapters.experience.gea_experience_store import GeaExperienceStoreConfig, SqliteGeaExperienceStore
from calosum.adapters.experience.gea_reflection_experience import ExperienceAwareGEAReflectionController


class GeaExperienceSharingTests(unittest.TestCase):
    def test_experience_store_records_and_returns_priors(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SqliteGeaExperienceStore(GeaExperienceStoreConfig(path=Path(temp_dir) / "gea.sqlite"))
            store.record_experience(
                context_type="technical",
                variant_id="analitico",
                score=1.2,
                reward=0.8,
                metadata={"test": True},
            )

            prior = store.variant_prior(context_type="technical", variant_id="analitico")
            stats = store.context_stats(context_type="technical")

            self.assertGreater(prior, 0.0)
            self.assertEqual(stats["variants"][0]["variant_id"], "analitico")

    def test_reflection_controller_persists_experience_after_group_turn(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SqliteGeaExperienceStore(GeaExperienceStoreConfig(path=Path(temp_dir) / "gea.sqlite"))
            controller = ExperienceAwareGEAReflectionController(experience_store=store)
            agent = CalosumAgent(reflection_controller=controller)

            turn = UserTurn(session_id="gea-session", user_text="Estou ansioso e preciso de um plano tecnico.")
            variants = [
                CognitiveVariantSpec(variant_id="analitico"),
                CognitiveVariantSpec(variant_id="empatico"),
            ]

            result = agent.process_group_turn(turn, variants)
            context_type = controller._infer_context_type(result.candidates)
            stats = store.context_stats(context_type=context_type)

            self.assertGreater(len(stats["variants"]), 0)
            self.assertIn(result.reflection.selected_variant_id, {v["variant_id"] for v in stats["variants"]})


if __name__ == "__main__":
    unittest.main()
