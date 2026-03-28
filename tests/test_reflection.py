from __future__ import annotations

import unittest

from calosum import CalosumAgent, CognitiveVariantSpec, UserTurn


class ReflectionTests(unittest.TestCase):
    def test_group_turn_selects_variant_and_updates_tokenizer(self) -> None:
        agent = CalosumAgent()
        turn = UserTurn(
            session_id="reflection-session",
            user_text="Estou ansioso e preciso de ajuda urgente para reorganizar o projeto.",
        )
        variants = [
            CognitiveVariantSpec(
                variant_id="empathetic_low_threshold",
                tokenizer_overrides={"salience_threshold": 0.45},
            ),
            CognitiveVariantSpec(
                variant_id="strict_high_threshold",
                tokenizer_overrides={"salience_threshold": 0.9},
            ),
        ]

        result = agent.process_group_turn(turn, variants)

        self.assertEqual(result.reflection.selected_variant_id, "strict_high_threshold")
        self.assertEqual(agent.tokenizer.config.salience_threshold, 0.9)
        dashboard = agent.cognitive_dashboard(turn.session_id)
        self.assertEqual(len(dashboard["reflection"]), 1)


if __name__ == "__main__":
    unittest.main()
