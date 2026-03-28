from __future__ import annotations

import unittest

from calosum import CalosumAgent, UserTurn


class DualMemoryTests(unittest.TestCase):
    def test_sleep_mode_promotes_rules_and_graph_preferences(self) -> None:
        agent = CalosumAgent()

        repeated_preference = "Prefiro respostas curtas com passos claros quando a situacao estiver urgente."
        agent.process_turn(UserTurn(session_id="memory-session", user_text=repeated_preference))
        agent.process_turn(UserTurn(session_id="memory-session", user_text=repeated_preference))

        report = agent.sleep_mode()

        rule_ids = [rule.rule_id for rule in report.promoted_rules]
        self.assertIn("emotion::urgente", rule_ids)
        self.assertTrue(any(rule.rule_id.startswith("preference::") for rule in report.promoted_rules))
        self.assertTrue(
            any(
                triple.predicate == "prefers_response_style" and triple.object == "short"
                for triple in report.graph_updates
            )
        )
        self.assertTrue(
            any(
                triple.predicate == "prefers_structure" and triple.object == "stepwise"
                for triple in report.graph_updates
            )
        )

        result = agent.process_turn(
            UserTurn(
                session_id="memory-session",
                user_text="Preciso de um plano urgente para reorganizar o projeto.",
            )
        )

        self.assertIn("response_style=short", result.left_result.reasoning_summary)
        plan_action = next(
            action for action in result.left_result.actions if action.action_type == "propose_plan"
        )
        self.assertEqual(plan_action.payload["style"], "short")
        self.assertGreaterEqual(len(result.memory_context.knowledge_triples), 1)


if __name__ == "__main__":
    unittest.main()
