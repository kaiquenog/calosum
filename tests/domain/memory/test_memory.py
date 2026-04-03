from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from calosum import (
    CalosumAgent,
    PersistentDualMemorySystem,
    UserTurn,
    ActionPlannerResult,
    TypedLambdaProgram,
    PrimitiveAction,
)


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

        class StructuralMockLeft:
            def reason(self, user_turn, bridge_packet, memory_context, runtime_feedback=None, attempt=0, workspace=None):
                # Simulate using structural knowledge from memory
                style = "standard"
                if any(t.predicate == "prefers_response_style" and t.object == "short" for t in memory_context.knowledge_triples):
                    style = "short"
                
                return ActionPlannerResult(
                    response_text="ok",
                    lambda_program=TypedLambdaProgram("", '{"plan": ["propose_plan"]}', ""),
                    actions=[
                        PrimitiveAction(
                            action_type="propose_plan",
                            typed_signature="P -> S",
                            payload={"steps": ["1"], "style": style},
                            safety_invariants=["safe"]
                        )
                    ],
                    reasoning_summary=[f"response_style={style}"],
                )

            async def areason(self, *args, **kwargs):
                return self.reason(args[0], args[1], args[2], workspace=kwargs.get("workspace"))

            def repair(self, *args, **kwargs):
                return self.reason(args[0], args[1], args[2], workspace=kwargs.get("workspace"))

            async def arepair(self, *args, **kwargs):
                return self.reason(args[0], args[1], args[2], workspace=kwargs.get("workspace"))

        agent.left_hemisphere = StructuralMockLeft()

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

    def test_recent_episode_context_is_scoped_to_same_session(self) -> None:
        agent = CalosumAgent()
        agent.process_turn(UserTurn(session_id="session-a", user_text="Preciso de um plano para bolo."))
        agent.process_turn(UserTurn(session_id="session-b", user_text="Preciso de um plano para projeto urgente."))

        result = agent.process_turn(
            UserTurn(session_id="session-a", user_text="Quero revisar o plano do bolo.")
        )

        self.assertGreaterEqual(len(result.memory_context.recent_episodes), 1)
        self.assertTrue(
            all(
                episode.user_turn.session_id == "session-a"
                for episode in result.memory_context.recent_episodes
            )
        )

    def test_sleep_mode_runs_night_trainer_automatically(self) -> None:
        class FakeNightTrainer:
            def __init__(self) -> None:
                self.invocations = 0

            def run_training_cycle(self, workspace=None):
                self.invocations += 1
                return {"status": "skipped", "reason": "test"}
            
            async def arun_training_cycle(self, workspace=None):
                return self.run_training_cycle(workspace)

        trainer = FakeNightTrainer()
        agent = CalosumAgent(night_trainer=trainer)
        agent.process_turn(UserTurn(session_id="session-trainer", user_text="Prefiro respostas curtas."))

        report = agent.sleep_mode()

        self.assertGreaterEqual(report.episodes_considered, 1)
        self.assertEqual(trainer.invocations, 1)


if __name__ == "__main__":
    unittest.main()
