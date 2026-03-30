from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from calosum import CalosumAgent, PersistentDualMemorySystem, UserTurn


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

    def test_legacy_partial_episode_records_are_hydrated_on_reload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            memory_dir = base_dir / "memory"
            memory_dir.mkdir(parents=True, exist_ok=True)

            legacy_episode = {
                "episode_id": "legacy-1",
                "recorded_at": "2026-03-28T10:00:00+00:00",
                "user_turn": {
                    "session_id": "legacy-session",
                    "user_text": "Registro antigo sem estado completo.",
                    "signals": [],
                    "observed_at": "2026-03-28T10:00:00+00:00",
                    "turn_id": "legacy-turn",
                },
            }
            (memory_dir / "episodic.jsonl").write_text(
                json.dumps(legacy_episode, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            reloaded = PersistentDualMemorySystem.from_directory(memory_dir)
            episode = reloaded.episodic_store.all()[0]

            self.assertEqual(episode.right_state.context_id, "legacy-turn")
            self.assertEqual(episode.bridge_packet.context_id, episode.right_state.context_id)
            self.assertEqual(episode.left_result.lambda_program.signature, "Placeholder")

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

            def run_training_cycle(self):
                self.invocations += 1
                return {"status": "skipped", "reason": "test"}

        trainer = FakeNightTrainer()
        agent = CalosumAgent(night_trainer=trainer)
        agent.process_turn(UserTurn(session_id="session-trainer", user_text="Prefiro respostas curtas."))

        report = agent.sleep_mode()

        self.assertGreaterEqual(report.episodes_considered, 1)
        self.assertEqual(trainer.invocations, 1)


if __name__ == "__main__":
    unittest.main()
