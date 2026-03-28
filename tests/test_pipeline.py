from __future__ import annotations

import unittest

from calosum import CalosumAgent, Modality, MultimodalSignal, UserTurn


class PipelineIntegrationTests(unittest.TestCase):
    def test_process_turn_generates_plan_executes_actions_and_records_telemetry(self) -> None:
        agent = CalosumAgent()
        turn = UserTurn(
            session_id="pipeline-session",
            user_text="Estou frustrado e preciso de um plano urgente para reorganizar este projeto.",
            signals=[
                MultimodalSignal(
                    modality=Modality.AUDIO,
                    source="microphone",
                    payload={"transcript": "voz trêmula"},
                    metadata={"emotion": "frustrado"},
                ),
                MultimodalSignal(
                    modality=Modality.TYPING,
                    source="keyboard",
                    payload={"cadence": "fast"},
                    metadata={"emotion": "ansioso"},
                ),
            ],
        )

        result = agent.process_turn(turn)

        self.assertGreaterEqual(result.right_state.salience, 0.8)
        self.assertTrue(result.bridge_packet.control.empathy_priority)
        self.assertIn("propose_plan", [action.action_type for action in result.left_result.actions])
        self.assertTrue(all(item.status == "executed" for item in result.execution_results))
        self.assertEqual(len(agent.memory_system.episodic_store.all()), 1)

        dashboard = agent.cognitive_dashboard(turn.session_id)
        self.assertEqual(len(dashboard["felt"]), 1)
        self.assertEqual(len(dashboard["thought"]), 1)
        self.assertEqual(len(dashboard["decision"]), 1)
        self.assertEqual(len(dashboard["execution"]), 1)


if __name__ == "__main__":
    unittest.main()
