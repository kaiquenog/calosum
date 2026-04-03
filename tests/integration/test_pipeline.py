from __future__ import annotations

import unittest

from calosum import CalosumAgent, ActionPlannerResult, Modality, MultimodalSignal, PrimitiveAction, TypedLambdaProgram, UserTurn

class MockLeftHemisphere:
    def reason(self, user_turn, bridge_packet, memory_context, runtime_feedback=None, attempt=0):
        return ActionPlannerResult(
            response_text="Mocked plan",
            lambda_program=TypedLambdaProgram("Context -> Plan", "lambda _: propose_plan()", "plan"),
            actions=[
                PrimitiveAction(
                    action_type="propose_plan",
                    typed_signature="Context -> Plan",
                    payload={"steps": ["1"]},
                    safety_invariants=["safe"]
                )
            ],
            reasoning_summary=[],
        )

    async def areason(self, *args, **kwargs):
        return self.reason(*args[:3])

    def repair(self, *args, **kwargs):
        return self.reason(*args[:3])

    async def arepair(self, *args, **kwargs):
        return self.reason(*args[:3])

class PipelineIntegrationTests(unittest.TestCase):
    def test_process_turn_generates_plan_executes_actions_and_records_telemetry(self) -> None:
        agent = CalosumAgent(left_hemisphere=MockLeftHemisphere())
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

        # Check telemetry enrichment (Sprint 0)
        thought_event = dashboard["thought"][0]
        self.assertIn("bridge_config", thought_event)
        self.assertIn("target_temperature", thought_event["bridge_config"])
        self.assertIn("active_variant", thought_event)
        self.assertIn("cognitive_override_detected", thought_event)

        decision_event = dashboard["decision"][0]
        self.assertIn("capabilities", decision_event)

        workspace = agent.workspace_for_session(turn.session_id)
        self.assertIsNotNone(workspace)
        assert workspace is not None
        self.assertGreater(len(workspace.runtime_feedback), 0)

if __name__ == "__main__":
    unittest.main()
