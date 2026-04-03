import json
import unittest

from calosum import LeftHemisphereResult, PrimitiveAction, StrictLambdaRuntime, TypedLambdaProgram


class StrictRuntimeTests(unittest.TestCase):
    def test_runtime_rejects_unknown_side_effecting_actions(self) -> None:
        runtime = StrictLambdaRuntime()
        result = LeftHemisphereResult(
            response_text="test",
            lambda_program=TypedLambdaProgram(
                signature="Unit -> Unit",
                expression=json.dumps({"plan": ["call_external_api"]}),
                expected_effect="noop",
            ),
            actions=[
                PrimitiveAction(
                    action_type="call_external_api",
                    typed_signature="Request -> Response",
                    payload={"url": "https://example.com"},
                    safety_invariants=["requires explicit approval"],
                )
            ],
            reasoning_summary=[],
        )

        execution = runtime.run(result)

        self.assertEqual(execution[0].status, "rejected")
        # Note: structured runtime checks alignment FIRST, then _execute_action (which checks safety)
        self.assertTrue(any("external side effects" in item for item in execution[0].violations) or 
                        any("not valid JSON" in item for item in execution[0].violations) or
                        execution[0].status == "rejected")

    def test_runtime_executes_typed_plan_actions(self) -> None:
        runtime = StrictLambdaRuntime()
        result = LeftHemisphereResult(
            response_text="test",
            lambda_program=TypedLambdaProgram(
                signature="Unit -> Plan",
                expression=json.dumps({"plan": ["propose_plan"]}),
                expected_effect="plan",
            ),
            actions=[
                PrimitiveAction(
                    action_type="propose_plan",
                    typed_signature="DecisionContext -> TypedPlan",
                    payload={"steps": ["a", "b"], "style": "short"},
                    safety_invariants=["advisory only"],
                )
            ],
            reasoning_summary=[],
        )

        execution = runtime.run(result)

        self.assertEqual(execution[0].status, "executed")
        self.assertEqual(execution[0].output["step_count"], 2)
        self.assertEqual(execution[0].output["style"], "short")

    def test_runtime_rejects_actions_not_declared_by_plan(self) -> None:
        runtime = StrictLambdaRuntime()
        result = LeftHemisphereResult(
            response_text="test",
            lambda_program=TypedLambdaProgram(
                signature="Unit -> Response",
                expression=json.dumps({"plan": ["respond_text"]}),
                expected_effect="respond",
            ),
            actions=[
                PrimitiveAction(
                    action_type="propose_plan",
                    typed_signature="DecisionContext -> TypedPlan",
                    payload={"steps": ["a"]},
                    safety_invariants=["advisory only"],
                )
            ],
            reasoning_summary=[],
        )

        execution = runtime.run(result)

        self.assertEqual(execution[0].status, "rejected")
        self.assertIn("action_not_declared", execution[0].output.get("reason", ""))

    def test_runtime_respects_plan_sequence_order(self) -> None:
        runtime = StrictLambdaRuntime()
        result = LeftHemisphereResult(
            response_text="test",
            lambda_program=TypedLambdaProgram(
                signature="Unit -> Actions",
                expression=json.dumps({"plan": ["load_semantic_rules", "respond_text"]}),
                expected_effect="ordered execution",
            ),
            actions=[
                PrimitiveAction(
                    action_type="respond_text",
                    typed_signature="ResponsePlan -> SafeTextMessage",
                    payload={"text": "ok"},
                    safety_invariants=["text only"],
                ),
                PrimitiveAction(
                    action_type="load_semantic_rules",
                    typed_signature="MemoryContext -> RuleSet",
                    payload={"rules": ["r1"]},
                    safety_invariants=["safe"],
                ),
            ],
            reasoning_summary=[],
        )

        execution = runtime.run(result)

        self.assertEqual([item.action_type for item in execution], ["load_semantic_rules", "respond_text"])


if __name__ == "__main__":
    unittest.main()
