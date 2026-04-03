from __future__ import annotations

import unittest
import json

from calosum import LeftHemisphereResult, PrimitiveAction, StrictLambdaRuntime, TypedLambdaProgram


class StructuredRuntimeTests(unittest.TestCase):
    def test_runtime_executes_json_plan(self) -> None:
        runtime = StrictLambdaRuntime()
        result = LeftHemisphereResult(
            response_text="test",
            lambda_program=TypedLambdaProgram(
                signature="Context -> Decision",
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

        self.assertEqual(len(execution), 1)
        self.assertEqual(execution[0].action_type, "propose_plan")
        self.assertEqual(execution[0].status, "executed")

    def test_runtime_rejects_malformed_json(self) -> None:
        runtime = StrictLambdaRuntime()
        result = LeftHemisphereResult(
            response_text="test",
            lambda_program=TypedLambdaProgram(
                signature="Context -> Decision",
                expression="{ malformed json : [ ",
                expected_effect="failure",
            ),
            actions=[
                PrimitiveAction(
                    action_type="respond_text",
                    typed_signature="ResponsePlan -> SafeTextMessage",
                    payload={"text": "ok"},
                    safety_invariants=["text only"],
                )
            ],
            reasoning_summary=[],
        )

        execution = runtime.run(result)

        self.assertEqual(len(execution), 1)
        self.assertEqual(execution[0].status, "rejected")
        self.assertEqual(execution[0].action_type, "structured_execution")
        self.assertTrue(any("not valid JSON" in item for item in execution[0].violations))

    def test_runtime_rejects_plan_with_undeclared_action(self) -> None:
        runtime = StrictLambdaRuntime()
        result = LeftHemisphereResult(
            response_text="test",
            lambda_program=TypedLambdaProgram(
                signature="Context -> Decision",
                expression=json.dumps({"plan": ["search_web"]}),
                expected_effect="failure",
            ),
            actions=[
                PrimitiveAction(
                    action_type="respond_text",
                    typed_signature="ResponsePlan -> SafeTextMessage",
                    payload={"text": "ok"},
                    safety_invariants=["text only"],
                )
            ],
            reasoning_summary=[],
        )

        execution = runtime.run(result)

        self.assertEqual(len(execution), 1)
        self.assertEqual(execution[0].status, "rejected")
        self.assertIn("action_not_declared", execution[0].output.get("reason", ""))

    def test_runtime_rejects_unused_declared_actions(self) -> None:
        runtime = StrictLambdaRuntime()
        result = LeftHemisphereResult(
            response_text="test",
            lambda_program=TypedLambdaProgram(
                signature="Context -> Decision",
                expression=json.dumps({"plan": ["respond_text"]}),
                expected_effect="failure",
            ),
            actions=[
                PrimitiveAction(
                    action_type="respond_text",
                    typed_signature="ResponsePlan -> SafeTextMessage",
                    payload={"text": "ok"},
                    safety_invariants=["text only"],
                ),
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

        self.assertEqual(len(execution), 1)
        self.assertEqual(execution[0].status, "rejected")
        self.assertEqual(execution[0].output["reason"], "unused_declared_actions")


if __name__ == "__main__":
    unittest.main()
