from __future__ import annotations

import unittest

from calosum import LeftHemisphereResult, PrimitiveAction, StrictLambdaRuntime, TypedLambdaProgram


class StrictRuntimeTests(unittest.TestCase):
    def test_runtime_rejects_unknown_side_effecting_actions(self) -> None:
        runtime = StrictLambdaRuntime()
        result = LeftHemisphereResult(
            response_text="test",
            lambda_program=TypedLambdaProgram(
                signature="Unit -> Unit",
                expression="(lambda _ noop)",
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
        self.assertTrue(any("unknown action type" in item for item in execution[0].violations))
        self.assertTrue(any("external side effects" in item for item in execution[0].violations))

    def test_runtime_executes_typed_plan_actions(self) -> None:
        runtime = StrictLambdaRuntime()
        result = LeftHemisphereResult(
            response_text="test",
            lambda_program=TypedLambdaProgram(
                signature="Unit -> Plan",
                expression="(lambda _ plan)",
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


if __name__ == "__main__":
    unittest.main()
