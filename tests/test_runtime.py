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
                expression="lambda _: noop()",
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
                expression="lambda _: plan()",
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

    def test_runtime_rejects_actions_not_declared_by_lambda_expression(self) -> None:
        runtime = StrictLambdaRuntime()
        result = LeftHemisphereResult(
            response_text="test",
            lambda_program=TypedLambdaProgram(
                signature="Unit -> Response",
                expression="lambda ctx: respond_text()",
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
        self.assertTrue(
            any("does not reference declared action" in item for item in execution[0].violations)
        )

    def test_runtime_allows_symbolic_lambda_program_that_emits_typed_actions(self) -> None:
        runtime = StrictLambdaRuntime()
        result = LeftHemisphereResult(
            response_text="test",
            lambda_program=TypedLambdaProgram(
                signature="Context -> Memory -> Decision",
                expression=(
                    "(lambda context memory "
                    "(synthesize "
                    "(apply_soft_prompts context.bridge.soft_prompts) "
                    "(retrieve memory.semantic_rules) "
                    "(walk memory.knowledge_graph) "
                    "(emit typed_actions)))"
                ),
                expected_effect="decision",
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

        self.assertEqual(execution[0].status, "executed")

    def test_runtime_respects_lambda_sequence_order(self) -> None:
        runtime = StrictLambdaRuntime()
        result = LeftHemisphereResult(
            response_text="test",
            lambda_program=TypedLambdaProgram(
                signature="Unit -> Actions",
                expression="lambda _: sequence(load_semantic_rules(), respond_text())",
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

    def test_runtime_supports_symbolic_conditionals(self) -> None:
        runtime = StrictLambdaRuntime()
        result = LeftHemisphereResult(
            response_text="test",
            lambda_program=TypedLambdaProgram(
                signature="Context -> Decision",
                expression="(lambda context (if (has_action propose_plan) (emit propose_plan) (emit respond_text)))",
                expected_effect="conditional execution",
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

        self.assertEqual(len(execution), 1)
        self.assertEqual(execution[0].action_type, "propose_plan")
        self.assertEqual(execution[0].status, "executed")


if __name__ == "__main__":
    unittest.main()
