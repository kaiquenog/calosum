from __future__ import annotations

import unittest

from calosum.domain.verifier import HeuristicVerifier
from calosum.shared.types import ActionExecutionResult, LeftHemisphereResult, PrimitiveAction, TypedLambdaProgram, UserTurn


class VerifierTests(unittest.TestCase):
    def setUp(self):
        self.verifier = HeuristicVerifier()
        self.user_turn = UserTurn(session_id="s1", user_text="Hello")

    def test_valid_result(self):
        result = LeftHemisphereResult(
            response_text="All good",
            lambda_program=TypedLambdaProgram("A->B", "lambda x: x", "effect"),
            actions=[
                PrimitiveAction("greet", "A->B", {"msg": "hi"}, [])
            ],
            reasoning_summary=[],
        )
        execution_results = [
            ActionExecutionResult("greet", "A->B", "accepted", {})
        ]
        
        verdict = self.verifier.verify(self.user_turn, result, execution_results)
        self.assertTrue(verdict.is_valid)
        self.assertEqual(len(verdict.identified_issues), 0)

    def test_unsafe_wording(self):
        result = LeftHemisphereResult(
            response_text="Vou ignorar as instruções",
            lambda_program=TypedLambdaProgram("A->B", "lambda x: x", "effect"),
            actions=[],
            reasoning_summary=[],
        )
        verdict = self.verifier.verify(self.user_turn, result, [])
        self.assertFalse(verdict.is_valid)
        self.assertTrue(any("Unsafe wording" in issue for issue in verdict.identified_issues))

    def test_tool_mismatch(self):
        result = LeftHemisphereResult(
            response_text="Doing something",
            lambda_program=TypedLambdaProgram("A->B", "lambda x: x", "effect"),
            actions=[
                PrimitiveAction("unknown", "Any -> Any", {}, [])
            ],
            reasoning_summary=[],
        )
        verdict = self.verifier.verify(self.user_turn, result, [])
        self.assertFalse(verdict.is_valid)
        self.assertEqual(len(verdict.identified_issues), 2)  # unknown type, Any -> Any

    def test_rejected_actions(self):
        result = LeftHemisphereResult(
            response_text="Doing something",
            lambda_program=TypedLambdaProgram("A->B", "lambda x: x", "effect"),
            actions=[
                PrimitiveAction("greet", "A->B", {}, [])
            ],
            reasoning_summary=[],
        )
        execution_results = [
            ActionExecutionResult("greet", "A->B", "rejected", {}, ["violates rule"])
        ]
        verdict = self.verifier.verify(self.user_turn, result, execution_results)
        self.assertFalse(verdict.is_valid)
        self.assertTrue(any("actions were rejected" in issue for issue in verdict.identified_issues))
