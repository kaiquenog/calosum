from __future__ import annotations

import unittest

from calosum.domain.execution.execution_utils import build_structured_mismatch_signal
from calosum.shared.models.types import ActionExecutionResult, CritiqueVerdict, FailureType


class StructuredMismatchSignalTests(unittest.TestCase):
    def test_builds_signal_from_runtime_and_verifier_feedback(self) -> None:
        signal = build_structured_mismatch_signal(
            [
                ActionExecutionResult(
                    action_type="write_file",
                    typed_signature="A -> B",
                    status="rejected",
                    output={"reason": "validation_failed"},
                    violations=["payload missing required field"],
                )
            ],
            CritiqueVerdict(
                is_valid=False,
                critique_reasoning=["invalid"],
                identified_issues=["schema mismatch"],
                suggested_fixes=["repair output"],
                confidence=0.7,
                failure_types=[FailureType.SCHEMA_VIOLATION],
            ),
        )

        assert signal is not None
        self.assertEqual(signal.source, "verifier")
        self.assertIn("write_file", signal.rejected_action_types)
        self.assertIn("schema", signal.failure_types)
        self.assertGreater(signal.severity, 0.0)


if __name__ == "__main__":
    unittest.main()
