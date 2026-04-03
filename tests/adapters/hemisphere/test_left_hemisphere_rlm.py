from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from calosum.adapters.hemisphere.action_planner_rlm import RlmAdapterConfig, RlmLeftHemisphereAdapter
from calosum.shared.models.types import (
    BridgeControlSignal,
    PerceptionSummary,
    MemoryContext,
    SoftPromptToken,
    TypedLambdaProgram,
    UserTurn,
)


def _bridge_packet() -> PerceptionSummary:
    return PerceptionSummary(
        context_id="ctx",
        soft_prompts=[SoftPromptToken(token="<affect:neutral>", weight=0.5, provenance="test")],
        control=BridgeControlSignal(target_temperature=0.2, empathy_priority=False, system_directives=[]),
        salience=0.4,
        latent_vector=[0.1, 0.2, 0.3],
    )


class RlmAdapterTests(unittest.TestCase):
    def test_local_recursive_mode_returns_typed_actions(self) -> None:
        adapter = RlmLeftHemisphereAdapter(RlmAdapterConfig(max_depth=2))
        result = adapter.reason(
            UserTurn(session_id="s", user_text="Preciso de um plano em passos curtos para reorganizar o projeto."),
            _bridge_packet(),
            MemoryContext(),
        )

        self.assertIn(result.telemetry["backend"], {"rlm_local_recursive", "rlm_runtime"})
        self.assertGreaterEqual(len(result.actions), 1)
        self.assertTrue(any(action.action_type == "respond_text" for action in result.actions))
        self.assertIsInstance(result.lambda_program, TypedLambdaProgram)

    def test_runtime_command_mode_parses_real_json_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            script = Path(temp_dir) / "fake_rlm.py"
            script.write_text(
                """
#!/usr/bin/env python3
import json, sys
_ = json.loads(sys.stdin.read() or "{}")
print(json.dumps({
    "response_text": "Plano recursivo pronto.",
    "lambda_expression": "(lambda context memory (sequence (emit respond_text)))",
    "reasoning_summary": ["depth=2"],
    "actions": [
        {
            "action_type": "respond_text",
            "typed_signature": "ResponsePlan -> SafeTextMessage",
            "payload": {"text": "Plano recursivo pronto."},
            "safety_invariants": ["safe output only"]
        }
    ]
}))
""".strip(),
                encoding="utf-8",
            )
            os.chmod(script, 0o755)

            adapter = RlmLeftHemisphereAdapter(
                RlmAdapterConfig(runtime_command=str(script), max_depth=2)
            )
            result = adapter.reason(
                UserTurn(session_id="s", user_text="gerar plano"),
                _bridge_packet(),
                MemoryContext(),
            )

            self.assertEqual(result.response_text, "Plano recursivo pronto.")
            self.assertEqual(result.actions[0].action_type, "respond_text")
            self.assertEqual(result.telemetry["backend"], "rlm_runtime")


if __name__ == "__main__":
    unittest.main()
