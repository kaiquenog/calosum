from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from calosum.adapters.execution.action_runtime import ConcreteActionRuntime
from calosum.bootstrap.wiring.agent_baseline import AgentBaseline, AgentBaselineConfig
from calosum.shared.models.types import (
    LeftHemisphereResult,
    PrimitiveAction,
    TypedLambdaProgram,
    UserTurn,
)


class _FakeEmbedder:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 32 for _ in texts]


class _FakeLeftHemisphere:
    def reason(self, user_turn, bridge_packet, memory_context, runtime_feedback=None, attempt=0, workspace=None):
        return LeftHemisphereResult(
            response_text="ok",
            lambda_program=TypedLambdaProgram("Context -> Response", "lambda _: ok", "respond"),
            actions=[
                PrimitiveAction(
                    "respond_text",
                    "ResponsePlan -> SafeTextMessage",
                    {"text": "ok"},
                    [],
                )
            ],
            reasoning_summary=["baseline"],
            telemetry={},
        )


class AgentBaselineTests(unittest.TestCase):
    def test_process_turn_persists_jsonl_memory_and_reports_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "baseline" / "memory.jsonl"
            agent = AgentBaseline(
                left_hemisphere=_FakeLeftHemisphere(),
                embedder=_FakeEmbedder(),
                action_runtime=ConcreteActionRuntime(),
                config=AgentBaselineConfig(memory_path=memory_path),
            )

            result = agent.process_turn(UserTurn(session_id="s1", user_text="ola"))
            self.assertEqual(result["response_text"], "ok")
            self.assertIn("tool_success_rate", result)
            self.assertTrue(memory_path.exists())

            lines = memory_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            saved = json.loads(lines[0])
            self.assertEqual(saved["response_text"], "ok")


if __name__ == "__main__":
    unittest.main()
