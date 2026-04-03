from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from calosum.adapters.execution.tool_runtime import ConcreteActionRuntime
from calosum.bootstrap.wiring.agent_baseline import AgentBaseline, AgentBaselineConfig
from calosum.bootstrap.infrastructure.settings import CalosumMode, InfrastructureSettings
from calosum.shared.models.types import (
    ActionPlannerResult,
    PrimitiveAction,
    TypedLambdaProgram,
    UserTurn,
)


class _FakeEmbedder:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 32 for _ in texts]


class _FakeLeftHemisphere:
    def reason(self, user_turn, bridge_packet, memory_context, runtime_feedback=None, attempt=0, workspace=None):
        return ActionPlannerResult(
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

    def test_from_settings_rejects_self_referential_default_in_api_mode(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "requires CALOSUM_LEFT_ENDPOINT in API mode"):
            AgentBaseline.from_settings(InfrastructureSettings(mode=CalosumMode.API))

    def test_from_settings_honors_require_left_endpoint_env_var(self) -> None:
        settings = InfrastructureSettings(
            mode=CalosumMode.LOCAL,
            left_hemisphere_provider="openai",
        )
        with patch.dict(os.environ, {"CALOSUM_REQUIRE_LEFT_ENDPOINT": "1"}):
            with self.assertRaisesRegex(RuntimeError, "CALOSUM_REQUIRE_LEFT_ENDPOINT=1"):
                AgentBaseline.from_settings(settings)


if __name__ == "__main__":
    unittest.main()
