from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from calosum.bootstrap import api as api_module


class ApiIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

        base = Path(self.temp_dir.name)
        env = {
            "CALOSUM_INFRA_PROFILE": "persistent",
            "CALOSUM_MEMORY_DIR": str(base / "memory"),
            "CALOSUM_OTLP_JSONL": str(base / "telemetry" / "events.jsonl"),
            "CALOSUM_BRIDGE_STATE_DIR": str(base / "state"),
            "CALOSUM_EVOLUTION_ARCHIVE_PATH": str(base / "evolution" / "archive.jsonl"),
            "CALOSUM_PERCEPTION_MODEL": "jepa",
            "CALOSUM_LEFT_ENDPOINT": "http://127.0.0.1:9/v1/chat/completions",
            "CALOSUM_LEFT_MODEL": "test-left-model",
            "CALOSUM_REASON_MODEL": "test-reason-model",
            "CALOSUM_LEFT_PROVIDER": "openai_compatible_chat",
        }
        self.env_patcher = patch.dict(os.environ, env, clear=False)
        self.env_patcher.start()
        self.addCleanup(self.env_patcher.stop)

        self._clear_caches()
        self.addCleanup(self._clear_caches)
        self.client = TestClient(api_module.app)

    def _clear_caches(self) -> None:
        api_module.get_settings.cache_clear()
        api_module.get_builder.cache_clear()
        api_module.get_agent.cache_clear()

    def test_health_check(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_system_info_returns_runtime_capabilities_and_routing_resolution(self) -> None:
        response = self.client.get("/v1/system/info")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("capabilities", data["info"])
        self.assertGreater(len(data["info"]["capabilities"]["tools"]), 0)
        self.assertIn("routing_resolution", data["info"])
        self.assertEqual(data["info"]["routing_resolution"]["perception"]["active_model"], "jepa")

    def test_system_architecture_returns_self_model(self) -> None:
        response = self.client.get("/v1/system/architecture")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("components", data["architecture"])
        self.assertIn("connections", data["architecture"])
        self.assertIn("adaptation_surface", data["architecture"])
        self.assertTrue(len(data["architecture"]["components"]) > 0)

    def test_system_capabilities_returns_capabilities_from_self_model(self) -> None:
        response = self.client.get("/v1/system/capabilities")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("right_hemisphere", data["capabilities"])
        self.assertIn("left_hemisphere", data["capabilities"])
        self.assertIn("tools", data["capabilities"])
        self.assertEqual(data["capabilities"]["routing_policy"]["reason_model"], "test-reason-model")

    def test_system_state_returns_workspace_not_found_initially(self) -> None:
        api_module.get_agent().last_workspace_by_session.clear()

        response = self.client.get("/v1/system/state", params={"session_id": "fresh-session"})
        self.assertEqual(response.status_code, 404)
        data = response.json()
        self.assertEqual(data["status"], "error")

    def test_system_state_returns_workspace_after_turn(self) -> None:
        post_response = self.client.post("/v1/chat/completions", json={"text": "Hello workspace"})
        self.assertEqual(post_response.status_code, 200)

        response = self.client.get("/v1/system/state")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        workspace = data["state"]

        self.assertIn("task_frame", workspace)
        self.assertIn("right_notes", workspace)
        self.assertIn("left_notes", workspace)
        self.assertIn("self_model_ref", workspace)
        self.assertIn("capability_snapshot", workspace)
        self.assertIn("pending_questions", workspace)
        self.assertEqual(workspace["task_frame"]["user_text"], "Hello workspace")

    def test_system_state_does_not_leak_other_session_workspace(self) -> None:
        response = self.client.post(
            "/v1/chat/completions",
            json={"text": "Hello from session A", "session_id": "session-a"},
        )
        self.assertEqual(response.status_code, 200)

        missing = self.client.get("/v1/system/state", params={"session_id": "session-b"})
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(missing.json()["status"], "error")

    def test_system_awareness_generates_extended_diagnostic(self) -> None:
        self.client.post("/v1/chat/completions", json={"text": "Quero uma resposta com ação externa e aprovação."})

        response = self.client.get("/v1/system/awareness", params={"session_id": "api-session"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("diagnostic", data)
        self.assertIn("bottlenecks", data["diagnostic"])
        self.assertIn("failure_types", data["diagnostic"])
        self.assertIn("pending_approval_backlog", data["diagnostic"])
        self.assertIn("pending_directive_count", data["diagnostic"])
        self.assertIn("surprise_trend", data["diagnostic"])

    def test_system_directives_returns_pending_directives(self) -> None:
        response = self.client.get("/v1/system/directives")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("directives", data)
        self.assertIsInstance(data["directives"], list)

    def test_system_introspect_returns_grounded_self_awareness(self) -> None:
        self.client.post(
            "/v1/chat/completions",
            json={"text": "Preciso de uma resposta curta", "session_id": "focused-session"},
        )

        response = self.client.post(
            "/v1/system/introspect",
            json={
                "query": "qual backend de raciocinio memoria e telemetria estao ativos nesta sessao?",
                "session_id": "focused-session",
            },
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("response", data)
        self.assertIn("Sessão focused-session", data["response"])
        self.assertIn("reason=", data["response"])
        self.assertIn("memory=", data["response"])
        self.assertIn("telemetry=", data["response"])


if __name__ == "__main__":
    unittest.main()
