from __future__ import annotations

import unittest
from fastapi.testclient import TestClient

from calosum.bootstrap.api import app


class ApiIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_health_check(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_system_info_returns_capabilities(self) -> None:
        response = self.client.get("/v1/system/info")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("capabilities", data["info"])

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

    def test_system_state_returns_workspace_not_found_initially(self) -> None:
        from calosum.bootstrap.api import get_agent
        get_agent().last_workspace_by_session.clear()

        response = self.client.get("/v1/system/state", params={"session_id": "fresh-session"})
        self.assertEqual(response.status_code, 404)
        data = response.json()
        self.assertEqual(data["status"], "error")

    def test_system_state_returns_workspace_after_turn(self) -> None:
        # Generate a turn
        post_response = self.client.post("/v1/chat/completions", json={"text": "Hello workspace"})
        self.assertEqual(post_response.status_code, 200)

        # Retrieve state
        response = self.client.get("/v1/system/state")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        workspace = data["state"]
        
        self.assertIn("task_frame", workspace)
        self.assertIn("right_notes", workspace)
        self.assertIn("left_notes", workspace)
        self.assertEqual(workspace["task_frame"]["user_text"], "Hello workspace")


if __name__ == "__main__":
    unittest.main()
