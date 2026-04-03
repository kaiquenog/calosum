from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from calosum.bootstrap.entry import api as api_module
from fastapi.testclient import TestClient


class ReadinessComponentHealthTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        env = {
            "CALOSUM_IGNORE_DOTENV": "true",
            "CALOSUM_MODE": "local",
            "CALOSUM_INFRA_PROFILE": "persistent",
            "CALOSUM_MEMORY_DIR": str(Path(self.temp_dir.name) / "memory"),
            "CALOSUM_OTLP_JSONL": str(Path(self.temp_dir.name) / "telemetry" / "events.jsonl"),
            "CALOSUM_LEFT_ENDPOINT": "http://127.0.0.1:9/v1/chat/completions",
            "CALOSUM_RIGHT_BACKEND": "vjepa21",
            "CALOSUM_RIGHT_BUDGET_MEMORY_MB": "256",
        }
        self.env_patcher = patch.dict(os.environ, env, clear=False)
        self.env_patcher.start()
        self.addCleanup(self.env_patcher.stop)
        api_module.get_settings.cache_clear()
        api_module.get_builder.cache_clear()
        api_module.get_agent.cache_clear()
        self.addCleanup(api_module.get_settings.cache_clear)
        self.addCleanup(api_module.get_builder.cache_clear)
        self.addCleanup(api_module.get_agent.cache_clear)

    def test_ready_exposes_budget_and_component_degradation(self) -> None:
        client = TestClient(api_module.app)
        response = client.get("/ready")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("operational_budgets", payload)
        self.assertIn("turn_contract", payload)
        self.assertEqual(payload["components"]["right_hemisphere"]["health"], "degraded")
        self.assertEqual(payload["operational_budgets"]["right_hemisphere"]["status"], "within_budget")


if __name__ == "__main__":
    unittest.main()
