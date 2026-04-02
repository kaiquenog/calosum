from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class CliIntegrationTests(unittest.TestCase):
    def test_run_scenario_outputs_json_with_dashboard_and_sleep_mode(self) -> None:
        scenario = {
            "session_id": "cli-session",
            "turns": [
                {
                    "text": "Estou frustrado e preciso de um plano urgente.",
                    "signals": [
                        {
                            "modality": "audio",
                            "source": "microphone",
                            "payload": {"transcript": "voz tensa"},
                            "metadata": {"emotion": "frustrado"},
                        }
                    ],
                },
                {
                    "text": "Prefiro respostas curtas com passos claros quando a situacao estiver urgente.",
                    "group_variants": [
                        {
                            "variant_id": "empathetic",
                            "tokenizer_overrides": {"salience_threshold": 0.45},
                        },
                        {
                            "variant_id": "strict",
                            "tokenizer_overrides": {"salience_threshold": 0.9},
                        },
                    ],
                },
            ],
            "sleep_mode": True,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            scenario_path = Path(temp_dir) / "scenario.json"
            memory_dir = Path(temp_dir) / "memory"
            otlp_path = Path(temp_dir) / "telemetry.jsonl"
            scenario_path.write_text(json.dumps(scenario), encoding="utf-8")

            env = os.environ.copy()
            env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
            env["CALOSUM_LEFT_ENDPOINT"] = "http://127.0.0.1:9999/v1/chat/completions"
            env["CALOSUM_PERCEPTION_MODEL"] = "jepa"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "calosum.bootstrap.entry.cli",
                    "run-scenario",
                    str(scenario_path),
                    "--memory-dir",
                    str(memory_dir),
                    "--otlp-jsonl",
                    str(otlp_path),
                ],
                cwd=PROJECT_ROOT,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertTrue((memory_dir / "episodic.jsonl").exists())
            self.assertTrue(otlp_path.exists())

        payload = json.loads(completed.stdout)
        self.assertEqual(payload["session_id"], "cli-session")
        self.assertEqual(len(payload["results"]), 2)
        self.assertIn("sleep_mode", payload)
        self.assertIn("dashboard", payload)
        self.assertIn("reflection", payload["dashboard"])


if __name__ == "__main__":
    unittest.main()
