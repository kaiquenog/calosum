from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
import json
from pathlib import Path

class TestCiScriptsExist(unittest.TestCase):
    def test_ci_scripts_exist(self):
        self.assertTrue(Path("scripts/ci_integration_benchmark.py").exists())
        self.assertTrue(Path("scripts/ci_benchmark_gate.py").exists())
        self.assertTrue(Path("scripts/coverage_gate_new_modules.py").exists())

    def test_ci_integration_benchmark_generates_real_metrics_payload(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "ci_integration_benchmark.py"
        with tempfile.TemporaryDirectory() as temp_dir:
            output_json = Path(temp_dir) / "integration.json"
            output_md = Path(temp_dir) / "integration.md"
            env = {**os.environ, "PYTHONPATH": str(repo_root / "src"), "CALOSUM_CI_BENCHMARK_TURNS": "3"}
            subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--turns",
                    "2",
                    "--latency-p95-threshold-ms",
                    "5000",
                    "--output-json",
                    str(output_json),
                    "--output-md",
                    str(output_md),
                ],
                check=True,
                cwd=temp_dir,
                env=env,
            )

            payload = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["metrics"]["turns_executed"], 3)
            self.assertIn("tool_success_rate", payload["metrics"])
            self.assertTrue(output_md.exists())

    def test_ci_benchmark_gate_supports_nested_metrics_payload(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "ci_benchmark_gate.py"
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline = Path(temp_dir) / "baseline.json"
            candidate = Path(temp_dir) / "candidate.json"
            baseline.write_text(json.dumps({"metrics": {"tool_success_rate": 1.0}}), encoding="utf-8")
            candidate.write_text(json.dumps({"metrics": {"tool_success_rate": 0.96}}), encoding="utf-8")
            subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--baseline",
                    str(baseline),
                    "--candidate",
                    str(candidate),
                    "--metric",
                    "tool_success_rate",
                    "--max-regression-percent",
                    "5.0",
                ],
                check=True,
                cwd=temp_dir,
            )

if __name__ == "__main__":
    unittest.main()
