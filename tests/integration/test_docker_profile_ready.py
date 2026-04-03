from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class DockerProfileReadyTests(unittest.TestCase):
    def test_docker_profile_smoke_generates_ready_artifact(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        script = repo_root / "scripts" / "docker_profile_ready.py"
        with tempfile.TemporaryDirectory() as temp_dir:
            output_json = Path(temp_dir) / "docker_ready.json"
            output_md = Path(temp_dir) / "docker_ready.md"
            env = {"PYTHONPATH": str(repo_root / "src")}
            subprocess.run(
                [
                    sys.executable,
                    str(script),
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
            self.assertEqual(payload["payload"]["status"], "ready")
            self.assertIn("operational_budgets", payload["payload"])
            self.assertTrue(output_md.exists())


if __name__ == "__main__":
    unittest.main()
