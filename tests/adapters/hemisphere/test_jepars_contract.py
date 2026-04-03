from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from calosum.adapters.hemisphere.input_perception_jepars import JepaRsConfig, JepaRsRightHemisphereAdapter
from calosum.shared.models.types import UserTurn


class JepaRsContractTests(unittest.TestCase):
    def test_adapter_sends_and_exposes_contract_version(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            script = Path(temp_dir) / "fake_jepars_contract.py"
            script.write_text(
                """
#!/usr/bin/env python3
import json, sys
payload = json.loads(sys.stdin.read() or "{}")
assert payload["contract_version"] == "jepa-rs-arrow-v1"
print(json.dumps({
    "latent_vector": [0.1, 0.2, 0.3, 0.4],
    "salience": 0.5,
    "confidence": 0.8,
    "surprise_score": 0.2,
    "emotional_labels": ["neutral"]
}))
""".strip(),
                encoding="utf-8",
            )
            os.chmod(script, 0o755)
            adapter = JepaRsRightHemisphereAdapter(JepaRsConfig(binary_path=str(script), timeout_seconds=10.0))

            state = adapter.perceive(UserTurn(session_id="s", user_text="teste de contrato"))

            self.assertEqual(state.telemetry["right_backend"], "jepars_local")
            self.assertEqual(state.telemetry["contract_version"], "jepa-rs-arrow-v1")
            self.assertEqual(state.telemetry["checkpoint_loaded"], False)

    def test_adapter_rejects_non_numeric_latent_vector(self) -> None:
        adapter = JepaRsRightHemisphereAdapter(JepaRsConfig(binary_path="jepa-rs"))

        with self.assertRaises(RuntimeError):
            adapter._validate_schema({"latent_vector": ["bad"]})


if __name__ == "__main__":
    unittest.main()
