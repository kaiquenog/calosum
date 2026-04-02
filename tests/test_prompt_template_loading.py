from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from calosum.adapters.llm.llm_payloads import build_left_hemisphere_prompt, load_left_prompt_template
from calosum.shared.types import (
    BridgeControlSignal,
    CognitiveBridgePacket,
    MemoryContext,
    UserTurn,
)


class PromptTemplateLoadingTests(unittest.TestCase):
    def test_load_left_prompt_template_from_env_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            template_path = Path(temp_dir) / "custom.md"
            template_path.write_text("Input={input_text}\nActions={available_actions}", encoding="utf-8")
            old = os.environ.get("CALOSUM_LEFT_PROMPT_PATH")
            os.environ["CALOSUM_LEFT_PROMPT_PATH"] = str(template_path)
            try:
                loaded = load_left_prompt_template()
                self.assertIn("Input={input_text}", loaded)
            finally:
                if old is None:
                    os.environ.pop("CALOSUM_LEFT_PROMPT_PATH", None)
                else:
                    os.environ["CALOSUM_LEFT_PROMPT_PATH"] = old

    def test_build_prompt_uses_external_template(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            template_path = Path(temp_dir) / "custom.md"
            template_path.write_text("CTX={input_text}\n{available_actions}", encoding="utf-8")
            old = os.environ.get("CALOSUM_LEFT_PROMPT_PATH")
            os.environ["CALOSUM_LEFT_PROMPT_PATH"] = str(template_path)
            try:
                prompt = build_left_hemisphere_prompt(
                    UserTurn(session_id="s", user_text="teste"),
                    CognitiveBridgePacket(
                        context_id="ctx",
                        soft_prompts=[],
                        control=BridgeControlSignal(
                            target_temperature=0.2,
                            empathy_priority=False,
                            system_directives=[],
                        ),
                        salience=0.1,
                    ),
                    MemoryContext(recent_episodes=[], semantic_rules=[], knowledge_triples=[]),
                    feedback=[],
                )
            finally:
                if old is None:
                    os.environ.pop("CALOSUM_LEFT_PROMPT_PATH", None)
                else:
                    os.environ["CALOSUM_LEFT_PROMPT_PATH"] = old

        self.assertIn("CTX=teste", prompt)
        self.assertIn("spawn_subordinate", prompt)


if __name__ == "__main__":
    unittest.main()
