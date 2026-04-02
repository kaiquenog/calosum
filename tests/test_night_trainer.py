from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

from calosum.adapters.night_trainer.night_trainer import NightTrainer


class NightTrainerTests(unittest.TestCase):
    def test_training_cycle_compiles_opro_lite_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            dataset_path = base / "nightly.jsonl"
            output_dir = base / "artifacts"
            dataset = [
                {
                    "category": "good",
                    "input_text": "Preciso reorganizar o projeto em passos.",
                    "response_text": "Vamos dividir em etapas seguras.",
                    "runtime_retry_count": 0,
                    "critique_revision_count": 0,
                    "actions": ["respond_text", "propose_plan"],
                },
                {
                    "category": "corrected",
                    "input_text": "Quero um plano curto.",
                    "response_text": "Segue um plano curto.",
                    "runtime_retry_count": 1,
                    "critique_revision_count": 1,
                    "actions": ["respond_text", "propose_plan"],
                },
                {
                    "category": "bad",
                    "input_text": "Chame uma ferramenta invalida.",
                    "response_text": "Tentando algo inseguro.",
                    "runtime_retry_count": 1,
                    "critique_revision_count": 0,
                    "actions": ["unknown_tool"],
                },
            ]
            dataset_path.write_text(
                "\n".join(json.dumps(item, ensure_ascii=False) for item in dataset) + "\n",
                encoding="utf-8",
            )

            trainer = NightTrainer("test-model", dataset_path, output_dir)
            result = trainer.run_training_cycle()

            self.assertEqual(result["status"], "success")
            artifact = json.loads((output_dir / "compiled_prompt.json").read_text(encoding="utf-8"))
            self.assertEqual(artifact["optimizer"], "OPROLiteHeuristic_v1")
            self.assertIn("valid JSON", artifact["selected_prompt"])
            self.assertEqual(artifact["selected_strategy"], result["selected_strategy"])
            self.assertGreaterEqual(len(artifact["few_shot_examples"]), 2)
            self.assertFalse(dataset_path.exists())

    def test_training_cycle_skips_when_no_good_examples_exist(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            dataset_path = base / "nightly.jsonl"
            output_dir = base / "artifacts"
            dataset_path.write_text(
                json.dumps(
                    {
                        "category": "bad",
                        "input_text": "Falhei",
                        "response_text": "Falhei",
                        "actions": ["unknown_tool"],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            trainer = NightTrainer("test-model", dataset_path, output_dir)
            result = trainer.run_training_cycle()

            self.assertEqual(result["status"], "skipped")
            self.assertEqual(result["reason"], "No valid examples")
            self.assertFalse(dataset_path.exists())

    def test_dspy_backend_compiles_artifact_when_module_is_available(self) -> None:
        class FakeLM:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeExample(dict):
            def with_inputs(self, *args):
                self["inputs"] = list(args)
                return self

        class FakeProgram:
            def __init__(self, signature):
                self.signature = signature

        class FakeCompiled:
            instructions = "Use typed JSON outputs and preserve safe actions."
            demos = [
                {
                    "user_message": "Organize a task list",
                    "response_text": "Aqui está um plano enxuto.",
                }
            ]

        class FakeGEPA:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def compile(self, student=None, trainset=None, **kwargs):
                return FakeCompiled()

        fake_dspy = types.ModuleType("dspy")
        fake_dspy.LM = FakeLM
        fake_dspy.Example = FakeExample
        fake_dspy.ChainOfThought = FakeProgram
        fake_dspy.GEPA = FakeGEPA
        fake_dspy.configure = lambda **kwargs: None
        sys.modules["dspy"] = fake_dspy

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                base = Path(temp_dir)
                dataset_path = base / "nightly.jsonl"
                output_dir = base / "artifacts"
                dataset_path.write_text(
                    json.dumps(
                        {
                            "category": "good",
                            "input_text": "Organize a task list",
                            "response_text": "Aqui está um plano enxuto.",
                            "actions": ["respond_text", "propose_plan"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )

                trainer = NightTrainer(
                    "test-model",
                    dataset_path,
                    output_dir,
                    api_url="https://llm.local/v1",
                    provider="openai_compatible",
                    backend="dspy",
                )
                result = trainer.run_training_cycle()
                artifact = json.loads((output_dir / "compiled_prompt.json").read_text(encoding="utf-8"))
        finally:
            sys.modules.pop("dspy", None)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["optimizer"], "DSPyGEPA")
        self.assertEqual(artifact["optimizer"], "DSPyGEPA")
        self.assertIn("typed JSON outputs", artifact["selected_prompt"])


if __name__ == "__main__":
    unittest.main()
