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

    def test_training_cycle_runs_bridge_and_right_hemisphere_adaptation_when_components_are_attached(self) -> None:
        class FakeFusion:
            def __init__(self) -> None:
                self.calls = 0

            def train_step(self, latent_vector, target_salience, learning_rate=0.001):
                self.calls += 1
                return 0.123

            def export_trainable_state(self):
                return {"fake": True}

        class FakeRightProvider:
            def __init__(self) -> None:
                self.records = 0

            def train_predictor_from_records(self, records, learning_rate=0.0015, epochs=2):
                self.records += len(records)
                return {"status": "success", "records_used": len(records), "avg_loss": 0.01}

        class FakeTokenizer:
            def __init__(self) -> None:
                self.fusion = FakeFusion()

        class WrappedRight:
            def __init__(self) -> None:
                self.base_adapter = type("Inner", (), {"provider": FakeRightProvider()})()

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            dataset_path = base / "nightly.jsonl"
            output_dir = base / "artifacts"
            dataset_path.write_text(
                json.dumps(
                    {
                        "category": "good",
                        "input_text": "Organize em passos",
                        "response_text": "Passos claros",
                        "actions": ["respond_text", "propose_plan"],
                        "latent_vector": [0.1, 0.2, 0.3],
                        "target_salience": 0.7,
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            right_dataset_path = base / "right_hemisphere_dataset.jsonl"
            right_dataset_path.write_text(
                json.dumps(
                    {
                        "latent_t": [0.1, 0.2, 0.3],
                        "latent_t1": [0.2, 0.3, 0.4],
                        "prediction_error": 0.9,
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            trainer = NightTrainer("test-model", dataset_path, output_dir, backend="opro_lite")
            trainer.attach_components(tokenizer=FakeTokenizer(), right_hemisphere=WrappedRight())
            result = trainer.run_training_cycle()

            self.assertEqual(result["status"], "success")
            self.assertEqual(result["bridge_plasticity"]["status"], "success")
            self.assertEqual(result["bridge_plasticity"]["updates"], 1)
            self.assertEqual(result["right_hemisphere_training"]["status"], "success")
            self.assertGreaterEqual(result["right_hemisphere_training"]["records_used"], 1)


if __name__ == "__main__":
    unittest.main()
