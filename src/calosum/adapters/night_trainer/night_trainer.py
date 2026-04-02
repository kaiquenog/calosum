from __future__ import annotations

import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class LocalDatasetExporter:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def export(self, dataset: list[dict[str, Any]], filename: str) -> str:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        export_path = self.output_dir / filename
        with export_path.open("w", encoding="utf-8") as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        return str(export_path)


class NightTrainer:
    """
    Compilador offline do ciclo noturno.

    Backends suportados:
    - `auto`: tenta DSPy e cai para OPRO-lite quando a dependência não existe;
    - `dspy`: força o caminho DSPy, mas ainda preserva fallback local;
    - `opro_lite`: usa apenas a compilação heurística local.
    """

    def __init__(
        self,
        model_name: str,
        dataset_path: Path,
        output_dir: Path,
        lora_dataset_path: Path | None = None,
        lora_output_dir: Path | None = None,
        *,
        api_url: str | None = None,
        api_key: str | None = None,
        provider: str | None = None,
        reasoning_effort: str | None = None,
        backend: str = "auto",
    ):
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.lora_dataset_path = lora_dataset_path or dataset_path.parent / "lora_sharegpt.jsonl"
        self.lora_output_dir = lora_output_dir or output_dir.parent / "lora_adapters" / "latest"
        self.api_url = api_url
        self.api_key = api_key
        self.provider = provider
        self.reasoning_effort = reasoning_effort
        self.backend = backend.lower()

    def run_training_cycle(self) -> dict[str, Any]:
        if self.backend in {"lora", "qlora"}:
            return self._run_lora_cycle(self.backend)

        if not self.dataset_path.exists():
            return {"status": "skipped", "reason": "No dataset found"}

        try:
            if self.backend in {"auto", "dspy"}:
                dspy_result = self._run_dspy_cycle()
                if dspy_result is not None:
                    status = dspy_result.get("status")
                    if status == "success":
                        return dspy_result
                    if self.backend == "dspy":
                        return dspy_result
                    if status == "skipped" and not self.dataset_path.exists():
                        return dspy_result

            if not self.dataset_path.exists():
                return {"status": "skipped", "reason": "No dataset found"}

            dataset = self._load_dataset()
            good_examples = [item for item in dataset if item.get("category") in ("good", "corrected")]

            if not good_examples:
                logger.info("No high-quality examples found for prompt compilation.")
                self._cleanup_dataset()
                return {"status": "skipped", "reason": "No valid examples"}

            ranked_examples = self._rank_examples(good_examples)
            compiled_artifact = self._build_compiled_artifact(dataset, ranked_examples)

            self.output_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = self.output_dir / "compiled_prompt.json"
            with artifact_path.open("w", encoding="utf-8") as f:
                json.dump(compiled_artifact, f, indent=2, ensure_ascii=False)

            logger.info("Night training artifact saved to %s", artifact_path)
            self._cleanup_dataset()
            return {
                "status": "success",
                "examples_learned": len(ranked_examples),
                "artifact_path": str(artifact_path),
                "selected_strategy": compiled_artifact["selected_strategy"],
            }
        except Exception as e:
            logger.error("Night training failed: %s", e)
            return {"status": "error", "reason": str(e)}

    def _run_lora_cycle(self, backend: str) -> dict[str, Any]:
        from calosum.adapters.night_trainer.night_trainer_lora import LoraNightTrainer

        trainer = LoraNightTrainer(
            base_model_name=self.model_name,
            dataset_path=self.lora_dataset_path,
            output_dir=self.lora_output_dir,
        )
        result = trainer.run_training()
        if result.get("status") == "success":
            result["backend"] = backend
        return result

    def _run_dspy_cycle(self) -> dict[str, Any] | None:
        from calosum.adapters.night_trainer.night_trainer_dspy import DSPyNightTrainer

        try:
            trainer = DSPyNightTrainer(
                model_name=self.model_name,
                dataset_path=self.dataset_path,
                output_dir=self.output_dir,
                api_url=self.api_url,
                api_key=self.api_key,
                provider=self.provider,
                reasoning_effort=self.reasoning_effort,
            )
            result = trainer.run_training_cycle()
        except Exception as exc:
            logger.warning("DSPy night training failed, falling back to OPRO-lite: %s", exc)
            return None

        if result.get("status") == "success":
            logger.info("DSPy night trainer compiled a prompt artifact.")
            return result
        if result.get("status") == "skipped":
            return result
        if self.backend == "dspy":
            logger.warning("DSPy requested but unavailable; falling back to OPRO-lite: %s", result)
        return None

    def _cleanup_dataset(self) -> None:
        if self.dataset_path.exists():
            os.remove(self.dataset_path)

    def _load_dataset(self) -> list[dict[str, Any]]:
        logger.info("Loading nightly dataset from %s", self.dataset_path)
        dataset: list[dict[str, Any]] = []
        with self.dataset_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    dataset.append(json.loads(line))
        return dataset

    def _rank_examples(self, examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ranked = []
        for item in examples:
            enriched = dict(item)
            enriched["training_score"] = self._example_score(item)
            ranked.append(enriched)
        ranked.sort(key=lambda item: item["training_score"], reverse=True)
        return ranked

    def _example_score(self, item: dict[str, Any]) -> float:
        category = item.get("category", "good")
        category_bonus = 1.0 if category == "good" else 0.82
        retry_penalty = min(0.35, 0.08 * int(item.get("runtime_retry_count", 0)))
        critique_penalty = min(0.35, 0.08 * int(item.get("critique_revision_count", 0)))
        actions = item.get("actions", [])
        action_bonus = min(0.15, 0.05 * len(actions)) if isinstance(actions, list) else 0.0
        response_text = str(item.get("response_text", ""))
        response_bonus = min(0.15, len(response_text.split()) / 60.0)
        return round(category_bonus + action_bonus + response_bonus - retry_penalty - critique_penalty, 3)

    def _build_compiled_artifact(
        self,
        dataset: list[dict[str, Any]],
        ranked_examples: list[dict[str, Any]],
    ) -> dict[str, Any]:
        top_examples = ranked_examples[:5]
        notes = self._derive_optimization_notes(dataset)
        candidates = self._build_prompt_candidates(notes, dataset)
        selected = max(candidates, key=lambda item: item["score"])

        return {
            "model_name": self.model_name,
            "compiled_at": "night_cycle",
            "optimizer": "OPROLiteHeuristic_v1",
            "selected_strategy": selected["strategy"],
            "selected_prompt": selected["prompt_text"],
            "optimization_notes": notes,
            "prompt_candidates": candidates,
            "few_shot_examples": [
                {
                    "input_text": item.get("input_text", ""),
                    "response_text": item.get("response_text", ""),
                    "category": item.get("category", "good"),
                    "training_score": item.get("training_score", 0.0),
                }
                for item in top_examples
            ],
            "selection_metrics": {
                "dataset_size": len(dataset),
                "good_examples": sum(1 for item in dataset if item.get("category") == "good"),
                "corrected_examples": sum(1 for item in dataset if item.get("category") == "corrected"),
                "bad_examples": sum(1 for item in dataset if item.get("category") == "bad"),
                "candidate_count": len(candidates),
                "selected_score": selected["score"],
            },
        }

    def _derive_optimization_notes(self, dataset: list[dict[str, Any]]) -> list[str]:
        notes = [
            "Always return valid JSON that matches the LeftHemisphereResult contract.",
            "Use specific typed signatures and never fall back to placeholders like Any -> Any.",
        ]
        action_counts = Counter(
            action
            for item in dataset
            for action in item.get("actions", [])
            if isinstance(action, str)
        )
        corrected_count = sum(1 for item in dataset if item.get("category") == "corrected")
        bad_count = sum(1 for item in dataset if item.get("category") == "bad")

        if action_counts.get("propose_plan", 0):
            notes.append(
                "When the user asks for organization or steps, prefer propose_plan with explicit ordered steps."
            )
        if action_counts.get("respond_text", 0):
            notes.append(
                "Keep response_text aligned with respond_text payload so the runtime and the user-facing answer stay synchronized."
            )
        if bad_count or corrected_count:
            notes.append(
                "Minimize the action frontier and keep payloads schema-valid to reduce runtime retries and critique repairs."
            )
        if action_counts.get("load_semantic_rules", 0):
            notes.append("Reuse semantic memory only when it adds factual grounding to the answer.")
        return notes

    def _build_prompt_candidates(
        self,
        notes: list[str],
        dataset: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        corrected_count = sum(1 for item in dataset if item.get("category") == "corrected")
        bad_count = sum(1 for item in dataset if item.get("category") == "bad")
        action_counts = Counter(
            action
            for item in dataset
            for action in item.get("actions", [])
            if isinstance(action, str)
        )

        prompts = [
            {
                "strategy": "baseline_structured",
                "prompt_text": " ".join(notes[:3]),
            },
            {
                "strategy": "runtime_minimal",
                "prompt_text": " ".join(
                    notes
                    + [
                        "Prefer the smallest valid action set that still satisfies the user request."
                    ]
                ),
            },
            {
                "strategy": "memory_grounded",
                "prompt_text": " ".join(
                    notes
                    + [
                        "Ground actions in semantic rules and knowledge triples whenever they materially reduce ambiguity."
                    ]
                ),
            },
        ]

        for item in prompts:
            score = 1.0
            text = item["prompt_text"]
            if "valid JSON" in text:
                score += 0.4
            if "typed signatures" in text:
                score += 0.25
            if action_counts.get("propose_plan", 0) and "propose_plan" in text:
                score += 0.2
            if corrected_count or bad_count:
                if "smallest valid action set" in text:
                    score += 0.3
                if "schema-valid" in text:
                    score += 0.15
            if action_counts.get("load_semantic_rules", 0) and "knowledge triples" in text:
                score += 0.15
            item["score"] = round(score, 3)

        return prompts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    trainer = NightTrainer(
        model_name=os.getenv("CALOSUM_LEFT_MODEL", "Qwen/Qwen-3.5-9B-Instruct"),
        dataset_path=Path(".calosum-runtime/nightly_data/dspy_dataset.jsonl"),
        output_dir=Path(".calosum-runtime/dspy_artifacts/latest"),
        api_url=os.getenv("CALOSUM_LEFT_ENDPOINT"),
        api_key=os.getenv("CALOSUM_LEFT_API_KEY"),
        provider=os.getenv("CALOSUM_LEFT_PROVIDER"),
        reasoning_effort=os.getenv("CALOSUM_LEFT_REASONING_EFFORT"),
        backend=os.getenv("CALOSUM_NIGHT_TRAINER_BACKEND", "auto"),
    )
    result = trainer.run_training_cycle()
    print(result)
