from __future__ import annotations

import importlib
import inspect
import json
import os
from pathlib import Path
from typing import Any


class DSPyNightTrainer:
    def __init__(
        self,
        model_name: str,
        dataset_path: Path,
        output_dir: Path,
        *,
        api_url: str | None = None,
        api_key: str | None = None,
        provider: str | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.api_url = api_url
        self.api_key = api_key
        self.provider = provider
        self.reasoning_effort = reasoning_effort

    def run_training_cycle(self) -> dict[str, Any]:
        try:
            dspy = importlib.import_module("dspy")
        except Exception:
            return {"status": "skipped", "reason": "DSPy unavailable"}

        if not self.dataset_path.exists():
            return {"status": "skipped", "reason": "No dataset found"}

        dataset = self._load_dataset()
        good_examples = [item for item in dataset if item.get("category") in {"good", "corrected"}]
        if not good_examples:
            os.remove(self.dataset_path)
            return {"status": "skipped", "reason": "No valid examples"}

        lm = self._configure_lm(dspy)
        trainset = [self._build_example(dspy, item) for item in good_examples[:8]]
        program = self._build_program(dspy)
        optimizer, optimizer_name = self._build_optimizer(dspy)
        compiled = self._compile_program(optimizer, program, trainset)

        artifact = {
            "model_name": self.model_name,
            "compiled_at": "night_cycle",
            "optimizer": optimizer_name,
            "selected_strategy": "dspy_compiled_prompt",
            "selected_prompt": self._extract_prompt(compiled),
            "optimization_notes": [
                "DSPy optimizer compiled the prompt artifact from successful episodic traces.",
                f"LM endpoint respected via provider={self.provider or 'auto'} and api_url={self.api_url or 'default'}.",
            ],
            "few_shot_examples": self._extract_few_shots(compiled, good_examples),
            "selection_metrics": {
                "dataset_size": len(dataset),
                "trainset_size": len(trainset),
                "lm_configured": bool(lm),
            },
        }

        self.output_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = self.output_dir / "compiled_prompt.json"
        artifact_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
        os.remove(self.dataset_path)
        return {
            "status": "success",
            "artifact_path": str(artifact_path),
            "optimizer": optimizer_name,
            "selected_strategy": artifact["selected_strategy"],
        }

    def _load_dataset(self) -> list[dict[str, Any]]:
        dataset: list[dict[str, Any]] = []
        with self.dataset_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    dataset.append(json.loads(line))
        return dataset

    def _configure_lm(self, dspy: Any) -> Any:
        lm_class = getattr(dspy, "LM", None)
        if lm_class is None:
            return None

        model_name = self.model_name
        if self.provider and "/" not in model_name and self.provider not in {"auto", "openai_compatible"}:
            model_name = f"{self.provider}/{model_name}"

        kwargs = {
            "model": model_name,
            "api_base": self.api_url,
            "base_url": self.api_url,
            "api_key": self.api_key,
            "reasoning_effort": self.reasoning_effort,
        }
        lm = lm_class(**_filter_kwargs(lm_class, kwargs))
        configure = getattr(dspy, "configure", None)
        if callable(configure):
            configure(lm=lm)
        return lm

    def _build_example(self, dspy: Any, item: dict[str, Any]) -> Any:
        example_class = getattr(dspy, "Example", None)
        payload = {
            "user_message": item.get("input_text", ""),
            "context": f"category={item.get('category', 'good')}; actions={','.join(item.get('actions', []))}",
            "response_text": item.get("response_text", ""),
            "actions": item.get("actions", []),
        }
        if example_class is None:
            return payload

        example = example_class(**payload)
        with_inputs = getattr(example, "with_inputs", None)
        if callable(with_inputs):
            return with_inputs("user_message", "context")
        return example

    def _build_program(self, dspy: Any) -> Any:
        signature = "user_message, context -> response_text, actions"
        for candidate_name in ("ChainOfThought", "Predict"):
            candidate = getattr(dspy, candidate_name, None)
            if candidate is not None:
                return candidate(signature)
        raise RuntimeError("DSPy does not expose ChainOfThought or Predict")

    def _build_optimizer(self, dspy: Any) -> tuple[Any, str]:
        optimizer_class = getattr(dspy, "GEPA", None)
        optimizer_name = "DSPyGEPA"
        if optimizer_class is None:
            optimizer_class = getattr(dspy, "MIPROv2", None)
            optimizer_name = "DSPyMIPROv2"
        if optimizer_class is None:
            raise RuntimeError("DSPy optimizer GEPA/MIPROv2 unavailable")

        kwargs = {"metric": _training_metric, "max_iterations": 6, "max_iters": 6}
        return optimizer_class(**_filter_kwargs(optimizer_class, kwargs)), optimizer_name

    def _compile_program(self, optimizer: Any, program: Any, trainset: list[Any]) -> Any:
        compile_method = getattr(optimizer, "compile", None)
        if not callable(compile_method):
            raise RuntimeError("DSPy optimizer has no compile method")

        attempts = [
            {"student": program, "trainset": trainset},
            {"program": program, "trainset": trainset},
            {"student": program, "trainset": trainset, "valset": trainset[:3]},
            {"program": program, "trainset": trainset, "valset": trainset[:3]},
        ]
        last_error: Exception | None = None
        for attempt in attempts:
            try:
                return compile_method(**_filter_kwargs(compile_method, attempt))
            except TypeError as exc:
                last_error = exc
                continue
        raise RuntimeError(f"DSPy compile signature unsupported: {last_error}")

    def _extract_prompt(self, compiled: Any) -> str:
        candidates = [
            getattr(compiled, "instructions", None),
            getattr(compiled, "system_prompt", None),
            getattr(compiled, "prompt", None),
        ]
        signature = getattr(compiled, "signature", None)
        if signature is not None:
            candidates.append(getattr(signature, "instructions", None))

        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        return (
            "Return valid JSON for LeftHemisphereResult. Keep typed actions explicit. "
            "Minimize repair loops and preserve semantic grounding."
        )

    def _extract_few_shots(
        self,
        compiled: Any,
        examples: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        demos = getattr(compiled, "demos", None)
        rendered: list[dict[str, Any]] = []
        if isinstance(demos, list):
            for item in demos[:3]:
                rendered.append(
                    {
                        "input_text": _coerce_mapping_value(item, "user_message"),
                        "response_text": _coerce_mapping_value(item, "response_text"),
                        "category": "good",
                    }
                )
        if rendered:
            return rendered

        return [
            {
                "input_text": item.get("input_text", ""),
                "response_text": item.get("response_text", ""),
                "category": item.get("category", "good"),
            }
            for item in examples[:3]
        ]


def _training_metric(example: Any, prediction: Any, trace: Any = None) -> float:
    expected = _coerce_mapping_value(example, "response_text")
    actual = _coerce_mapping_value(prediction, "response_text") or _coerce_mapping_value(prediction, "answer")
    actions = _coerce_mapping_value(prediction, "actions")
    score = 0.0
    if actual:
        score += 0.45
    if expected and actual:
        expected_terms = set(str(expected).lower().split())
        actual_terms = set(str(actual).lower().split())
        if expected_terms:
            score += 0.35 * (len(expected_terms.intersection(actual_terms)) / len(expected_terms))
    if isinstance(actions, list) and actions:
        score += 0.2
    elif isinstance(actions, str) and actions.strip():
        score += 0.1
    return round(min(1.0, score), 3)


def _coerce_mapping_value(item: Any, key: str) -> Any:
    if isinstance(item, dict):
        return item.get(key)
    if hasattr(item, key):
        return getattr(item, key)
    if hasattr(item, "get"):
        try:
            return item.get(key)
        except Exception:
            return None
    return None


def _filter_kwargs(target: Any, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):
        return {key: value for key, value in payload.items() if value is not None}

    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_var_kwargs:
        return {key: value for key, value in payload.items() if value is not None}

    return {
        key: value
        for key, value in payload.items()
        if key in signature.parameters and value is not None
    }
