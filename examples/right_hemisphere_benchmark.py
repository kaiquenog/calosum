from __future__ import annotations

import argparse
import json
import statistics
import time
import tracemalloc
import types
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

from calosum.adapters.right_hemisphere_hf import HuggingFaceRightHemisphereAdapter, HuggingFaceRightHemisphereConfig
from calosum.domain.cognition.right_hemisphere import RightHemisphereJEPA
from calosum.shared.types import UserTurn


@dataclass(slots=True)
class Sample:
    text: str
    target: str  # "high" or "neutral"


@dataclass(slots=True)
class AdapterRunResult:
    adapter: str
    mode: str
    target_threshold: float
    samples: int
    accuracy: float
    false_positive_rate: float
    avg_salience: float
    avg_confidence: float
    avg_latency_ms: float
    p95_latency_ms: float
    peak_memory_kb: float
    notes: str = ""


DATASET: list[Sample] = [
    Sample("Estou em emergência, preciso de ajuda urgente agora.", "high"),
    Sample("Sinto desespero e muita dor, preciso de suporte.", "high"),
    Sample("Estou ansioso e frustrado, sem saber por onde começar.", "high"),
    Sample("Tenho medo de errar e estou preocupado com o prazo.", "high"),
    Sample("Hoje foi um dia normal, sem novidades importantes.", "neutral"),
    Sample("Quero revisar um documento técnico em passos simples.", "neutral"),
    Sample("Obrigado pela ajuda anterior, vamos seguir com calma.", "neutral"),
    Sample("Preciso de um resumo curto da reunião de ontem.", "neutral"),
]


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = max(0, min(len(sorted_values) - 1, int(round((percentile / 100.0) * (len(sorted_values) - 1)))))
    return sorted_values[index]


def _summarize_run(
    *,
    adapter_name: str,
    mode: str,
    target_threshold: float,
    per_sample: list[dict[str, Any]],
    peak_memory_kb: float,
    notes: str = "",
) -> AdapterRunResult:
    if not per_sample:
        return AdapterRunResult(
            adapter=adapter_name,
            mode=mode,
            target_threshold=target_threshold,
            samples=0,
            accuracy=0.0,
            false_positive_rate=0.0,
            avg_salience=0.0,
            avg_confidence=0.0,
            avg_latency_ms=0.0,
            p95_latency_ms=0.0,
            peak_memory_kb=peak_memory_kb,
            notes=notes or "no samples executed",
        )

    correct = 0
    neutral_total = 0
    neutral_false_positives = 0
    saliences: list[float] = []
    confidences: list[float] = []
    latencies: list[float] = []

    for item in per_sample:
        salience = float(item["salience"])
        prediction = "high" if salience >= target_threshold else "neutral"
        if prediction == item["target"]:
            correct += 1
        if item["target"] == "neutral":
            neutral_total += 1
            if prediction == "high":
                neutral_false_positives += 1
        saliences.append(salience)
        confidences.append(float(item["confidence"]))
        latencies.append(float(item["latency_ms"]))

    return AdapterRunResult(
        adapter=adapter_name,
        mode=mode,
        target_threshold=target_threshold,
        samples=len(per_sample),
        accuracy=round(correct / len(per_sample), 3),
        false_positive_rate=round(neutral_false_positives / max(1, neutral_total), 3),
        avg_salience=round(statistics.mean(saliences), 3),
        avg_confidence=round(statistics.mean(confidences), 3),
        avg_latency_ms=round(statistics.mean(latencies), 3),
        p95_latency_ms=round(_percentile(latencies, 95), 3),
        peak_memory_kb=round(peak_memory_kb, 3),
        notes=notes,
    )


def _run_heuristic_benchmark(target_threshold: float) -> tuple[AdapterRunResult, list[dict[str, Any]]]:
    adapter = RightHemisphereJEPA()
    per_sample: list[dict[str, Any]] = []

    tracemalloc.start()
    for index, sample in enumerate(DATASET):
        turn = UserTurn(session_id=f"benchmark-heuristic-{index}", user_text=sample.text, signals=[])
        started = time.perf_counter()
        state = adapter.perceive(turn)
        latency_ms = (time.perf_counter() - started) * 1000.0
        per_sample.append(
            {
                "text": sample.text,
                "target": sample.target,
                "salience": state.salience,
                "confidence": state.confidence,
                "labels": state.emotional_labels,
                "latency_ms": round(latency_ms, 3),
            }
        )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    result = _summarize_run(
        adapter_name="RightHemisphereJEPA",
        mode="heuristic",
        target_threshold=target_threshold,
        per_sample=per_sample,
        peak_memory_kb=peak / 1024.0,
    )
    return result, per_sample


@contextmanager
def _patched_embedding_modules():
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.logging = types.SimpleNamespace(set_verbosity_error=lambda: None)
    fake_transformers.utils = types.SimpleNamespace(logging=types.SimpleNamespace(disable_progress_bar=lambda: None))

    labels = [
        "urgente",
        "emergencia",
        "triste",
        "ansioso",
        "feliz",
        "frustrado",
        "raiva",
        "medo",
        "preocupado",
        "dor",
        "desespero",
    ]
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    class FakeSentenceTransformer:
        def __init__(self, _: str) -> None:
            pass

        def encode(self, values):
            if isinstance(values, list) and values and values[0] in label_to_idx:
                size = len(label_to_idx)
                vectors = []
                for value in values:
                    vector = [0.02] * size
                    vector[label_to_idx[value]] = 1.0
                    vectors.append(vector)
                return vectors
            if isinstance(values, list):
                text = values[0].lower()
            else:
                text = str(values).lower()
            size = len(label_to_idx)
            vector = [0.03] * size
            keywords = [label for label in labels if label in text]
            for keyword in keywords:
                vector[label_to_idx[keyword]] = 0.95
            return [vector]

    fake_sentence_transformers = types.ModuleType("sentence_transformers")
    fake_sentence_transformers.SentenceTransformer = FakeSentenceTransformer

    with patch.dict(
        "sys.modules",
        {"transformers": fake_transformers, "sentence_transformers": fake_sentence_transformers},
    ):
        yield


def _run_embedding_simulated_benchmark(target_threshold: float) -> tuple[AdapterRunResult, list[dict[str, Any]]]:
    per_sample: list[dict[str, Any]] = []
    with _patched_embedding_modules():
        adapter = HuggingFaceRightHemisphereAdapter(HuggingFaceRightHemisphereConfig(latent_size=11))

        tracemalloc.start()
        for index, sample in enumerate(DATASET):
            turn = UserTurn(session_id=f"benchmark-embedding-{index}", user_text=sample.text, signals=[])
            started = time.perf_counter()
            state = adapter.perceive(turn)
            latency_ms = (time.perf_counter() - started) * 1000.0
            per_sample.append(
                {
                    "text": sample.text,
                    "target": sample.target,
                    "salience": state.salience,
                    "confidence": state.confidence,
                    "labels": state.emotional_labels,
                    "latency_ms": round(latency_ms, 3),
                }
            )
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    result = _summarize_run(
        adapter_name="HuggingFaceRightHemisphereAdapter",
        mode="embedding_simulated_offline",
        target_threshold=target_threshold,
        per_sample=per_sample,
        peak_memory_kb=peak / 1024.0,
        notes="simulation used because offline environment cannot download HF checkpoints",
    )
    return result, per_sample


def run_benchmark(target_threshold: float = 0.7) -> dict[str, Any]:
    heuristic_result, heuristic_samples = _run_heuristic_benchmark(target_threshold)
    embedding_result, embedding_samples = _run_embedding_simulated_benchmark(target_threshold)
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target_threshold": target_threshold,
        "dataset_size": len(DATASET),
        "results": [asdict(heuristic_result), asdict(embedding_result)],
        "samples": {
            "heuristic": heuristic_samples,
            "embedding_simulated_offline": embedding_samples,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark local do hemisferio direito (heuristico vs embedding simulado).")
    parser.add_argument("--output-json", type=Path, default=None, help="Caminho opcional para salvar o JSON do benchmark.")
    parser.add_argument("--threshold", type=float, default=0.7, help="Threshold de classificacao high/neutral via salience.")
    args = parser.parse_args()

    payload = run_benchmark(target_threshold=args.threshold)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
