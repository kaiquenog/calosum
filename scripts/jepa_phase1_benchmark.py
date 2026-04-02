from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from calosum.adapters.hemisphere.right_hemisphere_heuristic_jepa import HeuristicJEPAAdapter
from calosum.shared.models.types import UserTurn


@dataclass(slots=True)
class BenchmarkResult:
    total_turns: int
    ranking_top1_accuracy: float
    off_topic_surprise_rate: float
    ranking_gate_passed: bool
    off_topic_gate_passed: bool
    ranking_correct: int
    off_topic_high_surprise_count: int
    average_good_surprise: float
    average_off_topic_surprise: float


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            records.append(json.loads(raw))
    return records


async def _evaluate(records: list[dict[str, Any]]) -> BenchmarkResult:
    adapter = HeuristicJEPAAdapter()

    ranking_correct = 0
    off_topic_high_surprise_count = 0
    good_surprises: list[float] = []
    off_topic_surprises: list[float] = []

    for idx, record in enumerate(records):
        context_turns = [
            UserTurn(session_id=f"jepa-bench-{idx}", user_text=text)
            for text in record["context_turns"]
        ]
        context = await adapter.encode_context(context_turns)
        candidates = [record["good_response"], *record["bad_responses"]]
        ranked = adapter.score_candidates(context, candidates)
        if ranked and ranked[0][0] == record["good_response"]:
            ranking_correct += 1

        good_surprise = await adapter.compute_surprise(context, record["good_response"])
        off_topic_surprise = await adapter.compute_surprise(context, record["off_topic_response"])
        good_surprises.append(good_surprise.prediction_error)
        off_topic_surprises.append(off_topic_surprise.prediction_error)
        if off_topic_surprise.prediction_error > 0.5:
            off_topic_high_surprise_count += 1

    total = len(records)
    ranking_acc = ranking_correct / max(1, total)
    off_topic_rate = off_topic_high_surprise_count / max(1, total)
    avg_good = sum(good_surprises) / max(1, len(good_surprises))
    avg_off_topic = sum(off_topic_surprises) / max(1, len(off_topic_surprises))
    return BenchmarkResult(
        total_turns=total,
        ranking_top1_accuracy=round(ranking_acc, 3),
        off_topic_surprise_rate=round(off_topic_rate, 3),
        ranking_gate_passed=ranking_acc >= 0.60,
        off_topic_gate_passed=off_topic_rate >= 0.70,
        ranking_correct=ranking_correct,
        off_topic_high_surprise_count=off_topic_high_surprise_count,
        average_good_surprise=round(avg_good, 3),
        average_off_topic_surprise=round(avg_off_topic, 3),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="JEPA phase 1 ranking/surprise benchmark.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("docs/benchmarks/jepa_phase1/annotated_turns.jsonl"),
    )
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    records = _load_dataset(args.dataset)
    result = asyncio.run(_evaluate(records))
    payload = {
        "dataset": str(args.dataset),
        "result": asdict(result),
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
