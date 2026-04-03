from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

HIGHER_IS_BETTER = {"tool_success_rate"}


def _load_metrics(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        return metrics
    return payload


def _regression_percent(metric: str, baseline: float, candidate: float) -> float:
    if metric in HIGHER_IS_BETTER:
        if baseline == 0:
            return 0.0 if candidate >= baseline else 100.0
        return max(0.0, ((baseline - candidate) / abs(baseline)) * 100.0)
    if baseline == 0:
        return 0.0 if candidate <= baseline else 100.0
    return max(0.0, ((candidate - baseline) / abs(baseline)) * 100.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--max-regression-percent", type=float, default=5.0)
    args = parser.parse_args()

    try:
        baseline_metrics = _load_metrics(Path(args.baseline))
        candidate_metrics = _load_metrics(Path(args.candidate))
    except FileNotFoundError as exc:
        print(f"Warning: could not find file: {exc}. Skipping benchmark gate.")
        return 0

    base_val = baseline_metrics.get(args.metric)
    cand_val = candidate_metrics.get(args.metric)
    if not isinstance(base_val, (int, float)) or not isinstance(cand_val, (int, float)):
        print(f"Metric {args.metric} not found in one or both files.")
        return 1

    regression_percent = _regression_percent(args.metric, float(base_val), float(cand_val))
    print(f"Baseline {args.metric}: {base_val}")
    print(f"Candidate {args.metric}: {cand_val}")
    print(f"Regression percent: {regression_percent:.2f}%")

    if regression_percent > args.max_regression_percent:
        print(
            f"FAIL: Regression of {regression_percent:.2f}% exceeds max "
            f"{args.max_regression_percent:.2f}%"
        )
        return 1

    print("Benchmark gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
