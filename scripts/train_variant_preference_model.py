from __future__ import annotations

import argparse
import json
from pathlib import Path

from calosum.adapters.experience.variant_preference import (
    VariantPreferenceDatasetStore,
    VariantPreferenceModel,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train learned selector for group-turn variants")
    parser.add_argument(
        "--dataset",
        default=".calosum-runtime/reflection/group_turn_dataset.jsonl",
        help="Path to group-turn dataset JSONL",
    )
    parser.add_argument(
        "--artifact",
        default=".calosum-runtime/reflection/variant_preference_model.joblib",
        help="Path to trained model artifact",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=200,
        help="Minimum dataset size before training",
    )
    parser.add_argument(
        "--holdout-ratio",
        type=float,
        default=0.2,
        help="Holdout split ratio",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset_store = VariantPreferenceDatasetStore(Path(args.dataset))
    model = VariantPreferenceModel(
        artifact_path=Path(args.artifact),
        min_samples=max(1, args.min_samples),
        holdout_ratio=max(0.1, min(0.4, args.holdout_ratio)),
    )
    report = model.train(dataset_store.read_all())
    payload = {
        "trained": report.trained,
        "sample_count": report.sample_count,
        "holdout_accuracy": round(report.holdout_accuracy, 4),
        "reason": report.reason,
        "artifact": str(Path(args.artifact)),
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if report.trained else 2


if __name__ == "__main__":
    raise SystemExit(main())
