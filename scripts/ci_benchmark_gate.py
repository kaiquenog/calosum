from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_metric(path: Path, metric: str) -> float:
    payload = json.loads(path.read_text(encoding='utf-8'))
    value = payload.get(metric)
    if value is None:
        raise RuntimeError(f'metric {metric!r} not found in {path}')
    return float(value)


def _read_baseline(path: Path, metric: str) -> float:
    payload = json.loads(path.read_text(encoding='utf-8'))
    metrics = payload.get('metrics', {})
    if metric not in metrics:
        raise RuntimeError(f'baseline metric {metric!r} not found in {path}')
    return float(metrics[metric])


def main() -> int:
    parser = argparse.ArgumentParser(description='Gate de regressao para benchmark de CI.')
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--metric', type=str, default='tool_success_rate')
    parser.add_argument('--max-regression-percent', type=float, default=5.0)
    args = parser.parse_args()

    baseline = _read_baseline(args.baseline, args.metric)
    candidate = _read_metric(args.candidate, args.metric)
    min_allowed = baseline * (1.0 - args.max_regression_percent / 100.0)

    print(
        f'benchmark gate ({args.metric}): baseline={baseline:.4f}, '
        f'candidate={candidate:.4f}, min_allowed={min_allowed:.4f}'
    )

    if candidate < min_allowed:
        print(
            'benchmark gate: FALHOU '
            f'(regressao > {args.max_regression_percent:.2f}%)'
        )
        return 1

    print('benchmark gate: PASSOU')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
