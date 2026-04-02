from __future__ import annotations

import argparse
import json
from pathlib import Path


MINIMUM_COVERAGE = 80.0


def _normalize_path(raw: str) -> str:
    path = raw.strip().replace('\\', '/')
    while path.startswith('./'):
        path = path[2:]
    return path


def _load_coverage_percentages(coverage_json: Path) -> dict[str, float]:
    payload = json.loads(coverage_json.read_text(encoding='utf-8'))
    files = payload.get('files', {})
    percentages: dict[str, float] = {}
    for file_path, details in files.items():
        normalized = _normalize_path(file_path)
        summary = details.get('summary', {})
        covered = float(summary.get('covered_lines', 0.0))
        total = float(summary.get('num_statements', 0.0))
        percentage = 100.0 if total == 0 else (covered / total) * 100.0
        percentages[normalized] = percentage
    return percentages


def _load_changed_python_files(changed_list: Path) -> list[str]:
    files: list[str] = []
    for line in changed_list.read_text(encoding='utf-8').splitlines():
        normalized = _normalize_path(line)
        if normalized.startswith('src/calosum/') and normalized.endswith('.py'):
            files.append(normalized)
    return sorted(set(files))


def main() -> int:
    parser = argparse.ArgumentParser(description='Gate de cobertura para modulos Python novos/alterados.')
    parser.add_argument('--coverage-json', type=Path, required=True)
    parser.add_argument('--changed-files', type=Path, required=True)
    parser.add_argument('--minimum', type=float, default=MINIMUM_COVERAGE)
    args = parser.parse_args()

    changed = _load_changed_python_files(args.changed_files)
    if not changed:
        print('coverage gate: sem modulos Python alterados em src/calosum; gate ignorado')
        return 0

    coverage = _load_coverage_percentages(args.coverage_json)
    failed: list[tuple[str, float]] = []

    for module in changed:
        module_coverage = coverage.get(module)
        if module_coverage is None:
            failed.append((module, 0.0))
            continue
        if module_coverage < args.minimum:
            failed.append((module, module_coverage))

    if failed:
        print('coverage gate: FALHOU')
        for module, value in failed:
            print(f'- {module}: {value:.2f}% (< {args.minimum:.2f}%)')
        return 1

    print('coverage gate: PASSOU')
    for module in changed:
        print(f'- {module}: {coverage.get(module, 0.0):.2f}%')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
