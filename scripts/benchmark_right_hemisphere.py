"""Benchmark do hemisfério direito HuggingFace com checkpoint real.

Uso:
    PYTHONPATH=src .venv/bin/python3 scripts/benchmark_right_hemisphere.py

Requer:
    - sentence-transformers instalado (já listado em requirements/pyproject.toml)
    - CPU-only por padrão (sem GPU necessária)

Saída:
    docs/reports/benchmark_right_hemi_YYYY-MM-DD.md
"""
from __future__ import annotations

import datetime
import statistics
import sys
import time
from pathlib import Path

# Garante que src/ está no path quando rodado diretamente
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Modelo leve (~90 MB), CPU-friendly, já usado nos adapters
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
N_QUERIES = 50
REPORT_DIR = Path(__file__).resolve().parents[1] / "docs" / "reports"

SAMPLE_QUERIES = [
    "Qual é a capital do Brasil?",
    "Explique o conceito de entropia.",
    "Como funciona um transformador neural?",
    "O que é metacognição?",
    "Descreva o hemisfério direito do Calosum.",
    "Como resolver um conflito ético em IA?",
    "Qual a diferença entre RAG e fine-tuning?",
    "O que é active inference?",
    "Descreva a arquitetura Ports and Adapters.",
    "Como implementar um bandit UCB1?",
]


def _build_user_turn(text: str):
    """Cria um UserTurn mínimo para o hemisfério direito."""
    from calosum.shared.types import UserTurn
    return UserTurn(
        session_id="benchmark",
        turn_id=f"bench-{int(time.time_ns())}",
        user_text=text,
    )


def _run_benchmark() -> dict:
    """Executa N queries e coleta métricas de latência e surprise_score."""
    from calosum.adapters.right_hemisphere_hf import HuggingFaceRightHemisphere

    print(f"Carregando modelo: {MODEL_NAME} (CPU-only)...")
    t0 = time.perf_counter()
    adapter = HuggingFaceRightHemisphere(model_name=MODEL_NAME, device="cpu")
    load_time = time.perf_counter() - t0
    print(f"  Modelo carregado em {load_time:.2f}s")

    queries = (SAMPLE_QUERIES * ((N_QUERIES // len(SAMPLE_QUERIES)) + 1))[:N_QUERIES]

    latencies: list[float] = []
    surprise_scores: list[float] = []

    print(f"Rodando {N_QUERIES} queries...")
    for i, q in enumerate(queries):
        turn = _build_user_turn(q)
        t_start = time.perf_counter()
        result = adapter.perceive(turn)
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        latencies.append(elapsed_ms)
        surprise_scores.append(result.surprise_score)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{N_QUERIES} queries concluídas")

    return {
        "model": MODEL_NAME,
        "n_queries": N_QUERIES,
        "load_time_s": round(load_time, 3),
        "latency_mean_ms": round(statistics.mean(latencies), 2),
        "latency_median_ms": round(statistics.median(latencies), 2),
        "latency_p95_ms": round(sorted(latencies)[int(0.95 * len(latencies))], 2),
        "latency_min_ms": round(min(latencies), 2),
        "latency_max_ms": round(max(latencies), 2),
        "surprise_mean": round(statistics.mean(surprise_scores), 4),
        "surprise_stdev": round(statistics.stdev(surprise_scores) if len(surprise_scores) > 1 else 0.0, 4),
    }


def _write_report(metrics: dict) -> Path:
    """Gera relatório markdown em docs/reports/."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.date.today().isoformat()
    report_path = REPORT_DIR / f"benchmark_right_hemi_{today}.md"

    lines = [
        f"# Benchmark Hemisfério Direito — {today}",
        "",
        "## Configuração",
        "",
        f"| Parâmetro | Valor |",
        f"|---|---|",
        f"| Modelo | `{metrics['model']}` |",
        f"| Device | CPU-only |",
        f"| N queries | {metrics['n_queries']} |",
        f"| Tempo de carregamento | {metrics['load_time_s']}s |",
        "",
        "## Resultados de Latência (ms)",
        "",
        "| Métrica | Valor |",
        "|---|---|",
        f"| Média | {metrics['latency_mean_ms']} ms |",
        f"| Mediana | {metrics['latency_median_ms']} ms |",
        f"| P95 | {metrics['latency_p95_ms']} ms |",
        f"| Mín | {metrics['latency_min_ms']} ms |",
        f"| Máx | {metrics['latency_max_ms']} ms |",
        "",
        "## Qualidade — Surprise Score",
        "",
        "| Métrica | Valor |",
        "|---|---|",
        f"| Média | {metrics['surprise_mean']} |",
        f"| Desvio padrão | {metrics['surprise_stdev']} |",
        "",
        "## Notas",
        "",
        "- Baseline gerado para comparação futura entre backends (HF, VJEPA21, jepars).",
        "- Executado em CPU sem GPU — latências esperadamente mais altas que produção com GPU.",
        f"- Gerado automaticamente por `scripts/benchmark_right_hemisphere.py`.",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> int:
    try:
        metrics = _run_benchmark()
    except Exception as exc:
        print(f"ERRO durante benchmark: {exc}", file=sys.stderr)
        return 1

    print("\n=== Resultados ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    report = _write_report(metrics)
    print(f"\nRelatório salvo em: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
