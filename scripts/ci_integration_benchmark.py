from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from calosum.domain.agent.orchestrator import CalosumAgent
from calosum.shared.models.types import (
    ActionExecutionResult,
    ActionPlannerResult,
    PrimitiveAction,
    TypedLambdaProgram,
    UserTurn,
)


class BenchmarkLeftHemisphere:
    """Planner deterministico para smoke benchmark sem rede."""

    def reason(self, user_turn, bridge_packet, memory_context, runtime_feedback=None, attempt=0, workspace=None):
        return _plan_for_turn(user_turn.user_text)

    async def areason(self, user_turn, bridge_packet, memory_context, runtime_feedback=None, attempt=0, workspace=None):
        return self.reason(user_turn, bridge_packet, memory_context, runtime_feedback, attempt, workspace)

    def repair(
        self,
        user_turn,
        bridge_packet,
        memory_context,
        previous_result,
        rejected_results,
        attempt,
        critique_feedback=None,
        workspace=None,
    ):
        return _plan_for_turn(user_turn.user_text)

    async def arepair(
        self,
        user_turn,
        bridge_packet,
        memory_context,
        previous_result,
        rejected_results,
        attempt,
        critique_feedback=None,
        workspace=None,
    ):
        return self.repair(
            user_turn,
            bridge_packet,
            memory_context,
            previous_result,
            rejected_results,
            attempt,
            critique_feedback,
            workspace,
        )


def _plan_for_turn(text: str) -> ActionPlannerResult:
    return ActionPlannerResult(
        response_text=f"Plano seguro gerado para: {text}",
        lambda_program=TypedLambdaProgram(
            "Context -> ResponsePlan",
            json.dumps({"plan": ["respond_text"]}, ensure_ascii=False),
            "respond",
        ),
        actions=[
            PrimitiveAction(
                action_type="respond_text",
                typed_signature="ResponsePlan -> SafeTextMessage",
                payload={"text": text[:120]},
                safety_invariants=["no_external_side_effects"],
            )
        ],
        reasoning_summary=["ci_smoke_benchmark"],
        telemetry={"adapter": "BenchmarkLeftHemisphere"},
    )


def _turn_corpus() -> list[str]:
    return [
        "Estou ansioso e preciso reorganizar um projeto complexo com prazos apertados.",
        "Preciso de um plano curto e seguro para revisar erros antes do deploy.",
        "Resuma riscos operacionais e proponha uma resposta objetiva.",
        "Quero passos claros para priorizar tarefas técnicas sem perder confiabilidade.",
        "Explique a ordem correta para estabilizar CI, deploy e bootstrap.",
    ]


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * percentile)))
    return round(ordered[rank], 3)


def _tool_success_rate(execution_results: list[ActionExecutionResult]) -> float:
    if not execution_results:
        return 1.0
    successful = sum(1 for result in execution_results if result.status in {"executed", "planned", "success"})
    return successful / len(execution_results)


def run_benchmark(turns: int) -> dict[str, Any]:
    agent = CalosumAgent(left_hemisphere=BenchmarkLeftHemisphere())
    corpus = _turn_corpus()
    latencies: list[float] = []
    tool_rates: list[float] = []
    runtime_retry_count = 0
    heuristic_fallback_turns = 0

    for index in range(turns):
        prompt = corpus[index % len(corpus)]
        result = agent.process_turn(UserTurn(session_id="ci-benchmark", user_text=prompt))
        latencies.append(float(result.latency_ms))
        tool_rates.append(_tool_success_rate(result.execution_results))
        runtime_retry_count += int(result.runtime_retry_count)
        if result.right_state.telemetry.get("degraded_reason"):
            heuristic_fallback_turns += 1

    return {
        "turns_executed": turns,
        "latency_p50_ms": _percentile(latencies, 0.50),
        "latency_p95_ms": _percentile(latencies, 0.95),
        "tool_success_rate": round(sum(tool_rates) / len(tool_rates), 4) if tool_rates else 1.0,
        "runtime_retry_count": runtime_retry_count,
        "fallback_rate_heuristic_jepa": round(heuristic_fallback_turns / turns, 4) if turns else 0.0,
    }


def _write_outputs(output_json: Path, output_md: Path, metrics: dict[str, Any]) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "description": "Smoke benchmark deterministico do pipeline dual-hemisphere sem dependencia de endpoint externo.",
        "metrics": metrics,
    }
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Integration Benchmark Results",
        "",
        f"- turns_executed: {metrics['turns_executed']}",
        f"- latency_p50_ms: {metrics['latency_p50_ms']}",
        f"- latency_p95_ms: {metrics['latency_p95_ms']}",
        f"- tool_success_rate: {metrics['tool_success_rate']}",
        f"- runtime_retry_count: {metrics['runtime_retry_count']}",
        f"- fallback_rate_heuristic_jepa: {metrics['fallback_rate_heuristic_jepa']}",
    ]
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=int, default=20)
    parser.add_argument("--latency-p95-threshold-ms", type=int, default=5000)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    turns = int(os.environ.get("CALOSUM_CI_BENCHMARK_TURNS", args.turns))
    print(f"Running integration benchmark with {turns} turns...")

    metrics = run_benchmark(turns)
    _write_outputs(Path(args.output_json), Path(args.output_md), metrics)

    if metrics["latency_p95_ms"] > args.latency_p95_threshold_ms:
        print(
            f"FAIL: Latency p95 ({metrics['latency_p95_ms']}ms) exceeded threshold "
            f"({args.latency_p95_threshold_ms}ms)"
        )
        return 1

    print("Benchmark completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
