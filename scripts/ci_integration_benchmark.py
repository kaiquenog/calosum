from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

from calosum import (
    CalosumAgent,
    LeftHemisphereResult,
    PrimitiveAction,
    TypedLambdaProgram,
    UserTurn,
)


@dataclass(slots=True)
class IntegrationReport:
    turns: int
    latency_p50_ms: float
    latency_p95_ms: float
    tool_success_rate: float


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = int(round((len(ordered) - 1) * q))
    return ordered[idx]


class MockLeftHemisphere:
    def reason(self, user_turn, bridge_packet, memory_context, runtime_feedback=None, attempt=0):
        _ = user_turn, bridge_packet, memory_context, runtime_feedback, attempt
        return LeftHemisphereResult(
            response_text='Plano objetivo: priorize tarefas criticas, execute em blocos de 30min e revise riscos.',
            lambda_program=TypedLambdaProgram('Context -> Plan', 'lambda _: propose_plan()', 'plan'),
            actions=[
                PrimitiveAction(
                    action_type='propose_plan',
                    typed_signature='Context -> Plan',
                    payload={'steps': ['mapear backlog', 'executar top-3', 'revisar risco']},
                    safety_invariants=['safe'],
                )
            ],
            reasoning_summary=[],
        )

    async def areason(self, *args, **kwargs):
        return self.reason(*args[:3])

    def repair(self, *args, **kwargs):
        return self.reason(*args[:3])

    async def arepair(self, *args, **kwargs):
        return self.reason(*args[:3])


def _run(turns: int) -> IntegrationReport:
    agent = CalosumAgent(left_hemisphere=MockLeftHemisphere())
    latencies: list[float] = []
    executed_actions = 0
    successful_actions = 0

    for idx in range(turns):
        turn = UserTurn(
            session_id='ci-ephemeral-integration',
            user_text=(
                'Estou ansioso porque preciso reorganizar um projeto complexo com prazos apertados. '
                f'Turno {idx + 1}.'
            ),
        )
        started = perf_counter()
        result = agent.process_turn(turn)
        selected_result = getattr(result, 'selected_result', result)
        latencies.append((perf_counter() - started) * 1000.0)
        for item in selected_result.execution_results:
            executed_actions += 1
            if item.status not in {'failed', 'error', 'rejected'}:
                successful_actions += 1

    success_rate = 1.0 if executed_actions == 0 else successful_actions / executed_actions
    return IntegrationReport(
        turns=turns,
        latency_p50_ms=round(_percentile(latencies, 0.50), 3),
        latency_p95_ms=round(_percentile(latencies, 0.95), 3),
        tool_success_rate=round(success_rate, 4),
    )


def _to_markdown(report: IntegrationReport) -> str:
    return '\n'.join(
        [
            '# CI Integration Benchmark',
            '',
            '| Metric | Value |',
            '|---|---:|',
            f'| turns | {report.turns} |',
            f'| latency_p50_ms | {report.latency_p50_ms} |',
            f'| latency_p95_ms | {report.latency_p95_ms} |',
            f'| tool_success_rate | {report.tool_success_rate} |',
            '',
            'Gate principal: latency_p95_ms <= 5000 em modo ephemeral com LLM mockado.',
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description='Benchmark de integracao para gate de latencia/estabilidade no CI.')
    parser.add_argument('--turns', type=int, default=20)
    parser.add_argument('--latency-p95-threshold-ms', type=float, default=5000.0)
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--output-md', type=Path, required=True)
    args = parser.parse_args()

    report = _run(turns=max(1, args.turns))
    payload = asdict(report)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    args.output_md.write_text(_to_markdown(report), encoding='utf-8')

    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if report.latency_p95_ms > args.latency_p95_threshold_ms:
        print(
            'integration gate: FALHOU '
            f'(latency_p95_ms={report.latency_p95_ms} > {args.latency_p95_threshold_ms})'
        )
        return 1

    print('integration gate: PASSOU')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
