from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import httpx

from calosum.bootstrap.infrastructure.settings import InfrastructureSettings
from calosum.bootstrap.wiring.factory import CalosumAgentBuilder
from calosum.shared.models.types import UserTurn


@dataclass(slots=True)
class ArmMetrics:
    avg_judge_score: float
    avg_tool_success_rate: float
    latency_p50_ms: float
    latency_p95_ms: float


@dataclass(slots=True)
class AbStats:
    mean_diff: float
    p_value_one_sided: float
    n: int


@dataclass(slots=True)
class AbReport:
    generated_at: str
    dataset_path: str
    turns_evaluated: int
    judge_model: str
    control: ArmMetrics
    treatment_a: ArmMetrics
    treatment_b: ArmMetrics
    a_vs_control: AbStats
    a_vs_b: AbStats
    gate_passed: bool


def _load_dataset(path: Path, limit: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            records.append(json.loads(raw))
            if len(records) >= limit:
                break
    if not records:
        raise RuntimeError(f"dataset is empty: {path}")
    return records


def _load_local_env() -> dict[str, str]:
    env = dict(os.environ)
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            raw = line.strip()
            if not raw or raw.startswith("#") or "=" not in raw:
                continue
            key, value = raw.split("=", 1)
            key = key.strip()
            if key not in env:
                env[key] = value.strip().strip("'\"")
    return env


def _make_prompt(record: dict[str, Any]) -> str:
    context_turns = record.get("context_turns", [])
    context_block = "\n".join(f"- {item}" for item in context_turns)
    return (
        "Contexto da sessao:\n"
        f"{context_block}\n\n"
        "Forneca uma resposta curta (3-5 linhas), objetiva e acionavel. "
        "Priorize coerencia semantica com o contexto, clareza e seguranca."
    )


def _build_agent_for_arm(arm: str, base_env: dict[str, str]) -> Any:
    env = dict(base_env)
    env["CALOSUM_VECTOR_QUANTIZATION"] = "none"
    if arm == "control":
        env["CALOSUM_FUSION_ENABLED"] = "false"
    elif arm == "treatment_a":
        env["CALOSUM_FUSION_ENABLED"] = "true"
        env["CALOSUM_FUSION_CANDIDATES"] = "3"
        env["CALOSUM_FUSION_SELECTION_MODE"] = "guided"
        env["CALOSUM_FUSION_UNCERTAINTY_THRESHOLD"] = "0.5"
    elif arm == "treatment_b":
        env["CALOSUM_FUSION_ENABLED"] = "true"
        env["CALOSUM_FUSION_CANDIDATES"] = "3"
        env["CALOSUM_FUSION_SELECTION_MODE"] = "random"
        env["CALOSUM_FUSION_UNCERTAINTY_THRESHOLD"] = "0.5"
    else:
        raise ValueError(f"unknown arm: {arm}")
    settings = InfrastructureSettings.from_sources(environ=env).with_profile_defaults()
    builder = CalosumAgentBuilder(settings)
    return builder.build()


def _run_left_only_inference(agent: Any, turn: UserTurn) -> tuple[str, float]:
    started = time.perf_counter()
    memory_context = agent.memory_system.build_context(turn)
    right_state = agent.right_hemisphere.perceive(turn, memory_context)
    bridge_packet = agent.tokenizer.translate(right_state)
    left_result = agent.left_hemisphere.reason(
        turn,
        bridge_packet,
        memory_context,
        None,
        0,
        None,
    )
    latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
    return left_result.response_text or "", latency_ms


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = (len(ordered) - 1) * q
    low = int(math.floor(idx))
    high = int(math.ceil(idx))
    if low == high:
        return ordered[low]
    frac = idx - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def _judge_payload(
    *,
    prompt: str,
    responses: dict[str, str],
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are an impartial evaluator. "
                "Score each response from 1 to 5 for semantic coherence with the context. "
                "Output strict JSON only: "
                '{"control": number, "treatment_a": number, "treatment_b": number}.'
            ),
        },
        {
            "role": "user",
            "content": (
                f"Context and task:\n{prompt}\n\n"
                f"CONTROL:\n{responses['control']}\n\n"
                f"TREATMENT_A:\n{responses['treatment_a']}\n\n"
                f"TREATMENT_B:\n{responses['treatment_b']}\n"
            ),
        },
    ]


def _extract_json_object(text: str) -> dict[str, Any]:
    content = text.strip()
    if "```json" in content:
        content = content.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in content:
        content = content.split("```", 1)[1].split("```", 1)[0].strip()
    start = content.find("{")
    end = content.rfind("}")
    if start >= 0 and end > start:
        content = content[start : end + 1]
    try:
        payload = json.loads(content)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    # Fallback for partially malformed outputs: extract numeric scores by key.
    extracted: dict[str, float] = {}
    for key in ("control", "treatment_a", "treatment_b"):
        match = re.search(rf"{key}\D+([1-5](?:\.\d+)?)", content, flags=re.IGNORECASE)
        if match:
            extracted[key] = float(match.group(1))
    if len(extracted) == 3:
        return extracted
    raise ValueError(f"judge output is not parseable JSON: {content[:200]}")


def _judge_triplet(
    *,
    endpoint: str,
    api_key: str,
    model: str,
    provider: str,
    prompt: str,
    responses: dict[str, str],
    timeout_s: float,
    max_retries: int,
) -> dict[str, float]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    if provider.strip().lower() == "openrouter":
        headers["HTTP-Referer"] = "https://github.com/kaiquenog/calosum"
        headers["X-Title"] = "Calosum Fusion AB Test"
    lowered_provider = provider.strip().lower()
    if lowered_provider in {"openai_responses", "openai", "responses"}:
        prompt_messages = _judge_payload(prompt=prompt, responses=responses)
        prompt_text = "\n\n".join(
            f"{item['role'].upper()}:\n{item['content']}" for item in prompt_messages
        )
        payload = {
            "model": model,
            "input": prompt_text,
            "max_output_tokens": 300,
        }
    else:
        payload = {
            "model": model,
            "messages": _judge_payload(prompt=prompt, responses=responses),
            "temperature": 0.0,
            "max_tokens": 200,
            "response_format": {"type": "json_object"},
        }
    judge_url = _resolve_judge_url(endpoint, provider)
    with httpx.Client(timeout=timeout_s, headers=headers) as client:
        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                response = client.post(judge_url, json=payload)
                if response.status_code == 429 and attempt < max_retries:
                    backoff = 2 ** attempt
                    time.sleep(backoff)
                    continue
                if response.status_code >= 400:
                    body = response.text[:400]
                    if attempt < max_retries:
                        backoff = 2 ** attempt
                        time.sleep(backoff)
                        continue
                    raise RuntimeError(
                        f"judge HTTP {response.status_code} for {judge_url}: {body}"
                    )
                data = response.json()
                break
            except Exception as exc:
                last_error = exc
                if attempt >= max_retries:
                    raise
                backoff = 2 ** attempt
                time.sleep(backoff)
        else:
            raise RuntimeError(f"judge request failed: {last_error}")
    if lowered_provider in {"openai_responses", "openai", "responses"}:
        content = _extract_responses_content(data)
    else:
        content = str(data["choices"][0]["message"]["content"])
    parsed = _extract_json_object(content)
    return {
        "control": float(parsed["control"]),
        "treatment_a": float(parsed["treatment_a"]),
        "treatment_b": float(parsed["treatment_b"]),
    }


def _resolve_judge_url(endpoint: str, provider: str) -> str:
    base = endpoint.rstrip("/")
    lowered_provider = provider.strip().lower()
    if base.endswith("/chat/completions"):
        return base
    if lowered_provider in {"openai_responses", "openai", "responses"}:
        if base.endswith("/responses"):
            return base
        if base.endswith("/v1"):
            return f"{base}/responses"
        return f"{base}/v1/responses"
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return base


def _extract_responses_content(data: dict[str, Any]) -> str:
    output_text = data.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()
    chunks: list[str] = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            text = content.get("text", "")
            if isinstance(text, str) and text.strip():
                chunks.append(text.strip())
    return "\n".join(chunks)


def _paired_permutation_p_value_greater(
    left: list[float],
    right: list[float],
    *,
    samples: int,
    seed: int,
) -> AbStats:
    if len(left) != len(right):
        raise ValueError("paired arrays must have same length")
    if not left:
        return AbStats(mean_diff=0.0, p_value_one_sided=1.0, n=0)
    diffs = [a - b for a, b in zip(left, right, strict=False)]
    observed = mean(diffs)
    rng = random.Random(seed)
    ge_count = 0
    for _ in range(max(1, samples)):
        permuted = [d if rng.random() < 0.5 else -d for d in diffs]
        stat = mean(permuted)
        if stat >= observed:
            ge_count += 1
    p_value = (ge_count + 1) / (samples + 1)
    return AbStats(mean_diff=round(observed, 4), p_value_one_sided=round(p_value, 6), n=len(diffs))


def _arm_metrics(scores: list[float], tool_rates: list[float], latencies: list[float]) -> ArmMetrics:
    return ArmMetrics(
        avg_judge_score=round(mean(scores), 4),
        avg_tool_success_rate=round(mean(tool_rates), 4),
        latency_p50_ms=round(_percentile(latencies, 0.5), 3),
        latency_p95_ms=round(_percentile(latencies, 0.95), 3),
    )


def run(args: argparse.Namespace) -> AbReport:
    env = _load_local_env()
    env["CALOSUM_VECTOR_QUANTIZATION"] = "none"
    records = _load_dataset(args.dataset, args.limit)
    arms = ["control", "treatment_a", "treatment_b"]
    agents = {arm: _build_agent_for_arm(arm, env) for arm in arms}
    base_settings = InfrastructureSettings.from_sources(environ=env).with_profile_defaults()

    generated_text: dict[str, list[str]] = {arm: [] for arm in arms}
    tool_rates: dict[str, list[float]] = {arm: [] for arm in arms}
    latencies: dict[str, list[float]] = {arm: [] for arm in arms}
    judge_scores: dict[str, list[float]] = {arm: [] for arm in arms}
    last_judge_at: float | None = None

    endpoint = env.get("CALOSUM_FUSION_JUDGE_ENDPOINT") or base_settings.left_hemisphere_endpoint
    model = env.get("CALOSUM_FUSION_JUDGE_MODEL") or base_settings.left_hemisphere_model
    provider = env.get("CALOSUM_FUSION_JUDGE_PROVIDER") or base_settings.left_hemisphere_provider or "auto"
    api_key = env.get("CALOSUM_FUSION_JUDGE_API_KEY") or base_settings.left_hemisphere_api_key or ""
    if not endpoint or not model:
        raise RuntimeError(
            "missing judge config: set CALOSUM_FUSION_JUDGE_ENDPOINT/CALOSUM_FUSION_JUDGE_MODEL "
            "or reuse CALOSUM_LEFT_ENDPOINT/CALOSUM_LEFT_MODEL"
        )

    for idx, record in enumerate(records):
        prompt = _make_prompt(record)
        for arm in arms:
            turn = UserTurn(
                session_id=f"fusion-ab-{arm}-{idx}",
                user_text=prompt,
            )
            text, latency_ms = _run_left_only_inference(agents[arm], turn)
            generated_text[arm].append(text)
            tool_rates[arm].append(1.0)
            latencies[arm].append(latency_ms)

        if last_judge_at is not None and args.judge_throttle_s > 0:
            elapsed = time.time() - last_judge_at
            wait_for = args.judge_throttle_s - elapsed
            if wait_for > 0:
                time.sleep(wait_for)
        scored = _judge_triplet(
            endpoint=endpoint,
            api_key=api_key,
            model=model,
            provider=provider,
            prompt=prompt,
            responses={
                "control": generated_text["control"][-1],
                "treatment_a": generated_text["treatment_a"][-1],
                "treatment_b": generated_text["treatment_b"][-1],
            },
            timeout_s=args.judge_timeout_s,
            max_retries=args.judge_max_retries,
        )
        last_judge_at = time.time()
        for arm in arms:
            judge_scores[arm].append(scored[arm])
        print(
            f"[{idx + 1}/{len(records)}] judge "
            f"C={scored['control']:.2f} A={scored['treatment_a']:.2f} B={scored['treatment_b']:.2f}",
            flush=True,
        )

    control_metrics = _arm_metrics(judge_scores["control"], tool_rates["control"], latencies["control"])
    treatment_a_metrics = _arm_metrics(judge_scores["treatment_a"], tool_rates["treatment_a"], latencies["treatment_a"])
    treatment_b_metrics = _arm_metrics(judge_scores["treatment_b"], tool_rates["treatment_b"], latencies["treatment_b"])

    a_vs_control = _paired_permutation_p_value_greater(
        judge_scores["treatment_a"],
        judge_scores["control"],
        samples=args.permutation_samples,
        seed=args.seed,
    )
    a_vs_b = _paired_permutation_p_value_greater(
        judge_scores["treatment_a"],
        judge_scores["treatment_b"],
        samples=args.permutation_samples,
        seed=args.seed + 1,
    )

    gate_passed = (
        treatment_a_metrics.avg_judge_score > control_metrics.avg_judge_score
        and treatment_a_metrics.avg_judge_score > treatment_b_metrics.avg_judge_score
        and a_vs_control.p_value_one_sided < 0.05
        and a_vs_b.p_value_one_sided < 0.05
    )

    report = AbReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        dataset_path=str(args.dataset),
        turns_evaluated=len(records),
        judge_model=model,
        control=control_metrics,
        treatment_a=treatment_a_metrics,
        treatment_b=treatment_b_metrics,
        a_vs_control=a_vs_control,
        a_vs_b=a_vs_b,
        gate_passed=gate_passed,
    )

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(asdict(report), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Fusion A/B benchmark with LLM-as-judge.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("docs/benchmarks/jepa_phase1/annotated_turns.jsonl"),
    )
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output-json", type=Path, default=Path("docs/reports/fusion_ab_test_2026-04-02.json"))
    parser.add_argument("--permutation-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--judge-timeout-s", type=float, default=30.0)
    parser.add_argument("--judge-max-retries", type=int, default=5)
    parser.add_argument("--judge-throttle-s", type=float, default=20.0)
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
