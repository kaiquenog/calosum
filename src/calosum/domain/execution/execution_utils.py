from typing import Any

from calosum.shared.models.signals import StructuredMismatchSignal
from calosum.shared.models.types import (
    ActionExecutionResult,
    PerceptionSummary,
    CognitiveTelemetrySnapshot,
    CritiqueVerdict,
    ActionPlannerResult,
    InputPerceptionState,
)

def build_execution_telemetry(
    right_state: InputPerceptionState,
    bridge_packet: PerceptionSummary,
    left_result: ActionPlannerResult,
    execution_results: list[ActionExecutionResult],
    retry_count: int,
    critique_revision_count: int,
    critique_verdict: CritiqueVerdict | None = None,
    capabilities: dict[str, Any] | None = None,
    variant_label: str | None = None,
) -> CognitiveTelemetrySnapshot:
    action_count = len(execution_results)
    executed_count = sum(1 for result in execution_results if result.status == "executed")
    rejected_count = sum(1 for result in execution_results if result.status == "rejected")
    tool_success_rate = round(executed_count / action_count, 3) if action_count else 1.0
    
    bridge_config = {
        "target_temperature": bridge_packet.control.target_temperature,
        "empathy_priority": bridge_packet.control.empathy_priority,
        "system_directives_count": len(bridge_packet.control.system_directives),
    }
    mismatch_signal = left_result.telemetry.get("structured_mismatch_signal")
    
    return CognitiveTelemetrySnapshot(
        felt={
            "context_id": right_state.context_id,
            "emotional_labels": right_state.emotional_labels,
            "salience": right_state.salience,
            "world_hypotheses": right_state.world_hypotheses,
            "surprise_score": right_state.surprise_score,
            "telemetry": right_state.telemetry,
        },
        thought={
            "lambda_signature": left_result.lambda_program.signature,
            "reasoning_summary": left_result.reasoning_summary,
            "system_directives": left_result.telemetry.get("system_directives", []),
            "runtime_retry_count": retry_count,
            "critique_revision_count": critique_revision_count,
            "critique_verdict": {
                "is_valid": critique_verdict.is_valid,
                "failure_types": [item.value for item in critique_verdict.failure_types],
                "identified_issues": critique_verdict.identified_issues,
                "suggested_fixes": critique_verdict.suggested_fixes,
                "confidence": critique_verdict.confidence,
            } if critique_verdict else None,
            "cognitive_override_detected": bool(mismatch_signal),
            "structured_mismatch_signal": mismatch_signal,
        },
        decision={
            "response_text": left_result.response_text,
            "action_types": [action.action_type for action in left_result.actions],
            "action_count": action_count,
            "tool_success_rate": tool_success_rate,
            "runtime_retry_count": retry_count,
            "runtime_rejected_count": rejected_count,
            "critique_revision_count": critique_revision_count,
        },
        capabilities=capabilities or {},
        bridge_config=bridge_config,
        active_variant=variant_label,
    )


def build_structured_mismatch_signal(
    execution_results: list[ActionExecutionResult],
    critique_verdict: CritiqueVerdict | None,
) -> StructuredMismatchSignal | None:
    rejected = [result for result in execution_results if result.status == "rejected"]
    verifier_invalid = critique_verdict is not None and not critique_verdict.is_valid
    if not rejected and not verifier_invalid:
        return None

    reasons = [violation for result in rejected for violation in result.violations]
    if critique_verdict is not None:
        reasons.extend(critique_verdict.identified_issues)
    reasons = list(dict.fromkeys(reason for reason in reasons if reason))

    failure_types = []
    if critique_verdict is not None:
        failure_types = [item.value for item in critique_verdict.failure_types]
    source = "verifier" if verifier_invalid else "runtime"
    severity = min(
        1.0,
        (0.2 * len(rejected))
        + (0.15 * len(failure_types))
        + (0.1 if verifier_invalid else 0.0),
    )
    directives = [
        "prioritize contract-safe actions before expressive responses",
        "if uncertain, clarify or inspect before acting",
    ]
    return StructuredMismatchSignal(
        source=source,
        severity=severity,
        reasons=reasons[:6],
        rejected_action_types=[result.action_type for result in rejected],
        failure_types=failure_types,
        recommended_bridge_directives=directives,
    )

def ensure_response_text(
    left_result: ActionPlannerResult,
    execution_results: list[ActionExecutionResult],
) -> str:
    if left_result.response_text.strip():
        return left_result.response_text

    for result in reversed(execution_results):
        if result.status != "executed": continue
        if result.action_type == "respond_text":
            text = str(result.output.get("message") or result.output.get("result") or "").strip()
            if text: return text

    for result in reversed(execution_results):
        if result.status != "executed" or result.action_type != "propose_plan": continue
        steps = result.output.get("steps")
        if isinstance(steps, list) and steps:
            rendered = " ".join(f"{index + 1}. {str(step).strip()}" for index, step in enumerate(steps[:3]))
            if rendered: return f"Plano sugerido: {rendered}"
    
    if any(result.status == "executed" for result in execution_results):
        return "Execucao concluida com sucesso."
    return left_result.response_text
