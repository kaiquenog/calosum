from __future__ import annotations

import os
from dataclasses import dataclass, field

from calosum.shared.models.types import CognitiveWorkspace, InputPerceptionState, UserTurn


@dataclass(slots=True)
class BranchingDecision:
    candidate_count: int
    reasons: list[str] = field(default_factory=list)


def decide_branching(
    *,
    user_turn: UserTurn,
    right_state: InputPerceptionState,
    workspace: CognitiveWorkspace | None = None,
) -> BranchingDecision:
    max_candidates = max(1, _env_int("CALOSUM_GEA_MAX_CANDIDATES", 1))
    surprise_threshold = _env_float("CALOSUM_GEA_BRANCH_SURPRISE_THRESHOLD", 0.68)
    complexity_threshold = _env_float("CALOSUM_GEA_BRANCH_COMPLEXITY_THRESHOLD", 0.6)

    reasons: list[str] = []
    if float(right_state.surprise_score) >= surprise_threshold:
        reasons.append("high_surprise")
    if float(right_state.world_hypotheses.get("interaction_complexity", 0.0)) >= complexity_threshold:
        reasons.append("high_complexity")
    if float(right_state.telemetry.get("jepa_uncertainty", 0.0)) >= 0.7:
        reasons.append("high_uncertainty")
    if _looks_explicitly_complex(user_turn.user_text):
        reasons.append("complex_task")
    if _has_recurrent_runtime_repairs(workspace):
        reasons.append("recurrent_repairs")

    if not reasons:
        return BranchingDecision(candidate_count=1)

    candidate_count = min(max_candidates, 1 + len(reasons))
    return BranchingDecision(candidate_count=candidate_count, reasons=reasons)


def _has_recurrent_runtime_repairs(workspace: CognitiveWorkspace | None) -> bool:
    if workspace is None:
        return False
    feedback = workspace.task_frame.get("previous_runtime_feedback", [])
    if not isinstance(feedback, list):
        return False
    repeated_rejections = sum(
        int(item.get("rejected_count", 0))
        for item in feedback
        if isinstance(item, dict)
    )
    return repeated_rejections >= 2


def _looks_explicitly_complex(text: str) -> bool:
    lowered = text.lower()
    markers = (
        "complex",
        "complexo",
        "arquitetura",
        "compare",
        "comparar",
        "estrateg",
        "benchmark",
        "plano",
        "roadmap",
        "tradeoff",
        "refator",
    )
    return any(marker in lowered for marker in markers)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default
