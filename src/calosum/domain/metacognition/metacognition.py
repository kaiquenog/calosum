from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from calosum.shared.models.types import AgentTurnResult, InputPerceptionState, UserTurn


@dataclass(slots=True)
class CognitiveVariantSpec:
    variant_id: str
    tokenizer_overrides: dict[str, Any] = field(default_factory=dict)
    left_overrides: dict[str, Any] = field(default_factory=dict)
    bridge_directives: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CognitiveCandidate:
    variant: CognitiveVariantSpec
    turn_result: AgentTurnResult


@dataclass(slots=True)
class ReflectionScore:
    variant_id: str
    score: float
    reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ReflectionOutcome:
    selected_variant_id: str
    scoreboard: list[ReflectionScore] = field(default_factory=list)
    selected_by: str = "linear_pass"
    bridge_adjustments: dict[str, Any] = field(default_factory=dict)
    selected_metrics: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    pruning_reasons: dict[str, str] = field(default_factory=dict)
    cost_metrics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected_variant_id": self.selected_variant_id,
            "selected_by": self.selected_by,
        }


@dataclass(slots=True)
class GroupTurnResult:
    user_turn: UserTurn
    right_state: InputPerceptionState
    candidates: list[CognitiveCandidate]
    selected_result: AgentTurnResult
    reflection: ReflectionOutcome


def default_cognitive_personas(max_width: int = 1) -> list[CognitiveVariantSpec]:
    """Simplified to return a single base persona."""
    return [
        CognitiveVariantSpec(
            variant_id="base",
            notes=["linear_flow"],
        )
    ]


class LinearReflectionController:
    """
    Substituto simplificado para o GEAReflectionController.
    Mantém a interface mas executa apenas uma passagem linear sem branching.
    """

    def __init__(self, *args, **kwargs) -> None:
        pass

    def evaluate(
        self,
        candidates: list[CognitiveCandidate],
        base_tokenizer: Any,
    ) -> ReflectionOutcome:
        if not candidates:
            raise ValueError("Reflection requires at least one candidate")
        
        winner = candidates[0]
        return ReflectionOutcome(
            selected_variant_id=winner.variant.variant_id,
            selected_by="linear_no_branch",
            notes=["GEA disabled for performance"],
        )

    async def aevaluate(
        self,
        candidates: list[CognitiveCandidate],
        base_tokenizer: Any,
    ) -> ReflectionOutcome:
        return self.evaluate(candidates, base_tokenizer)

    def apply_config_adaptation(self, *args, **kwargs):
        pass

    def apply_neuroplasticity(self, *args, **kwargs):
        pass


GEAReflectionController = LinearReflectionController
CognitiveVariantSelector = LinearReflectionController
