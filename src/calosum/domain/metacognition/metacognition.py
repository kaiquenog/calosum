from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from calosum.shared.models.types import AgentTurnResult, InputPerceptionState, UserTurn
from calosum.shared.utils.math_cognitive import calculate_efe, kl_divergence_gaussian


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


class GEAReflectionController:
    """
    Group Evolution of Agents (GEA) reflection controller.
    Uses Expected Free Energy (EFE) to select the best cognitive variant.
    """

    def __init__(self, *args, **kwargs) -> None:
        pass

    def evaluate(
        self,
        candidates: list[CognitiveCandidate],
        base_tokenizer: Any,
    ) -> ReflectionOutcome:
        import numpy as np

        if not candidates:
            raise ValueError("Reflection requires at least one candidate")
        scoreboard = []
        latent_dim = len(candidates[0].turn_result.right_state.latent_vector)
        preferred_mu = np.zeros(latent_dim)
        preferred_logvar = np.ones(latent_dim) * -5.0
        for candidate in candidates:
            res = candidate.turn_result
            right = res.right_state
            mu = np.array(right.latent_mu) if right.latent_mu else np.array(right.latent_vector)
            logvar = np.array(right.latent_logvar) if right.latent_logvar else np.ones_like(mu) * -2.0
            ambiguity = 1.0 - right.confidence
            risk = kl_divergence_gaussian(mu, logvar, preferred_mu, preferred_logvar)
            efe = calculate_efe(mu, logvar, preferred_mu, preferred_logvar, ambiguity)
            score = 1.0 / (1.0 + efe)
            scoreboard.append(
                ReflectionScore(
                    variant_id=candidate.variant.variant_id,
                    score=float(score),
                    reasons=[
                        f"EFE={efe:.4f}",
                        f"risk={risk:.4f}",
                        f"ambiguity={ambiguity:.4f}",
                        f"confidence={right.confidence:.2f}",
                    ],
                )
            )

        scoreboard.sort(key=lambda x: x.score, reverse=True)
        winner_id = scoreboard[0].variant_id
        return ReflectionOutcome(
            selected_variant_id=winner_id,
            scoreboard=scoreboard,
            selected_by="efe_minimization_loop",
            notes=[f"evaluated={len(candidates)}"],
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


CognitiveVariantSelector = GEAReflectionController
