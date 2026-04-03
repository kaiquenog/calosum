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
    contract_version: str = "group_turn.v1"

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)

    def as_agent_turn_result(self) -> AgentTurnResult:
        return self.selected_result


def default_cognitive_personas(max_width: int = 1) -> list[CognitiveVariantSpec]:
    personas = [
        CognitiveVariantSpec(
            variant_id="base",
            notes=["linear_flow"],
        ),
        CognitiveVariantSpec(
            variant_id="analitico",
            bridge_directives=["prefer explicit decomposition before action"],
            notes=["deep_verification"],
        ),
        CognitiveVariantSpec(
            variant_id="empatico",
            bridge_directives=["lead with empathy before dense logic"],
            notes=["affective_alignment"],
        ),
        CognitiveVariantSpec(
            variant_id="pragmatico",
            bridge_directives=["optimize for shortest safe path to execution"],
            notes=["fast_resolution"],
        ),
    ]
    return personas[: max(1, max_width)]


class GEAReflectionController:
    """
    Group Evolution of Agents (GEA) reflection controller.
    Uses Expected Free Energy (EFE) to select the best cognitive variant.
    """

    def __init__(self, adaptation_step: float = 0.05, *args: Any, **kwargs: Any) -> None:
        self.adaptation_step = max(0.0, min(0.5, float(adaptation_step)))

    def evaluate(
        self,
        candidates: list[CognitiveCandidate],
        base_tokenizer: Any,
    ) -> ReflectionOutcome:
        import numpy as np

        if not candidates:
            raise ValueError("Reflection requires at least one candidate")
        scoreboard = []
        metrics_by_variant: dict[str, dict[str, float]] = {}
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
            complexity_penalty = max(0.0, risk)
            ambiguity_cost = max(0.0, ambiguity)
            tool_success_rate = self._tool_success_rate(res)
            retry_penalty = min(1.0, 0.1 * float(res.runtime_retry_count + res.critique_revision_count))
            social_bonus = self._social_bonus(candidate.variant.variant_id, right)
            score = max(0.0, (1.0 / (1.0 + efe)) + (0.25 * tool_success_rate) + social_bonus - retry_penalty)
            metrics_by_variant[candidate.variant.variant_id] = {
                "complexity_penalty": float(complexity_penalty),
                "ambiguity_cost": float(ambiguity_cost),
                "tool_success_rate": tool_success_rate,
                "social_bonus": social_bonus,
                "runtime_retry_count": float(res.runtime_retry_count),
            }
            scoreboard.append(
                ReflectionScore(
                    variant_id=candidate.variant.variant_id,
                    score=float(score),
                    reasons=[
                        f"EFE={efe:.4f}",
                        f"complexity_penalty={complexity_penalty:.4f}",
                        f"ambiguity_cost={ambiguity_cost:.4f}",
                        f"confidence={right.confidence:.2f}",
                        f"tool_success_rate={tool_success_rate:.2f}",
                        f"social_bonus={social_bonus:.3f}",
                    ],
                )
            )

        scoreboard.sort(key=lambda x: x.score, reverse=True)
        winner_id = scoreboard[0].variant_id
        winner_metrics = metrics_by_variant.get(winner_id, {})
        return ReflectionOutcome(
            selected_variant_id=winner_id,
            scoreboard=scoreboard,
            selected_by="efe_minimization_loop",
            selected_metrics=self._selected_metrics(
                next(candidate for candidate in candidates if candidate.variant.variant_id == winner_id),
                float(scoreboard[0].score),
            ),
            cost_metrics={
                "objective": "expected_free_energy",
                "candidate_count": len(candidates),
                "branch_count": len(candidates),
            },
            notes=[f"evaluated={len(candidates)}"],
        )

    async def aevaluate(
        self,
        candidates: list[CognitiveCandidate],
        base_tokenizer: Any,
    ) -> ReflectionOutcome:
        return self.evaluate(candidates, base_tokenizer)

    def apply_config_adaptation(self, *args, **kwargs):
        tokenizer = args[0] if args else None
        outcome = args[1] if len(args) > 1 else None
        if tokenizer is None or outcome is None or not hasattr(tokenizer, "config"):
            return
        adjustments = getattr(outcome, "bridge_adjustments", {}) or {}
        for key, value in adjustments.items():
            if hasattr(tokenizer.config, key):
                setattr(tokenizer.config, key, value)
        if hasattr(tokenizer, "persist_adaptation_state"):
            tokenizer.persist_adaptation_state()

    def apply_neuroplasticity(self, *args, **kwargs):
        tokenizer = args[0] if args else None
        outcome = args[1] if len(args) > 1 else None
        if tokenizer is None or outcome is None or not hasattr(tokenizer, "config"):
            return
        selected_by = getattr(outcome, "selected_by", "")
        current = float(getattr(tokenizer.config, "salience_gain", 1.0))
        if selected_by == "learned_model":
            tokenizer.config.salience_gain = min(1.5, current + self.adaptation_step)
        elif selected_by == "rule_based":
            tokenizer.config.salience_gain = max(0.8, current - (self.adaptation_step / 2.0))

    def _infer_context_type(self, candidates: list[CognitiveCandidate]) -> str:
        if not candidates:
            return "unknown"
        text = candidates[0].turn_result.user_turn.user_text.lower()
        if any(marker in text for marker in ("ansioso", "triste", "emocional", "medo", "frustr")):
            return "emotional"
        if any(marker in text for marker in ("benchmark", "compare", "comparar", "tradeoff", "arquitetura")):
            return "analytical"
        if any(marker in text for marker in ("passos", "curto", "objetivo", "resuma", "checklist")):
            return "pragmatic"
        return "general"

    def _compute_reward(self, candidate: CognitiveCandidate, score: float) -> float:
        tool_success_rate = self._tool_success_rate(candidate.turn_result)
        retry_penalty = 0.05 * float(candidate.turn_result.runtime_retry_count + candidate.turn_result.critique_revision_count)
        return round(max(0.0, min(1.0, (0.6 * score) + (0.4 * tool_success_rate) - retry_penalty)), 4)

    def _selected_metrics(self, candidate: CognitiveCandidate, score: float) -> dict[str, Any]:
        right = candidate.turn_result.right_state
        return {
            "score": round(float(score), 6),
            "selection_objective": "min_efe_plus_quality",
            "tool_success_rate": round(self._tool_success_rate(candidate.turn_result), 4),
            "runtime_retry_count": int(candidate.turn_result.runtime_retry_count),
            "critique_revision_count": int(candidate.turn_result.critique_revision_count),
            "surprise_score": round(float(right.surprise_score), 4),
            "confidence": round(float(right.confidence), 4),
            "peer_latents_count": int(right.telemetry.get("peer_latents_count", 0)),
        }

    def _propose_bridge_adjustments(self, candidate: CognitiveCandidate, base_tokenizer: Any) -> dict[str, Any]:
        right = candidate.turn_result.right_state
        current_threshold = float(getattr(getattr(base_tokenizer, "config", object()), "salience_threshold", 0.7))
        if right.surprise_score >= 0.75:
            return {"salience_threshold": round(max(0.45, current_threshold - self.adaptation_step), 3)}
        if right.confidence >= 0.85 and candidate.variant.variant_id == "pragmatico":
            return {"salience_threshold": round(min(0.9, current_threshold + self.adaptation_step), 3)}
        return {}

    def _social_bonus(self, variant_id: str, right_state: InputPerceptionState) -> float:
        peer_count = float(right_state.telemetry.get("peer_latents_count", 0))
        peer_alignment = float(right_state.telemetry.get("peer_latent_alignment", 0.0))
        if peer_count <= 0:
            return 0.0
        normalized_peer_factor = min(0.08, (peer_count * 0.01) + (0.04 * max(0.0, peer_alignment)))
        if variant_id == "empatico" and right_state.emotional_labels:
            return normalized_peer_factor
        if variant_id == "analitico" and right_state.world_hypotheses.get("interaction_complexity", 0.0) >= 0.6:
            return normalized_peer_factor / 2.0
        if variant_id == "pragmatico":
            return min(0.03, normalized_peer_factor / 3.0)
        return 0.0

    def _tool_success_rate(self, result: AgentTurnResult) -> float:
        telemetry = getattr(result, "telemetry", None)
        if isinstance(telemetry, dict):
            decision = telemetry.get("decision", {})
            if isinstance(decision, dict):
                return float(decision.get("tool_success_rate", 1.0))
            return 1.0
        decision = getattr(telemetry, "decision", {})
        if isinstance(decision, dict):
            return float(decision.get("tool_success_rate", 1.0))
        return 1.0


CognitiveVariantSelector = GEAReflectionController
