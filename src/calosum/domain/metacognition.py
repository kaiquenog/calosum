from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from calosum.domain.bridge import CognitiveTokenizer
from calosum.shared.types import AgentTurnResult, RightHemisphereState, UserTurn


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
    scoreboard: list[ReflectionScore]
    bridge_adjustments: dict[str, Any] = field(default_factory=dict)
    selected_metrics: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    pruning_reasons: dict[str, str] = field(default_factory=dict)
    cost_metrics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GroupTurnResult:
    user_turn: UserTurn
    right_state: RightHemisphereState
    candidates: list[CognitiveCandidate]
    selected_result: AgentTurnResult
    reflection: ReflectionOutcome


def default_cognitive_personas(max_width: int = 3) -> list[CognitiveVariantSpec]:
    personas = [
        CognitiveVariantSpec(
            variant_id="analitico",
            tokenizer_overrides={"base_temperature": 0.18, "salience_threshold": 0.78},
            bridge_directives=[
                "priorize consistencia logica e verificabilidade",
                "identifique contradicoes antes de agir",
            ],
            notes=["logic_first", "low_temperature"],
        ),
        CognitiveVariantSpec(
            variant_id="empatico",
            tokenizer_overrides={"base_temperature": 0.34, "salience_threshold": 0.45},
            bridge_directives=[
                "considere impacto emocional antes da solucao",
                "valide o contexto afetivo com linguagem segura",
            ],
            notes=["emotion_first", "supportive"],
        ),
        CognitiveVariantSpec(
            variant_id="pragmatico",
            tokenizer_overrides={"base_temperature": 0.22, "max_directives": 3},
            bridge_directives=[
                "minimize a fronteira de acoes",
                "entregue a menor resposta segura que resolva a solicitacao",
            ],
            notes=["minimal_actions", "concise"],
        ),
    ]
    return personas[:max_width]


class GEAReflectionController:
    """
    Juiz metacognitivo inspirado em Group-Evolving Agents.

    A funcao dele aqui e:
    - comparar variantes cognitivas competidoras;
    - escolher a melhor sintese entre empatia, seguranca e simplicidade;
    - devolver pequenos ajustes para o "corpo caloso".
    """

    def __init__(self, adaptation_step: float = 0.05) -> None:
        self.adaptation_step = adaptation_step

    def evaluate(
        self,
        candidates: list[CognitiveCandidate],
        base_tokenizer: CognitiveTokenizer,
    ) -> ReflectionOutcome:
        if not candidates:
            raise ValueError("GEA reflection requires at least one candidate")

        scoreboard: list[ReflectionScore] = []
        pruning_reasons: dict[str, str] = {}
        for candidate in candidates:
            score, reasons = self._score_candidate(candidate)
            scoreboard.append(
                ReflectionScore(
                    variant_id=candidate.variant.variant_id,
                    score=round(score, 3),
                    reasons=reasons,
                )
            )
            if score < 1.0:
                pruning_reasons[candidate.variant.variant_id] = "Low score due to: " + ", ".join(reasons)

        winner = max(scoreboard, key=lambda item: item.score)
        selected_candidate = next(
            candidate
            for candidate in candidates
            if candidate.variant.variant_id == winner.variant_id
        )

        bridge_adjustments = self._propose_bridge_adjustments(
            selected_candidate,
            base_tokenizer,
        )
        notes = [
            "winner chosen by combined empathy, runtime safety and action simplicity",
            f"selected_variant={winner.variant_id}",
        ]
        
        cost_metrics = {
            "branch_count": len(candidates),
            "variants_evaluated": len(candidates),
            "total_latency_ms": sum(c.turn_result.latency_ms for c in candidates)
        }

        return ReflectionOutcome(
            selected_variant_id=winner.variant_id,
            scoreboard=scoreboard,
            bridge_adjustments=bridge_adjustments,
            selected_metrics=self._selected_metrics(selected_candidate, winner.score),
            notes=notes,
            pruning_reasons=pruning_reasons,
            cost_metrics=cost_metrics,
        )

    async def aevaluate(
        self,
        candidates: list[CognitiveCandidate],
        base_tokenizer: CognitiveTokenizer,
    ) -> ReflectionOutcome:
        return self.evaluate(candidates, base_tokenizer)

    def apply_neuroplasticity(
        self,
        tokenizer: CognitiveTokenizer,
        outcome: ReflectionOutcome,
    ) -> None:
        for key, value in outcome.bridge_adjustments.items():
            if hasattr(tokenizer.config, key):
                setattr(tokenizer.config, key, value)
        if hasattr(tokenizer, "record_reflection_event"):
            tokenizer.record_reflection_event(outcome.as_dict())
        if hasattr(tokenizer, "persist_adaptation_state"):
            tokenizer.persist_adaptation_state()

    def _score_candidate(self, candidate: CognitiveCandidate) -> tuple[float, list[str]]:
        turn_result = candidate.turn_result
        bridge = turn_result.bridge_packet
        score = 1.0
        reasons: list[str] = []

        if bridge.control.empathy_priority:
            score += 0.35
            reasons.append("empathy priority activated")
        else:
            score += 0.1
            reasons.append("logic-first configuration retained")

        if "seguro" in turn_result.left_result.response_text.lower():
            score += 0.2
            reasons.append("safe-response language present")

        rejected = sum(
            1 for item in turn_result.execution_results if item.status == "rejected"
        )
        if rejected == 0:
            score += 0.35
            reasons.append("strict runtime accepted all actions")
        else:
            score -= rejected * 0.4
            reasons.append(f"runtime rejected {rejected} action(s)")

        if len(turn_result.left_result.actions) <= 2:
            score += 0.1
            reasons.append("compact action frontier")

        if turn_result.memory_context.semantic_rules:
            score += 0.15
            reasons.append("semantic memory reused")

        if turn_result.runtime_retry_count == 0:
            score += 0.1
            reasons.append("no runtime retries needed")
        else:
            score -= turn_result.runtime_retry_count * 0.15
            reasons.append(f"runtime retries={turn_result.runtime_retry_count}")

        salience_gap = abs(
            bridge.salience - turn_result.bridge_packet.control.annotations["salience_threshold"]
        )
        score += max(0.0, 0.15 - salience_gap * 0.1)
        reasons.append(f"salience_gap={salience_gap:.2f}")

        return score, reasons

    def _propose_bridge_adjustments(
        self,
        selected_candidate: CognitiveCandidate,
        base_tokenizer: CognitiveTokenizer,
    ) -> dict[str, Any]:
        overrides = dict(selected_candidate.variant.tokenizer_overrides)
        direct_overrides = {
            key: value
            for key, value in overrides.items()
            if key in {
                "salience_threshold",
                "base_temperature",
                "max_directives",
                "bottleneck_tokens",
                "salience_gain",
                "salience_bias",
                "temperature_bias",
            }
        }
        if direct_overrides:
            return direct_overrides

        current_threshold = base_tokenizer.config.salience_threshold
        current_temperature = base_tokenizer.config.base_temperature
        current_directives = base_tokenizer.config.max_directives
        current_tokens = base_tokenizer.config.bottleneck_tokens
        current_salience_gain = getattr(base_tokenizer.config, "salience_gain", 1.0)
        current_salience_bias = getattr(base_tokenizer.config, "salience_bias", 0.0)
        current_temperature_bias = getattr(base_tokenizer.config, "temperature_bias", 0.0)

        turn_result = selected_candidate.turn_result
        empathy_priority = turn_result.bridge_packet.control.empathy_priority
        rejected_count = sum(1 for item in turn_result.execution_results if item.status == "rejected")
        emotional_bandwidth = len(turn_result.right_state.emotional_labels)

        if empathy_priority:
            proposed_threshold = max(0.1, round(current_threshold - self.adaptation_step, 2))
            proposed_tokens = min(8, current_tokens + (1 if emotional_bandwidth >= 2 else 0))
            proposed_temperature = min(0.45, round(current_temperature + 0.02, 2))
            proposed_directives = min(6, current_directives + 1)
            proposed_salience_gain = min(1.5, round(current_salience_gain + 0.05, 2))
            proposed_salience_bias = min(0.25, round(current_salience_bias + 0.02, 2))
            proposed_temperature_bias = min(0.2, round(current_temperature_bias + 0.01, 2))
        else:
            proposed_threshold = min(0.95, round(current_threshold + self.adaptation_step, 2))
            proposed_tokens = max(4, current_tokens - (1 if emotional_bandwidth <= 1 else 0))
            proposed_temperature = max(0.1, round(current_temperature - 0.01, 2))
            proposed_directives = max(2, current_directives - 1)
            proposed_salience_gain = max(0.6, round(current_salience_gain - 0.03, 2))
            proposed_salience_bias = max(-0.25, round(current_salience_bias - 0.01, 2))
            proposed_temperature_bias = max(-0.15, round(current_temperature_bias - 0.01, 2))

        if rejected_count or turn_result.runtime_retry_count:
            proposed_temperature = max(0.1, round(current_temperature - self.adaptation_step, 2))
            proposed_directives = max(2, current_directives - 1)
            proposed_salience_gain = max(0.6, round(current_salience_gain - 0.05, 2))
            proposed_temperature_bias = max(-0.15, round(current_temperature_bias - 0.03, 2))

        return {
            "salience_threshold": proposed_threshold,
            "base_temperature": proposed_temperature,
            "max_directives": proposed_directives,
            "bottleneck_tokens": proposed_tokens,
            "salience_gain": proposed_salience_gain,
            "salience_bias": proposed_salience_bias,
            "temperature_bias": proposed_temperature_bias,
        }

    def _selected_metrics(
        self,
        selected_candidate: CognitiveCandidate,
        winner_score: float,
    ) -> dict[str, Any]:
        turn_result = selected_candidate.turn_result
        rejected_count = sum(1 for item in turn_result.execution_results if item.status == "rejected")
        return {
            "score": round(winner_score, 3),
            "empathy_priority": turn_result.bridge_packet.control.empathy_priority,
            "runtime_retry_count": turn_result.runtime_retry_count,
            "runtime_rejected_count": rejected_count,
            "semantic_rules": len(turn_result.memory_context.semantic_rules),
            "action_count": len(turn_result.left_result.actions),
            "calibrated_salience": turn_result.bridge_packet.salience,
        }
