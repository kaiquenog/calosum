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
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GroupTurnResult:
    user_turn: UserTurn
    right_state: RightHemisphereState
    candidates: list[CognitiveCandidate]
    selected_result: AgentTurnResult
    reflection: ReflectionOutcome


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
        for candidate in candidates:
            score, reasons = self._score_candidate(candidate)
            scoreboard.append(
                ReflectionScore(
                    variant_id=candidate.variant.variant_id,
                    score=round(score, 3),
                    reasons=reasons,
                )
            )

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

        return ReflectionOutcome(
            selected_variant_id=winner.variant_id,
            scoreboard=scoreboard,
            bridge_adjustments=bridge_adjustments,
            notes=notes,
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
        if "salience_threshold" in overrides:
            return {"salience_threshold": overrides["salience_threshold"]}

        current_threshold = base_tokenizer.config.salience_threshold
        if selected_candidate.turn_result.bridge_packet.control.empathy_priority:
            proposed_threshold = max(0.1, round(current_threshold - self.adaptation_step, 2))
        else:
            proposed_threshold = min(0.95, round(current_threshold + self.adaptation_step, 2))

        return {"salience_threshold": proposed_threshold}
