from __future__ import annotations

from dataclasses import replace
from typing import Any

from calosum.domain.cognition.bridge import CognitiveTokenizer
from calosum.domain.metacognition.metacognition import (
    CognitiveCandidate,
    GEAReflectionController,
    ReflectionOutcome,
    ReflectionScore,
)


class ExperienceAwareGEAReflectionController(GEAReflectionController):
    """GEA reflection controller enhanced with persistent experience priors."""

    def __init__(self, experience_store: Any, adaptation_step: float = 0.05, prior_weight: float = 0.2) -> None:
        super().__init__(adaptation_step=adaptation_step)
        self.experience_store = experience_store
        self.prior_weight = prior_weight

    def evaluate(
        self,
        candidates: list[CognitiveCandidate],
        base_tokenizer: CognitiveTokenizer,
    ) -> ReflectionOutcome:
        outcome = super().evaluate(candidates, base_tokenizer)
        context_type = self._infer_context_type(candidates)

        adjusted: list[ReflectionScore] = []
        for score in outcome.scoreboard:
            prior = float(
                self.experience_store.variant_prior(
                    context_type=context_type,
                    variant_id=score.variant_id,
                    limit=100,
                )
            )
            corrected = round(score.score + (self.prior_weight * prior), 3)
            adjusted.append(
                ReflectionScore(
                    variant_id=score.variant_id,
                    score=corrected,
                    reasons=list(score.reasons) + [f"experience_prior={prior:.3f}"],
                )
            )

        winner = max(adjusted, key=lambda item: item.score)
        selected = next(c for c in candidates if c.variant.variant_id == winner.variant_id)
        reward = self._compute_reward(selected, winner.score)

        for item in adjusted:
            self.experience_store.record_experience(
                context_type=context_type,
                variant_id=item.variant_id,
                score=item.score,
                reward=reward if item.variant_id == winner.variant_id else 0.0,
                metadata={
                    "selected": item.variant_id == winner.variant_id,
                    "reasons": item.reasons,
                },
            )

        bridge_adjustments = self._propose_bridge_adjustments(selected, base_tokenizer)
        return replace(
            outcome,
            selected_variant_id=winner.variant_id,
            scoreboard=adjusted,
            bridge_adjustments=bridge_adjustments,
            selected_metrics=self._selected_metrics(selected, winner.score),
            notes=list(outcome.notes) + [f"experience_context={context_type}"],
            cost_metrics={
                **outcome.cost_metrics,
                "experience_store_enabled": True,
            },
        )
