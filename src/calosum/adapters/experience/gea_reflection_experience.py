from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from calosum.adapters.experience.variant_preference import (
    PreferenceFeatures,
    PreferenceTrainingReport,
    VariantPreferenceDatasetStore,
    VariantPreferenceModel,
    VariantTrainingExample,
    canonical_variant_id,
)
from calosum.domain.cognition.bridge import CognitiveTokenizer
from calosum.domain.metacognition.metacognition import (
    CognitiveCandidate,
    GEAReflectionController,
    ReflectionOutcome,
    ReflectionScore,
)


class LearnedPreferenceGEAReflectionController(GEAReflectionController):
    """GEA reflection controller with learned variant preference and graceful fallback."""

    def __init__(
        self,
        *,
        adaptation_step: float = 0.05,
        dataset_path: Path | None = None,
        model_path: Path | None = None,
        min_training_samples: int = 200,
        min_holdout_accuracy: float = 0.65,
    ) -> None:
        super().__init__(adaptation_step=adaptation_step)
        runtime_root = Path(".calosum-runtime") / "reflection"
        self.dataset_store = VariantPreferenceDatasetStore(dataset_path or (runtime_root / "group_turn_dataset.jsonl"))
        self.preference_model = VariantPreferenceModel(
            artifact_path=model_path or (runtime_root / "variant_preference_model.joblib"),
            min_samples=min_training_samples,
        )
        self.min_training_samples = max(1, int(min_training_samples))
        self.min_holdout_accuracy = max(0.0, min(1.0, float(min_holdout_accuracy)))

    def evaluate(
        self,
        candidates: list[CognitiveCandidate],
        base_tokenizer: CognitiveTokenizer,
    ) -> ReflectionOutcome:
        outcome = super().evaluate(candidates, base_tokenizer)
        context_type = self._infer_context_type(candidates)
        features = self._build_features(candidates, context_type)
        available_by_canonical = self._available_variant_ids(candidates)

        selected_variant_id = outcome.selected_variant_id
        selected_by = "legacy"
        selection_notes: list[str] = []
        training_report: PreferenceTrainingReport | None = None

        learned_metadata = self.preference_model.metadata()
        learned_ready = self._is_learned_model_ready(learned_metadata)
        if learned_ready:
            predicted, info = self.preference_model.predict(features)
            predicted_variant_id = self._resolve_variant_id(predicted, available_by_canonical)
            if predicted_variant_id is not None:
                selected_variant_id = predicted_variant_id
                selected_by = "learned_model"
                holdout = float(learned_metadata.get("holdout_accuracy", 0.0))
                selection_notes.append(f"learned_model_holdout_accuracy={holdout:.3f}")
            else:
                selection_notes.append(f"learned_model_unmapped:{info.get('reason', 'unknown')}")

        emotional_intensity = self._emotional_intensity(candidates)
        short_response_expected = self._expects_short_response(candidates)
        if selected_by != "learned_model":
            rule_selected = self._rule_based_choice(
                features,
                available_by_canonical,
                emotional_intensity=emotional_intensity,
                short_response_expected=short_response_expected,
            )
            if rule_selected is not None:
                selected_variant_id = rule_selected
                selected_by = "rule_based"
            else:
                selected_variant_id = self._legacy_weighted_choice(
                    outcome.scoreboard,
                    available_by_canonical,
                    default_variant=outcome.selected_variant_id,
                )
                selected_by = "legacy"

        selected_candidate = next(
            candidate
            for candidate in candidates
            if candidate.variant.variant_id == selected_variant_id
        )
        selected_score = next(
            (item.score for item in outcome.scoreboard if item.variant_id == selected_variant_id),
            outcome.selected_metrics.get("score", 1.0),
        )

        response_rating = self._compute_reward(selected_candidate, float(selected_score))
        self.dataset_store.append(
            VariantTrainingExample(
                session_id=selected_candidate.turn_result.user_turn.session_id,
                turn_id=selected_candidate.turn_result.user_turn.turn_id,
                recorded_at=selected_candidate.turn_result.user_turn.observed_at.isoformat(),
                variant_scores={item.variant_id: float(item.score) for item in outcome.scoreboard},
                selected_variant=selected_variant_id,
                response_rating=float(response_rating),
                context={
                    "intent_type": context_type,
                    "surprise_score": float(features.surprise_score),
                    "ambiguity_score": float(features.ambiguity_score),
                    "session_length": int(features.session_length),
                    "avg_tool_success_rate": float(features.avg_tool_success_rate),
                    "jepa_uncertainty": float(features.jepa_uncertainty),
                    "short_response_expected": bool(short_response_expected),
                    "emotional_intensity": float(emotional_intensity),
                },
            )
        )

        dataset_count = self.dataset_store.count()
        if dataset_count >= self.min_training_samples and not learned_ready:
            training_report = self.preference_model.train(self.dataset_store.read_all())

        bridge_adjustments = self._propose_bridge_adjustments(selected_candidate, base_tokenizer)
        return replace(
            outcome,
            selected_variant_id=selected_variant_id,
            selected_metrics=self._selected_metrics(selected_candidate, float(selected_score)),
            bridge_adjustments=bridge_adjustments,
            selected_by=selected_by,
            notes=list(outcome.notes)
            + selection_notes
            + [
                f"selection_method={selected_by}",
                f"dataset_size={dataset_count}",
            ]
            + self._training_notes(training_report),
            cost_metrics={
                **outcome.cost_metrics,
                "selector_dataset_size": dataset_count,
                "selector_model_accuracy": float(learned_metadata.get("holdout_accuracy", 0.0)),
                "selected_by": selected_by,
            },
        )

    def _build_features(self, candidates: list[CognitiveCandidate], context_type: str) -> PreferenceFeatures:
        sample = candidates[0].turn_result
        right = sample.right_state
        avg_tool_success = 1.0
        if candidates:
            values = [
                float(candidate.turn_result.telemetry.decision.get("tool_success_rate", 1.0))
                for candidate in candidates
            ]
            avg_tool_success = sum(values) / len(values)

        return PreferenceFeatures(
            surprise_score=float(getattr(right, "surprise_score", 0.0)),
            ambiguity_score=float(right.world_hypotheses.get("interaction_complexity", 0.0)),
            intent_type=context_type,
            session_length=max(1, len(sample.memory_context.recent_episodes) + 1),
            avg_tool_success_rate=float(avg_tool_success),
            jepa_uncertainty=float(right.telemetry.get("jepa_uncertainty", 0.0)),
        )

    def _available_variant_ids(self, candidates: list[CognitiveCandidate]) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for candidate in candidates:
            canonical = canonical_variant_id(candidate.variant.variant_id)
            if canonical not in mapping:
                mapping[canonical] = candidate.variant.variant_id
        return mapping

    def _resolve_variant_id(self, canonical_variant: str | None, available: dict[str, str]) -> str | None:
        if not canonical_variant:
            return None
        return available.get(canonical_variant_id(canonical_variant))

    def _is_learned_model_ready(self, metadata: dict[str, Any]) -> bool:
        sample_count = int(metadata.get("sample_count", 0))
        holdout_accuracy = float(metadata.get("holdout_accuracy", 0.0))
        return sample_count >= self.min_training_samples and holdout_accuracy >= self.min_holdout_accuracy

    def _rule_based_choice(
        self,
        features: PreferenceFeatures,
        available_variants: dict[str, str],
        *,
        emotional_intensity: float,
        short_response_expected: bool,
    ) -> str | None:
        if emotional_intensity >= 0.7 or (
            features.intent_type == "emotional" and emotional_intensity >= 0.45
        ):
            return available_variants.get("empatico")
        if features.avg_tool_success_rate <= 0.6:
            return available_variants.get("analitico")
        if short_response_expected:
            return available_variants.get("pragmatico")
        return None

    def _legacy_weighted_choice(
        self,
        scoreboard: list[ReflectionScore],
        available_variants: dict[str, str],
        *,
        default_variant: str,
    ) -> str:
        legacy_weights = {
            "analitico": 0.5,
            "empatico": 0.3,
            "pragmatico": 0.2,
        }
        score_by_variant = {item.variant_id: float(item.score) for item in scoreboard}
        ranked: list[tuple[float, str]] = []
        for canonical, weight in legacy_weights.items():
            variant_id = available_variants.get(canonical)
            if variant_id is None:
                continue
            ranked.append((weight + (0.05 * score_by_variant.get(variant_id, 0.0)), variant_id))
        if not ranked:
            return default_variant
        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked[0][1]

    def _expects_short_response(self, candidates: list[CognitiveCandidate]) -> bool:
        if not candidates:
            return False
        text = candidates[0].turn_result.user_turn.user_text.lower()
        markers = ["resuma", "curta", "objetivo", "rapido", "passos", "checklist"]
        return any(marker in text for marker in markers)

    def _emotional_intensity(self, candidates: list[CognitiveCandidate]) -> float:
        if not candidates:
            return 0.0
        right = candidates[0].turn_result.right_state
        label_count = len(getattr(right, "emotional_labels", []))
        surprise = float(getattr(right, "surprise_score", 0.0))
        return max(0.0, min(1.0, 0.2 * label_count + 0.6 * surprise))

    def _training_notes(self, report: PreferenceTrainingReport | None) -> list[str]:
        if report is None:
            return []
        if not report.trained:
            return [f"selector_training_skipped:{report.reason or 'unknown'}"]
        return [f"selector_trained_holdout_accuracy={report.holdout_accuracy:.3f}"]


class ExperienceAwareGEAReflectionController(LearnedPreferenceGEAReflectionController):
    """GEA reflection controller enhanced with persistent experience priors."""

    def __init__(
        self,
        experience_store: Any,
        adaptation_step: float = 0.05,
        prior_weight: float = 0.2,
        dataset_path: Path | None = None,
        model_path: Path | None = None,
    ) -> None:
        super().__init__(
            adaptation_step=adaptation_step,
            dataset_path=dataset_path,
            model_path=model_path,
        )
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

        winner = next(item for item in adjusted if item.variant_id == outcome.selected_variant_id)
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
                    "selected_by": outcome.selected_by,
                    "reasons": item.reasons,
                },
            )

        return replace(
            outcome,
            scoreboard=adjusted,
            selected_metrics=self._selected_metrics(selected, winner.score),
            notes=list(outcome.notes) + [f"experience_context={context_type}"],
            cost_metrics={
                **outcome.cost_metrics,
                "experience_store_enabled": True,
            },
        )
