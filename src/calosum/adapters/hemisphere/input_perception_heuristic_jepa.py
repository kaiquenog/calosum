from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from calosum.shared.models.jepa import ContextEmbedding, ResponsePrediction, SurpriseScore
from calosum.shared.models.types import CognitiveWorkspace, MemoryContext, InputPerceptionState, UserTurn
from calosum.adapters.memory.text_embeddings import TextEmbeddingAdapter, TextEmbeddingAdapterConfig


@dataclass(slots=True)
class HeuristicJEPAConfig:
    embedding_dim: int = 384
    prediction_method: str = "jepa_literal_embedding"
    uncertainty_ignore_threshold: float = 0.7
    embedding_backend: str = "lexical"  # OpenAI, HuggingFace or lexical
    salience_keywords: dict[str, float] = field(
        default_factory=lambda: {
            "urgente": 1.0,
            "emergencia": 1.0,
            "triste": 0.85,
            "ansioso": 0.75,
            "feliz": 0.35,
            "frustrado": 0.8,
            "raiva": 0.9,
            "medo": 0.9,
            "preocupado": 0.8,
            "dor": 0.9,
            "desespero": 0.95,
        }
    )


class HeuristicJEPAAdapter:
    """
    JEPA textual fase 1: predicao de embedding de resposta por media ponderada.

    - Abandona o hash artesanal em favor de Literal Embeddings (via TextEmbeddingAdapter).
    - Predicao por media ponderada por recencia.
    - Surprise por erro preditivo real: distancia entre embedding previsto e observado.
    """

    def __init__(self, config: HeuristicJEPAConfig | None = None) -> None:
        self.config = config or HeuristicJEPAConfig()
        self.embedder = TextEmbeddingAdapter(
            TextEmbeddingAdapterConfig(
                provider=self.config.embedding_backend,
                vector_size=self.config.embedding_dim,
            )
        )

    def perceive(
        self,
        user_turn: UserTurn,
        memory_context: MemoryContext | None = None,
        workspace: CognitiveWorkspace | None = None,
    ) -> InputPerceptionState:
        context_turns = self._history_turns(memory_context)
        if not context_turns:
            context_turns = [user_turn]
        
        # O encode de contexto agora é literal
        context = self._encode_context(context_turns)
        prediction = self._predict_response_embedding(context)
        surprise = self._compute_surprise(context, user_turn.user_text)
        runtime_feedback_bias = self._runtime_feedback_bias(workspace)

        emotional_labels = self._extract_emotional_labels(user_turn.user_text)
        raw_salience = self._estimate_salience(user_turn.user_text, emotional_labels)
        salience = round(min(1.0, raw_salience + runtime_feedback_bias), 3)
        confidence = round(max(0.0, min(1.0, 1.0 - prediction.uncertainty)), 3)
        surprise_band = self._surprise_band(surprise.score)
        ignored = surprise.ignored_due_to_uncertainty

        telemetry = {
            "model_name": "heuristic-jepa-phase1",
            "right_backend": f"heuristic_jepa_literal_{self.embedder.backend_name()}",
            "right_model_name": "heuristic-jepa-phase1",
            "right_mode": "literal_embedding",
            "degraded_reason": None,
            "prediction_method": prediction.prediction_method,
            "jepa_uncertainty": prediction.uncertainty,
            "prediction_error": surprise.prediction_error,
            "surprise_source": surprise.source,
            "surprise_band": surprise_band,
            "ignore_surprise_for_branching": ignored,
            "raw_salience": raw_salience,
            "runtime_feedback_bias": runtime_feedback_bias,
            "context_turns_used": len(context_turns),
        }
        world_hypotheses = {
            "interaction_complexity": min(1.0, len(user_turn.user_text) / 240.0),
            "urgency": salience,
            "semantic_density": self._semantic_density(prediction.predicted_embedding),
            "prediction_uncertainty": prediction.uncertainty,
            "operational_risk": runtime_feedback_bias,
        }
        state = InputPerceptionState(
            context_id=user_turn.turn_id,
            latent_vector=prediction.predicted_embedding,
            salience=salience,
            emotional_labels=emotional_labels or ["neutral"],
            world_hypotheses=world_hypotheses,
            confidence=confidence,
            surprise_score=surprise.score,
            telemetry=telemetry,
        )
        if workspace is not None:
            workspace.right_notes.update(
                {
                    "backend": "heuristic_jepa_literal_embedding",
                    "surprise_score": state.surprise_score,
                    "surprise_band": surprise_band,
                    "prediction_method": prediction.prediction_method,
                    "surprise_source": surprise.source,
                    "uncertainty": prediction.uncertainty,
                    "ignore_surprise_for_branching": ignored,
                    "runtime_feedback_bias": runtime_feedback_bias,
                }
            )
        return state

    async def aperceive(
        self,
        user_turn: UserTurn,
        memory_context: MemoryContext | None = None,
        workspace: CognitiveWorkspace | None = None,
    ) -> InputPerceptionState:
        return self.perceive(user_turn, memory_context, workspace)

    async def encode_context(self, turns: list[UserTurn]) -> ContextEmbedding:
        return self._encode_context(turns)

    async def predict_response_embedding(self, ctx: ContextEmbedding) -> ResponsePrediction:
        return self._predict_response_embedding(ctx)

    async def compute_surprise(
        self,
        ctx: ContextEmbedding,
        actual_response: str,
    ) -> SurpriseScore:
        return self._compute_surprise(ctx, actual_response)

    def score_candidates(self, ctx: ContextEmbedding, candidates: list[str]) -> list[tuple[str, float]]:
        prediction = self._predict_response_embedding(ctx)
        context_terms = set(ctx.context_terms)
        scored = []
        for candidate in candidates:
            candidate_terms = set(self._tokenize(candidate))
            overlap = 0.0
            if candidate_terms and context_terms:
                overlap = len(candidate_terms.intersection(context_terms)) / len(candidate_terms)
            
            # Encode via embedder
            candidate_vector = self.embedder.embed_texts([candidate])[0]
            cosine = self._cosine_similarity(
                prediction.predicted_embedding,
                candidate_vector,
            )
            cosine01 = (cosine + 1.0) / 2.0
            negation_penalty = 0.0
            if {"nao", "ignore", "evite"}.intersection(candidate_terms):
                negation_penalty = 0.2
            final_score = (0.7 * overlap) + (0.3 * cosine01) - negation_penalty
            scored.append((candidate, round(final_score, 6)))
        return sorted(scored, key=lambda item: item[1], reverse=True)

    def _encode_context(self, turns: list[UserTurn]) -> ContextEmbedding:
        # Encode via embedder
        vectors = self.embedder.embed_texts([t.user_text for t in turns])
        if not vectors:
            vectors = [self.embedder.embed_texts([""])[0]]
            
        weights = self._softmax([float(index + 1) for index in range(len(vectors))])
        merged = [0.0] * self.config.embedding_dim
        for vector, weight in zip(vectors, weights, strict=False):
            for index, value in enumerate(vector):
                merged[index] += value * weight
        return ContextEmbedding(
            vector=self._l2_normalize(merged),
            turns_count=len(turns),
            turn_embeddings=vectors,
            turn_ids=[turn.turn_id for turn in turns],
            context_terms=sorted({term for turn in turns for term in self._tokenize(turn.user_text)}),
        )

    def _predict_response_embedding(self, ctx: ContextEmbedding) -> ResponsePrediction:
        vectors = ctx.turn_embeddings or [ctx.vector]
        weights = self._softmax([float(index + 1) for index in range(len(vectors))])
        predicted = [0.0] * self.config.embedding_dim
        for vector, weight in zip(vectors, weights, strict=False):
            for index, value in enumerate(vector):
                predicted[index] += value * weight
        uncertainty = self._uncertainty(vectors)
        return ResponsePrediction(
            predicted_embedding=self._l2_normalize(predicted),
            uncertainty=uncertainty,
            prediction_method="jepa_literal_embedding",
        )

    def _compute_surprise(self, ctx: ContextEmbedding, actual_response: str) -> SurpriseScore:
        prediction = self._predict_response_embedding(ctx)
        actual = self.embedder.embed_texts([actual_response])[0]
        cosine = self._cosine_similarity(prediction.predicted_embedding, actual)
        base_error = max(0.0, min(1.0, (1.0 - cosine) / 2.0))
        context_terms = set(ctx.context_terms)
        response_terms = set(self._tokenize(actual_response))
        overlap = 0.0
        if context_terms and response_terms:
            overlap = len(context_terms.intersection(response_terms)) / len(response_terms)
        prediction_error = round(max(0.0, min(1.0, (base_error * 1.45) - (0.35 * overlap))), 3)
        ignore = prediction.uncertainty > self.config.uncertainty_ignore_threshold
        return SurpriseScore(
            score=prediction_error,
            prediction_error=prediction_error,
            uncertainty=prediction.uncertainty,
            prediction_method="jepa_literal_embedding",
            source="jepa_prediction_error",
            ignored_due_to_uncertainty=ignore,
        )

    def _history_turns(self, memory_context: MemoryContext | None) -> list[UserTurn]:
        if memory_context is None:
            return []
        turns: list[UserTurn] = []
        for episode in memory_context.recent_episodes:
            if episode.user_turn:
                turns.append(episode.user_turn)
        return turns

    def _tokenize(self, text: str) -> list[str]:
        return [
            token.strip(".,;:!?()[]{}\"'").lower()
            for token in text.split()
            if token.strip(".,;:!?()[]{}\"'")
        ]

    def _uncertainty(self, vectors: list[list[float]]) -> float:
        if len(vectors) < 2:
            return 1.0
        best_similarity = -1.0
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                best_similarity = max(best_similarity, self._cosine_similarity(vectors[i], vectors[j]))
        if best_similarity < -1.0:
            return 1.0
        return round(max(0.0, min(1.0, 1.0 - best_similarity)), 3)

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            return 0.0
        dot = sum(a * b for a, b in zip(left, right, strict=False))
        mag_l = math.sqrt(sum(a * a for a in left))
        mag_r = math.sqrt(sum(b * b for b in right))
        if mag_l == 0.0 or mag_r == 0.0:
            return 0.0
        value = dot / (mag_l * mag_r)
        return max(-1.0, min(1.0, value))

    def _l2_normalize(self, vector: list[float]) -> list[float]:
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return [0.0] * len(vector)
        return [round(value / norm, 6) for value in vector]

    def _softmax(self, values: list[float]) -> list[float]:
        if not values:
            return []
        max_value = max(values)
        exps = [math.exp(value - max_value) for value in values]
        total = sum(exps) or 1.0
        return [value / total for value in exps]

    def _extract_emotional_labels(self, text: str) -> list[str]:
        labels: list[str] = []
        lowered = text.lower()
        for keyword in self.config.salience_keywords:
            if keyword in lowered:
                labels.append(keyword)
        return sorted(set(labels))

    def _estimate_salience(self, text: str, labels: list[str]) -> float:
        salience = 0.15
        for label in labels:
            salience = max(salience, self.config.salience_keywords.get(label, 0.45))
        if "!" in text:
            salience = min(1.0, salience + 0.1)
        return round(max(0.0, min(1.0, salience)), 3)

    def _semantic_density(self, vector: list[float]) -> float:
        if not vector:
            return 0.0
        mean_abs = sum(abs(value) for value in vector) / len(vector)
        return round(max(0.0, min(1.0, mean_abs * 2.5)), 3)

    def _surprise_band(self, surprise: float) -> str:
        if surprise < 0.3:
            return "low"
        if surprise <= 0.6:
            return "medium"
        return "high"

    def _runtime_feedback_bias(self, workspace: CognitiveWorkspace | None) -> float:
        if workspace is None:
            return 0.0
        previous_feedback = workspace.task_frame.get("previous_runtime_feedback", [])
        if not isinstance(previous_feedback, list) or not previous_feedback:
            return 0.0
        rejected = sum(
            int(item.get("rejected_count", 0))
            for item in previous_feedback
            if isinstance(item, dict)
        )
        executed = sum(
            int(item.get("executed_count", 0))
            for item in previous_feedback
            if isinstance(item, dict)
        )
        attempts = max(1, len(previous_feedback))
        rejection_rate = rejected / max(1, rejected + executed)
        intensity = min(0.15, (rejection_rate * 0.12) + (attempts * 0.01))
        return round(max(0.0, intensity), 3)
