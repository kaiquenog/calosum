from __future__ import annotations
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from calosum.shared.models.jepa import ContextEmbedding, ResponsePrediction, SurpriseScore
from calosum.shared.models.types import CognitiveWorkspace, MemoryContext, InputPerceptionState, UserTurn
@dataclass(slots=True)
class TrainedJEPAConfig:
    embedding_dim: int = 384
    hidden: int = 512
    dropout: float = 0.1
    max_turns: int = 3
    uncertainty_samples: int = 10
    uncertainty_ignore_threshold: float = 0.7
    encoder_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    checkpoint_dir: Path = Path("adapters/jepa_predictor/v1.0")
    checkpoint_filename: str = "predictor.pt"
    metadata_filename: str = "training_metadata.json"
    model_name: str = "trained-jepa-v1.0"
class TrainedJEPAAdapter:
    """Adapter JEPA fase 2: preditor MLP treinado com incerteza por MC-dropout."""
    def __init__(self, config: TrainedJEPAConfig | None = None) -> None:
        self.config = config or TrainedJEPAConfig()
        self.degraded_reason: str | None = None
        self._embedder: Any | None = None
        self._torch: Any | None = None
        self._model: Any | None = None
        self._device: str = "cpu"
        self._metadata: dict[str, Any] = {}
        self._load_predictor()
    @property
    def is_available(self) -> bool:
        return self._model is not None
    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)
    def perceive(
        self,
        user_turn: UserTurn,
        memory_context: MemoryContext | None = None,
        workspace: CognitiveWorkspace | None = None,
    ) -> InputPerceptionState:
        turns = self._history_turns(memory_context)
        turns.append(user_turn)
        turns = turns[-self.config.max_turns :]
        context = self._encode_context(turns)
        posterior_mu, posterior_logvar, prediction = self._predict_posterior(context)
        surprise = self._compute_surprise(context, user_turn.user_text, prediction)
        runtime_feedback_bias = self._runtime_feedback_bias(workspace)
        salience = self._estimate_salience(
            posterior_mu,
            posterior_logvar,
            surprise_score=surprise.score,
            runtime_feedback_bias=runtime_feedback_bias,
        )
        emotional_labels = self._infer_emotional_labels(
            salience=salience,
            uncertainty=prediction.uncertainty,
            surprise_score=surprise.score,
        )
        telemetry = {
            "model_name": self.config.model_name,
            "right_backend": "trained_jepa_local",
            "right_model_name": self.config.model_name,
            "right_mode": "predictive",
            "degraded_reason": self.degraded_reason,
            "prediction_method": prediction.prediction_method,
            "jepa_uncertainty": prediction.uncertainty,
            "prediction_error": surprise.prediction_error,
            "surprise_source": surprise.source,
            "surprise_band": self._surprise_band(surprise.score),
            "ignore_surprise_for_branching": surprise.ignored_due_to_uncertainty,
            "salience_strategy": "latent_uncertainty_gradient",
            "context_turns_used": len(turns),
            "checkpoint_dir": str(self.config.checkpoint_dir),
            "checkpoint_loaded": self.is_available,
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
            emotional_labels=emotional_labels,
            world_hypotheses=world_hypotheses,
            confidence=round(max(0.0, min(1.0, 1.0 - prediction.uncertainty)), 3),
            surprise_score=surprise.score,
            latent_mu=posterior_mu,
            latent_logvar=posterior_logvar,
            telemetry=telemetry,
        )
        if workspace is not None:
            workspace.right_notes.update(
                {
                    "backend": "trained_jepa_local",
                    "surprise_score": state.surprise_score,
                    "surprise_band": telemetry["surprise_band"],
                    "prediction_method": prediction.prediction_method,
                    "surprise_source": surprise.source,
                    "uncertainty": prediction.uncertainty,
                    "ignore_surprise_for_branching": surprise.ignored_due_to_uncertainty,
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
    async def compute_surprise(self, ctx: ContextEmbedding, actual_response: str) -> SurpriseScore:
        return self._compute_surprise(ctx, actual_response)
    def predict_with_uncertainty(
        self, ctx: ContextEmbedding, n_samples: int | None = None
    ) -> tuple[list[float], float]:
        mean, _, uncertainty = self._predict_distribution(ctx, n_samples)
        return mean, uncertainty
    def _predict_distribution(
        self, ctx: ContextEmbedding, n_samples: int | None = None
    ) -> tuple[list[float], list[float], float]:
        vectors = self._ctx_to_matrix(ctx)
        if not self.is_available:
            predicted = self._weighted_merge(vectors)
            mean = self._l2_normalize(predicted)
            return mean, [-0.5] * self.config.embedding_dim, 1.0
        assert self._torch is not None
        assert self._model is not None
        sample_count = max(1, n_samples or self.config.uncertainty_samples)
        tensor = self._torch.tensor([vectors], dtype=self._torch.float32, device=self._device)
        was_training = bool(self._model.training)
        self._model.train()
        with self._torch.no_grad():
            preds = self._torch.stack([self._model(tensor).squeeze(0) for _ in range(sample_count)], dim=0)
        if not was_training:
            self._model.eval()
        mean = preds.mean(dim=0)
        mean = mean / (mean.norm(p=2) + 1e-8)
        variances = preds.var(dim=0).clamp(min=1e-6)
        uncertainty = max(0.0, min(1.0, float(variances.mean().item())))
        logvar = [round(float(math.log(item)), 6) for item in variances.cpu().tolist()]
        return mean.cpu().tolist(), logvar, round(uncertainty, 6)
    def _encode_context(self, turns: list[UserTurn]) -> ContextEmbedding:
        selected = turns[-self.config.max_turns :] if turns else []
        embeddings = [self._encode_text(turn.user_text) for turn in selected]
        matrix = self._pad_left(embeddings, self.config.max_turns)
        merged = self._weighted_merge(matrix)
        context_terms = sorted({term for turn in selected for term in self._tokenize(turn.user_text)})
        return ContextEmbedding(
            vector=self._l2_normalize(merged),
            turns_count=len(selected),
            turn_embeddings=matrix,
            turn_ids=[turn.turn_id for turn in selected],
            context_terms=context_terms,
        )
    def _predict_response_embedding(self, ctx: ContextEmbedding) -> ResponsePrediction:
        predicted, uncertainty = self.predict_with_uncertainty(
            ctx, self.config.uncertainty_samples
        )
        method = "jepa_trained" if self.is_available else "mean_pooling"
        return ResponsePrediction(
            predicted_embedding=self._l2_normalize(predicted),
            uncertainty=round(max(0.0, min(1.0, uncertainty)), 3),
            prediction_method=method,
        )
    def _compute_surprise(
        self,
        ctx: ContextEmbedding,
        actual_response: str,
        prediction: ResponsePrediction | None = None,
    ) -> SurpriseScore:
        prediction = prediction or self._predict_response_embedding(ctx)
        actual = self._encode_text(actual_response)
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
            prediction_method=prediction.prediction_method,
            source="jepa_prediction_error",
            ignored_due_to_uncertainty=ignore,
        )
    def _predict_posterior(
        self,
        ctx: ContextEmbedding,
    ) -> tuple[list[float], list[float], ResponsePrediction]:
        mu, logvar, uncertainty = self._predict_distribution(ctx, self.config.uncertainty_samples)
        method = "jepa_trained" if self.is_available else "mean_pooling"
        prediction = ResponsePrediction(
            predicted_embedding=self._l2_normalize(mu),
            uncertainty=round(max(0.0, min(1.0, uncertainty)), 3),
            prediction_method=method,
        )
        return prediction.predicted_embedding, logvar, prediction
    def _load_predictor(self) -> None:
        checkpoint = self.config.checkpoint_dir / self.config.checkpoint_filename
        metadata_path = self.config.checkpoint_dir / self.config.metadata_filename
        if metadata_path.exists():
            try:
                self._metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception:
                self._metadata = {}
        if not checkpoint.exists():
            self.degraded_reason = "checkpoint_missing"
            return
        try:
            import torch
            from torch import nn
            import torch.nn.functional as F
        except Exception as exc:
            self.degraded_reason = f"torch_unavailable:{exc.__class__.__name__}"
            return
        class JEPAPredictor(nn.Module):
            def __init__(self, embed_dim: int, hidden: int, dropout: float) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(embed_dim * 3, hidden),
                    nn.GELU(),
                    nn.LayerNorm(hidden),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, hidden // 2),
                    nn.GELU(),
                    nn.Linear(hidden // 2, embed_dim),
                )
            def forward(self, ctx_embeds):
                return F.normalize(self.net(ctx_embeds.flatten(1)), dim=-1)
        self._torch = torch
        model = JEPAPredictor(self.config.embedding_dim, self.config.hidden, self.config.dropout)
        payload = torch.load(checkpoint, map_location="cpu")
        state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
        model.load_state_dict(state_dict)
        model.eval()
        self._model = model
        self._device = "cpu"
        self.degraded_reason = None
    def _history_turns(self, memory_context: MemoryContext | None) -> list[UserTurn]:
        if memory_context is None:
            return []
        turns: list[UserTurn] = []
        for episode in memory_context.recent_episodes:
            if episode.user_turn:
                turns.append(episode.user_turn)
        return turns
    def _encode_text(self, text: str) -> list[float]:
        try:
            if self._embedder is None:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(
                    self.config.encoder_model_name,
                    local_files_only=True,
                )
            vec = self._embedder.encode([text], normalize_embeddings=True)[0].tolist()
            return self._normalize_size(vec)
        except Exception:
            return self._lexical_vector(text)
    def _normalize_size(self, vector: list[float]) -> list[float]:
        if len(vector) > self.config.embedding_dim:
            vector = vector[: self.config.embedding_dim]
        elif len(vector) < self.config.embedding_dim:
            vector = vector + ([0.0] * (self.config.embedding_dim - len(vector)))
        return self._l2_normalize(vector)
    def _lexical_vector(self, text: str) -> list[float]:
        tokens = self._tokenize(text) or ["silence"]
        vector = [0.0] * self.config.embedding_dim
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for index in range(0, len(digest), 4):
                chunk = digest[index : index + 4]
                pos = int.from_bytes(chunk[:2], "big") % self.config.embedding_dim
                sign = 1.0 if chunk[2] % 2 == 0 else -1.0
                magnitude = 1.0 + (chunk[3] / 255.0)
                vector[pos] += sign * magnitude
        return self._l2_normalize(vector)
    def _tokenize(self, text: str) -> list[str]:
        return [token.strip(".,;:!?()[]{}\"'").lower() for token in text.split() if token.strip(".,;:!?()[]{}\"'")]
    def _pad_left(self, vectors: list[list[float]], size: int) -> list[list[float]]:
        vectors = vectors[-size:]
        pad_count = max(0, size - len(vectors))
        pad = [[0.0] * self.config.embedding_dim for _ in range(pad_count)]
        return pad + vectors
    def _ctx_to_matrix(self, ctx: ContextEmbedding) -> list[list[float]]:
        matrix = list(ctx.turn_embeddings or [])
        matrix = [self._normalize_size(vector) for vector in matrix]
        return self._pad_left(matrix, self.config.max_turns)
    def _weighted_merge(self, vectors: list[list[float]]) -> list[float]:
        weights = self._softmax([float(index + 1) for index in range(len(vectors))])
        merged = [0.0] * self.config.embedding_dim
        for vector, weight in zip(vectors, weights, strict=False):
            for idx, value in enumerate(vector):
                merged[idx] += value * weight
        return merged
    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            return 0.0
        dot = sum(a * b for a, b in zip(left, right, strict=False))
        mag_l = math.sqrt(sum(a * a for a in left))
        mag_r = math.sqrt(sum(b * b for b in right))
        if mag_l == 0.0 or mag_r == 0.0:
            return 0.0
        return max(-1.0, min(1.0, dot / (mag_l * mag_r)))
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
    def _estimate_salience(
        self,
        latent_mu: list[float],
        latent_logvar: list[float],
        *,
        surprise_score: float,
        runtime_feedback_bias: float,
    ) -> float:
        norm = math.sqrt(sum(value * value for value in latent_mu)) / math.sqrt(
            max(1, len(latent_mu))
        )
        precision = 1.0 - max(0.0, min(1.0, math.exp(sum(latent_logvar) / max(1, len(latent_logvar)))))
        signal = (0.35 * norm) + (0.35 * surprise_score) + (0.30 * precision)
        return round(min(1.0, max(0.0, signal + runtime_feedback_bias)), 3)
    def _infer_emotional_labels(
        self,
        *,
        salience: float,
        uncertainty: float,
        surprise_score: float,
    ) -> list[str]:
        if salience < 0.2 and uncertainty < 0.35:
            return ["neutral"]
        labels = ["activated"] if salience >= 0.55 else ["focused"]
        if uncertainty >= 0.6:
            labels.append("uncertain")
        if surprise_score >= 0.7:
            labels.append("novel")
        return labels
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
        rejected = sum(int(item.get("rejected_count", 0)) for item in previous_feedback if isinstance(item, dict))
        executed = sum(int(item.get("executed_count", 0)) for item in previous_feedback if isinstance(item, dict))
        attempts = max(1, len(previous_feedback))
        rejection_rate = rejected / max(1, rejected + executed)
        intensity = min(0.15, (rejection_rate * 0.12) + (attempts * 0.01))
        return round(max(0.0, intensity), 3)
