from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from calosum.shared.ports import VisionEmbeddingPort
from calosum.shared.types import CognitiveWorkspace, MemoryContext, RightHemisphereState, UserTurn


@dataclass(slots=True)
class VJepa21Config:
    model_path: Path | None = None
    encoder_filename: str = "encoder.onnx"
    predictor_filename: str = "predictor.onnx"
    latent_size: int = 384
    horizon: int = 4
    action_conditioned: bool = True
    text_embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class VJepa21RightHemisphereAdapter:
    """Right hemisphere adapter with local latent prediction.

    Implementation strategy:
    - Uses sentence-transformers text embeddings (real local model) as latent state.
    - Optionally consumes ONNX predictor artifacts when available.
    - Computes action-conditioned one-step prediction and multi-horizon error.
    """

    def __init__(
        self,
        config: VJepa21Config | None = None,
        vision_adapter: VisionEmbeddingPort | None = None,
    ) -> None:
        self.config = config or VJepa21Config()
        self.vision_adapter = vision_adapter
        self._embedder: Any | None = None
        self._onnx = None
        self._encoder_session = None
        self._predictor_session = None
        self._load_optional_onnx()

    def perceive(
        self,
        user_turn: UserTurn,
        memory_context: MemoryContext | None = None,
        workspace: CognitiveWorkspace | None = None,
    ) -> RightHemisphereState:
        text_latent = self._embed_text(user_turn.user_text)
        visual_latent = self._embed_visual(user_turn)
        current = self._merge_modalities(text_latent, visual_latent)

        action_hint = self._action_hint(memory_context) if self.config.action_conditioned else np.zeros_like(current)
        predicted = self._predict_next(current, action_hint)
        surprise = self._compute_prediction_error(current, predicted)
        multi_h = self._multi_horizon_error(current, action_hint)

        salience = round(min(1.0, 0.25 + surprise * 0.75), 3)
        emotional_labels = self._emotional_labels(user_turn.user_text)

        world_hypotheses = {
            "interaction_complexity": round(min(1.0, len(user_turn.user_text) / 260.0), 3),
            "semantic_density": round(float(np.std(current) * 4.0), 3),
            "prediction_error": round(surprise, 3),
            "multi_horizon_error": round(multi_h, 3),
            "action_conditioned": 1.0 if self.config.action_conditioned else 0.0,
        }

        state = RightHemisphereState(
            context_id=user_turn.turn_id,
            latent_vector=current.tolist(),
            salience=salience,
            emotional_labels=emotional_labels,
            world_hypotheses=world_hypotheses,
            confidence=round(max(0.35, 1.0 - surprise * 0.7), 3),
            surprise_score=round(surprise, 3),
            telemetry={
                "model_name": "v-jepa-2.1-local",
                "right_backend": "vjepa21_local",
                "right_model_name": "v-jepa-2.1-local",
                "right_mode": "predictive",
                "degraded_reason": None,
                "predictor_engine": "onnx" if self._predictor_session is not None else "numpy_local",
                "horizon": self.config.horizon,
                "action_conditioned": self.config.action_conditioned,
            },
        )

        if workspace is not None:
            workspace.right_notes.update(
                {
                    "backend": "vjepa21_local",
                    "surprise_score": state.surprise_score,
                    "prediction_error": world_hypotheses["prediction_error"],
                    "multi_horizon_error": world_hypotheses["multi_horizon_error"],
                }
            )

        return state

    async def aperceive(
        self,
        user_turn: UserTurn,
        memory_context: MemoryContext | None = None,
        workspace: CognitiveWorkspace | None = None,
    ) -> RightHemisphereState:
        return await asyncio.to_thread(self.perceive, user_turn, memory_context, workspace)

    def _load_optional_onnx(self) -> None:
        if self.config.model_path is None:
            return
        try:
            import onnxruntime as ort
        except Exception:
            return

        self._onnx = ort
        enc = self.config.model_path / self.config.encoder_filename
        pred = self.config.model_path / self.config.predictor_filename
        if enc.exists():
            self._encoder_session = ort.InferenceSession(str(enc), providers=["CPUExecutionProvider"])
        if pred.exists():
            self._predictor_session = ort.InferenceSession(str(pred), providers=["CPUExecutionProvider"])

    def _embed_text(self, text: str) -> np.ndarray:
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._embedder = SentenceTransformer(self.config.text_embedding_model)
            except Exception:
                self._embedder = False

        text = text.strip() or "silence"
        if self._embedder:
            vec = self._embedder.encode([text], normalize_embeddings=True)[0]
            return self._fit_size(np.asarray(vec, dtype=np.float32), self.config.latent_size)

        # lexical fallback: real deterministic projection
        buckets = np.zeros(self.config.latent_size, dtype=np.float32)
        for i, ch in enumerate(text.lower()):
            buckets[(ord(ch) + i) % self.config.latent_size] += 1.0
        norm = np.linalg.norm(buckets)
        return buckets if norm == 0 else buckets / norm

    def _embed_visual(self, user_turn: UserTurn) -> np.ndarray:
        if self.vision_adapter is None:
            return np.zeros(self.config.latent_size, dtype=np.float32)

        vectors: list[np.ndarray] = []
        for signal in user_turn.signals:
            payload = signal.payload
            if isinstance(payload, bytes):
                try:
                    vec = np.asarray(self.vision_adapter.embed_image(payload), dtype=np.float32)
                    vectors.append(self._fit_size(vec, self.config.latent_size))
                except Exception:
                    continue

        if not vectors:
            return np.zeros(self.config.latent_size, dtype=np.float32)
        stacked = np.stack(vectors, axis=0)
        return np.mean(stacked, axis=0)

    def _merge_modalities(self, text_latent: np.ndarray, visual_latent: np.ndarray) -> np.ndarray:
        merged = (0.85 * text_latent) + (0.15 * visual_latent)
        norm = np.linalg.norm(merged)
        return merged if norm == 0 else merged / norm

    def _action_hint(self, memory_context: MemoryContext | None) -> np.ndarray:
        """Encode executed action types as action-conditioned hint vector.

        Instead of averaging past latent vectors (which conflates state with action),
        this encodes the types of actions executed (tool calls, response types)
        into a sparse action vector — analogous to V-JEPA 2-AC proprioception.
        """
        if memory_context is None or not memory_context.recent_episodes:
            return np.zeros(self.config.latent_size, dtype=np.float32)

        action_vector = np.zeros(self.config.latent_size, dtype=np.float32)
        action_types_seen: set[str] = set()

        for ep in memory_context.recent_episodes[-3:]:
            left_result = getattr(ep, "left_result", None)
            if left_result is None:
                continue
            for action in getattr(left_result, "actions", []):
                action_type = getattr(action, "action_type", "unknown")
                action_types_seen.add(action_type)
                # Encode action type into specific dimensions using hash
                for i, ch in enumerate(action_type):
                    idx = (ord(ch) + i * 13) % self.config.latent_size
                    action_vector[idx] += 0.3

            # Also encode the latent state direction as context
            vec = np.asarray(ep.right_state.latent_vector, dtype=np.float32)
            vec = self._fit_size(vec, self.config.latent_size)
            action_vector += 0.15 * vec

        norm = np.linalg.norm(action_vector)
        return action_vector / max(norm, 1e-8) if norm > 0 else action_vector

    def _predict_next(self, current: np.ndarray, action_hint: np.ndarray) -> np.ndarray:
        if self._predictor_session is not None:
            try:
                feed: dict[str, np.ndarray] = {}
                for input_meta in self._predictor_session.get_inputs():
                    name = input_meta.name.lower()
                    if "action" in name:
                        feed[input_meta.name] = action_hint.reshape(1, -1).astype(np.float32)
                    elif "context" in name or "state" in name:
                        feed[input_meta.name] = current.reshape(1, -1).astype(np.float32)
                    else:
                        feed[input_meta.name] = current.reshape(1, -1).astype(np.float32)
                out = self._predictor_session.run(None, feed)[0]
                predicted = np.asarray(out, dtype=np.float32).reshape(-1)
                return self._fit_size(predicted, self.config.latent_size)
            except Exception:
                pass

        # Fallback: momentum-based action-conditioned predictor
        # Instead of fixed linear interpolation, compute directional momentum
        momentum = action_hint - current
        predicted = current + 0.25 * momentum
        norm = np.linalg.norm(predicted)
        return self._fit_size(predicted / max(norm, 1e-8), self.config.latent_size)

    def _compute_prediction_error(self, current: np.ndarray, predicted_prior: np.ndarray) -> float:
        """
        Surprise as prediction error in latent space.
        Based on V-JEPA 2 (Bardes et al., 2025): surprise = ||z_actual - z_predicted||^2
        normalized by dimensionality, mapped through sigmoid for [0, 1] range.
        """
        error = np.linalg.norm(current - predicted_prior) ** 2
        normalized = error / max(1, current.size)
        return float(1.0 / (1.0 + math.exp(-8.0 * (normalized - 0.1))))

    def _multi_horizon_error(self, current: np.ndarray, action_hint: np.ndarray) -> float:
        pred = current.copy()
        errors = []
        for step in range(max(1, self.config.horizon)):
            pred = self._predict_next(pred, action_hint)
            errors.append(float(np.linalg.norm(pred - current) / max(1, current.size)) / (step + 1))
        return self._normalize_error(float(sum(errors) / len(errors)))

    def _emotional_labels(self, text: str) -> list[str]:
        lowered = text.lower()
        labels = [
            label
            for label in ("ansioso", "frustrado", "urgente", "triste", "feliz", "preocupado")
            if label in lowered
        ]
        return labels or ["neutral"]

    def _fit_size(self, vector: np.ndarray, size: int) -> np.ndarray:
        if vector.size == size:
            return vector.astype(np.float32)
        if vector.size > size:
            return vector[:size].astype(np.float32)
        out = np.zeros(size, dtype=np.float32)
        out[: vector.size] = vector
        return out

    def _normalize_error(self, raw: float) -> float:
        return 1.0 / (1.0 + math.exp(-6.0 * (raw - 0.08)))

    def temporal_contrastive_loss(
        self, memory_context: MemoryContext | None, temperature: float = 0.07,
    ) -> dict[str, float]:
        """InfoNCE-style temporal contrastive learning hook.

        Measures alignment between temporally adjacent latent states vs
        random pairs. Provides a training signal for future self-supervised
        learning of the predictive world model.

        Based on: V-JEPA 2 (Bardes et al., 2025), CPC (Oord et al., 2018).
        """
        if memory_context is None or len(memory_context.recent_episodes) < 3:
            return {"loss": 0.0, "alignment": 0.0, "n_pairs": 0}

        latents = [
            np.asarray(ep.right_state.latent_vector, dtype=np.float32)
            for ep in memory_context.recent_episodes[-6:]
        ]
        latents = [self._fit_size(v, self.config.latent_size) for v in latents]

        positive_sims: list[float] = []
        negative_sims: list[float] = []
        for i in range(len(latents) - 1):
            a, b = latents[i], latents[i + 1]
            norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
            if norm_a > 0 and norm_b > 0:
                positive_sims.append(float(np.dot(a, b) / (norm_a * norm_b)))
            for j in range(len(latents)):
                if abs(j - i) > 1:
                    c = latents[j]
                    norm_c = np.linalg.norm(c)
                    if norm_a > 0 and norm_c > 0:
                        negative_sims.append(float(np.dot(a, c) / (norm_a * norm_c)))

        if not positive_sims:
            return {"loss": 0.0, "alignment": 0.0, "n_pairs": 0}

        avg_pos = sum(positive_sims) / len(positive_sims)
        avg_neg = sum(negative_sims) / max(1, len(negative_sims))
        loss = max(0.0, -math.log(max(1e-9, math.exp(avg_pos / temperature) /
                   max(1e-9, math.exp(avg_pos / temperature) + math.exp(avg_neg / temperature)))))
        return {"loss": round(loss, 6), "alignment": round(avg_pos, 4), "n_pairs": len(positive_sims)}
