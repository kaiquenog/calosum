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
        surprise = self._normalize_error(float(np.linalg.norm(predicted - current) / max(1, current.size)))
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
        if memory_context is None or not memory_context.recent_episodes:
            return np.zeros(self.config.latent_size, dtype=np.float32)
        recent = memory_context.recent_episodes[-3:]
        deltas = []
        for ep in recent:
            vec = np.asarray(ep.right_state.latent_vector, dtype=np.float32)
            deltas.append(self._fit_size(vec, self.config.latent_size))
        return np.mean(np.stack(deltas, axis=0), axis=0)

    def _predict_next(self, current: np.ndarray, action_hint: np.ndarray) -> np.ndarray:
        if self._predictor_session is not None:
            try:
                feed: dict[str, np.ndarray] = {}
                for input_meta in self._predictor_session.get_inputs():
                    name = input_meta.name.lower()
                    if "action" in name:
                        feed[input_meta.name] = action_hint.reshape(1, -1).astype(np.float32)
                    else:
                        feed[input_meta.name] = current.reshape(1, -1).astype(np.float32)
                out = self._predictor_session.run(None, feed)[0]
                predicted = np.asarray(out, dtype=np.float32).reshape(-1)
                return self._fit_size(predicted, self.config.latent_size)
            except Exception:
                pass

        # local action-conditioned predictor
        return self._fit_size((0.88 * current) + (0.12 * action_hint), self.config.latent_size)

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
