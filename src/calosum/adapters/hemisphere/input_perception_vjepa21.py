from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from calosum.shared.models.ports import InputPerceptionPort, VectorCodecPort, VisionEmbeddingPort
from calosum.shared.models.types import (
    CognitiveWorkspace,
    ComponentHealth,
    MemoryContext,
    MultimodalSignal,
    InputPerceptionState,
    PerceptionStatus,
    UserTurn,
)

@dataclass(slots=True)
class VJepa21Config:
    model_path: Path | None = None
    encoder_filename: str = "encoder.onnx"
    predictor_filename: str = "predictor.onnx"
    latent_size: int = 384
    horizon: int = 4
    action_conditioned: bool = True
    text_embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class VJepa21RightHemisphereAdapter(InputPerceptionPort):
    """V-JEPA 2.1 adapter com degradação explícita e incerteza por MC-dropout."""

    def __init__(
        self,
        config: VJepa21Config | None = None,
        *,
        vision_adapter: VisionEmbeddingPort | None = None,
        codec: VectorCodecPort | None = None,
    ) -> None:
        self.config = config or VJepa21Config()
        self._model_path = str(self.config.model_path) if self.config.model_path else os.getenv("CALOSUM_VJEPA2_MODEL_PATH")
        self._onnx_path = os.getenv("CALOSUM_VJEPA2_ONNX_PATH")
        self._action_conditioned = self.config.action_conditioned
        self._horizon = self.config.horizon
        self._vision_adapter = vision_adapter
        self._vector_codec = codec
        self._health = ComponentHealth.UNAVAILABLE
        self._predictor: Any = None
        self._encoder: Any = None
        self._text_embedder: Any | None = None
        self._latent_size = 768
        self._mc_samples = 12
        self._load_model()
        self._init_text_embedder()

    def _load_model(self) -> None:
        if self._onnx_path and Path(self._onnx_path).exists():
            self._load_onnx()
        elif self._model_path and Path(self._model_path).exists():
            self._load_torch()
        else:
            self._health = ComponentHealth.DEGRADED

    def _load_onnx(self) -> None:
        self._health = ComponentHealth.DEGRADED

    def _load_torch(self) -> None:
        try:
            import torch  # noqa: F401
            from transformers import AutoModel

            self._encoder = AutoModel.from_pretrained(
                self._model_path,
                local_files_only=True,
                torch_dtype=torch.float16,
            )
            self._encoder.eval()
            self._predictor = self._build_predictor()
            self._health = ComponentHealth.HEALTHY
        except Exception:
            self._health = ComponentHealth.DEGRADED

    def _init_text_embedder(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self._text_embedder = SentenceTransformer(
                self.config.text_embedding_model,
                local_files_only=True,
            )
        except Exception:
            self._text_embedder = None

    def _build_predictor(self) -> Any:
        import torch
        from torch import nn

        class LatentPredictor(nn.Module):
            def __init__(self, latent_dim: int, action_dim: int, horizon: int) -> None:
                super().__init__()
                self.horizon = horizon
                self.action_conditioned = action_dim > 0
                input_dim = latent_dim + action_dim if self.action_conditioned else latent_dim
                self.predictor = nn.Sequential(
                    nn.Linear(input_dim, latent_dim * 2),
                    nn.GELU(),
                    nn.Dropout(p=0.15),
                    nn.Linear(latent_dim * 2, latent_dim * horizon),
                )

            def forward(self, z_t: torch.Tensor, action: torch.Tensor | None = None) -> torch.Tensor:
                if self.action_conditioned and action is not None:
                    x = torch.cat([z_t, action], dim=-1)
                else:
                    x = z_t
                predictions = self.predictor(x)
                return predictions.reshape(*z_t.shape[:-1], self.horizon, -1)

        latent_dim = 768
        action_dim = 64 if self._action_conditioned else 0
        return LatentPredictor(latent_dim, action_dim, self._horizon)

    def perceive(
        self,
        user_turn: UserTurn,
        memory_context: MemoryContext | None = None,
        workspace: CognitiveWorkspace | None = None,
    ) -> InputPerceptionState:
        from calosum.shared.utils.math_cognitive import calculate_surprise

        degraded_reason: str | None = None
        perception_status = PerceptionStatus.OBSERVED
        visual_signals = [s for s in user_turn.signals if s.modality.value in ("image", "video")]

        if visual_signals:
            latent_vector, visual_error = self._encode_visual(visual_signals[0])
            if visual_error:
                degraded_reason = visual_error
                perception_status = PerceptionStatus.DEGRADED
                try:
                    latent_vector = self._text_to_latent(user_turn.user_text)
                except self.NullLatentError:
                    latent_vector = None
        else:
            try:
                latent_vector = self._text_to_latent(user_turn.user_text)
            except self.NullLatentError:
                latent_vector = None
            if self._health != ComponentHealth.HEALTHY:
                degraded_reason = "vjepa_weights_unavailable_text_mode"
                perception_status = PerceptionStatus.DEGRADED

        if latent_vector is None:
            degraded_reason = degraded_reason or "no_latent_signal"
            perception_status = PerceptionStatus.BLIND
            latent_vector = np.zeros(self._latent_size, dtype=np.float32)

        latent_mu, latent_logvar, predictor_uncertainty, prediction_error = self._estimate_distribution(latent_vector)
        if latent_mu is None or latent_logvar is None:
            degraded_reason = degraded_reason or "predictive_distribution_unavailable"
            perception_status = PerceptionStatus.BLIND
            latent_mu = latent_vector
            latent_logvar = np.ones_like(latent_vector, dtype=np.float32) * np.log(2.0)
            prediction_error = self._heuristic_prediction_error(latent_vector, memory_context)

        surprise = calculate_surprise(
            latent_vector,
            latent_mu,
            latent_logvar,
        )

        surprise_clamped = max(0.0, min(1.0, float(surprise)))
        uncertainty = max(
            0.0,
            min(
                1.0,
                float(
                    predictor_uncertainty
                    if predictor_uncertainty is not None
                    else min(1.0, float(np.exp(np.mean(latent_logvar))))
                ),
            ),
        )
        confidence = max(0.0, min(1.0, 1.0 - max(surprise_clamped, uncertainty)))
        emotional_labels = self._decode_emotions(latent_vector, user_turn.user_text)

        telemetry = {
            "model_name": "v-jepa-2.1" if self._health == ComponentHealth.HEALTHY else "v-jepa-2.1-fallback",
            "right_backend": "vjepa21_local",
            "right_mode": "predictive",
            "degraded_reason": degraded_reason,
            "perception_status": perception_status.value,
            "surprise_backend": "math_cognitive",
            "jepa_uncertainty": round(uncertainty, 4),
        }
        if perception_status in (PerceptionStatus.BLIND, PerceptionStatus.DEGRADED):
            telemetry["preferred_variant"] = "pragmatico"
            telemetry["left_hemisphere_priority"] = "required"

        world_hypotheses = {
            "prediction_error": float(prediction_error),
            "semantic_density": float(np.std(latent_vector) * 4.0),
            "surprise": surprise_clamped,
            "interaction_complexity": min(1.0, len(user_turn.user_text) / 240.0),
        }

        state = InputPerceptionState(
            context_id=user_turn.turn_id,
            latent_vector=latent_vector.tolist(),
            latent_mu=latent_mu.tolist(),
            latent_logvar=latent_logvar.tolist(),
            salience=self._calibrate_salience(surprise_clamped, emotional_labels),
            emotional_labels=emotional_labels,
            world_hypotheses=world_hypotheses,
            confidence=confidence,
            surprise_score=surprise_clamped,
            perception_status=perception_status,
            telemetry=telemetry,
        )

        if workspace is not None:
            workspace.right_notes.update(
                {
                    "backend": telemetry["right_backend"],
                    "surprise_score": surprise_clamped,
                    "prediction_error": prediction_error,
                    "perception_status": perception_status.value,
                }
            )

        return state

    def _encode_visual(self, signal: MultimodalSignal) -> tuple[np.ndarray | None, str | None]:
        payload = signal.payload
        if isinstance(payload, dict) and "embedding" in payload:
            raw = np.array(payload["embedding"], dtype=np.float32)
        else:
            if self._vision_adapter:
                try:
                    raw = np.asarray(self._vision_adapter.embed_image(payload), dtype=np.float32)
                except Exception as exc:
                    return None, f"vision_adapter_failure:{exc.__class__.__name__}"
            else:
                return None, "vision_adapter_missing"

        if self._vector_codec:
            encoded = self._vector_codec.encode(raw.tolist())
            raw = np.asarray(self._vector_codec.decode(encoded), dtype=np.float32)

        raw = self._project_to_latent(raw)
        norm = np.linalg.norm(raw)
        if norm > 0:
            raw = raw / norm
        return raw, None

    def _estimate_distribution(
        self, latent: np.ndarray
    ) -> tuple[np.ndarray | None, np.ndarray | None, float | None, float]:
        if self._predictor is None:
            return latent, np.ones_like(latent, dtype=np.float32) * np.log(2.0), 1.0, 0.5
        try:
            import torch

            z_t = torch.from_numpy(latent).unsqueeze(0)
            was_training = bool(self._predictor.training)
            self._predictor.train()
            with torch.no_grad():
                draws = [self._predictor(z_t)[:, 0, :].squeeze(0) for _ in range(self._mc_samples)]
            if not was_training:
                self._predictor.eval()
            stacked = torch.stack(draws, dim=0)
            mu = stacked.mean(dim=0).cpu().numpy().astype(np.float32)
            var = stacked.var(dim=0, unbiased=False).clamp(min=1e-6).cpu().numpy().astype(np.float32)
            logvar = np.log(var)
            error = float(np.mean((mu - latent) ** 2))
            uncertainty = float(min(1.0, max(0.0, var.mean() * 8.0)))
            return mu, logvar, uncertainty, error
        except Exception:
            return None, None, None, 0.5

    class NullLatentError(Exception):
        pass

    def _text_to_latent(self, text: str) -> np.ndarray:
        if self._text_embedder is not None:
            try:
                embedding = np.asarray(self._text_embedder.encode(text), dtype=np.float32)
                return self._project_to_latent(embedding)
            except Exception as e:
                raise self.NullLatentError(f"Embedder failed: {e}")
        vector = np.zeros(self._latent_size, dtype=np.float32)
        lowered = text.lower()
        for index, char in enumerate(lowered):
            codepoint = ord(char)
            if codepoint < 32:
                continue
            slot = (index * 131 + codepoint * 17) % self._latent_size
            signal = 1.0 + ((codepoint % 13) / 13.0)
            if (index + codepoint) % 2:
                signal *= -1.0
            vector[slot] += signal
        norm = float(np.linalg.norm(vector))
        if norm > 0.0:
            vector = vector / norm
        return vector

    def _project_to_latent(self, vector: np.ndarray) -> np.ndarray:
        projected = np.zeros(self._latent_size, dtype=np.float32)
        if vector.size == 0:
            return projected
        length = min(vector.size, self._latent_size)
        projected[:length] = vector[:length]
        norm = np.linalg.norm(projected)
        if norm > 0:
            projected = projected / norm
        return projected

    def _heuristic_prediction_error(self, latent: np.ndarray, memory_context: MemoryContext | None) -> float:
        if memory_context and memory_context.recent_episodes:
            recent = memory_context.recent_episodes[-1]
            if hasattr(recent, "right_state") and recent.right_state.latent_vector:
                prev = np.array(recent.right_state.latent_vector, dtype=np.float32)
                if prev.size < self._latent_size:
                    expanded = np.zeros(self._latent_size, dtype=np.float32)
                    expanded[:prev.size] = prev
                    prev = expanded
                elif prev.size > self._latent_size:
                    prev = prev[: self._latent_size]
                cosine_sim = float(np.dot(latent, prev) / (np.linalg.norm(latent) * np.linalg.norm(prev) + 1e-8))
                return max(0.0, 1.0 - cosine_sim)
        return 0.3

    def _decode_emotions(self, latent: np.ndarray, text: str) -> list[str]:
        lowered = text.lower()
        labels: list[str] = []
        if any(marker in lowered for marker in ("urgente", "erro", "falha", "bloqueado", "ansioso")):
            labels.append("frustrated")
        if any(marker in lowered for marker in ("como", "explique", "entender", "por que")):
            labels.append("curious")
        if any(marker in lowered for marker in ("ok", "resolvido", "claro", "feito")):
            labels.append("confident")
        if not labels:
            energy = float(np.std(latent))
            labels.append("curious" if energy > 0.06 else "calm")
        return labels[:3]

    def _calibrate_salience(self, surprise: float, emotions: list[str]) -> float:
        base = 0.5 + 0.3 * surprise
        if "frustrated" in emotions:
            base += 0.15
        if "curious" in emotions:
            base += 0.1
        return min(1.0, base)

    def train_predictor_from_records(
        self,
        records: list[dict[str, Any]],
        *,
        learning_rate: float = 1e-3,
        epochs: int = 1,
    ) -> dict[str, Any]:
        if self._predictor is None:
            return {"status": "skipped", "reason": "predictor_unavailable"}
        if not records:
            return {"status": "skipped", "reason": "empty_dataset"}
        try:
            import torch
            import torch.nn.functional as F

            optimizer = torch.optim.SGD(self._predictor.parameters(), lr=learning_rate)
            self._predictor.train()
            losses: list[float] = []
            for _ in range(max(1, int(epochs))):
                for item in records:
                    latent_t = np.asarray(item.get("latent_t", []), dtype=np.float32)
                    latent_t1 = np.asarray(item.get("latent_t1", []), dtype=np.float32)
                    if latent_t.size == 0 or latent_t1.size == 0:
                        continue
                    z_t = torch.from_numpy(self._project_to_latent(latent_t)).unsqueeze(0)
                    target = torch.from_numpy(self._project_to_latent(latent_t1)).unsqueeze(0)
                    pred = self._predictor(z_t)[:, 0, :]
                    loss = F.mse_loss(pred, target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.detach().item()))
            if not losses:
                return {"status": "skipped", "reason": "no_valid_records"}
            return {
                "status": "success",
                "records_used": len(losses),
                "avg_loss": round(float(sum(losses) / len(losses)), 6),
            }
        except Exception as exc:
            return {"status": "error", "reason": repr(exc)}

    async def aperceive(self, *args: Any, **kwargs: Any) -> InputPerceptionState:
        return self.perceive(*args, **kwargs)
