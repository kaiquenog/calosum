from __future__ import annotations

import asyncio
from dataclasses import dataclass

import numpy as np

from calosum.adapters.hemisphere.input_perception_vjepa21 import VJepa21Config, VJepa21RightHemisphereAdapter
from calosum.shared.models.types import (
    CognitiveWorkspace,
    ComponentHealth,
    MemoryContext,
    InputPerceptionState,
    PerceptionStatus,
    UserTurn,
)


@dataclass(slots=True)
class VLJepaConfig(VJepa21Config):
    hierarchy_levels: int = 3
    visual_weight: float = 0.25


class VLJepaRightHemisphereAdapter(VJepa21RightHemisphereAdapter):
    """Multimodal extension with hierarchical dense latent features."""

    CONTRACT_VERSION = "vljepa-multimodal-v1"

    def __init__(self, config: VLJepaConfig | None = None, vision_adapter=None) -> None:
        super().__init__(config=config or VLJepaConfig(), vision_adapter=vision_adapter)
        self.config: VLJepaConfig

    def perceive(
        self,
        user_turn: UserTurn,
        memory_context: MemoryContext | None = None,
        workspace: CognitiveWorkspace | None = None,
    ) -> InputPerceptionState:
        from calosum.shared.utils.math_cognitive import calculate_surprise

        visual_signals = [signal for signal in user_turn.signals if signal.modality.value in {"image", "video"}]
        degraded_reason: str | None = None
        perception_status = PerceptionStatus.OBSERVED
        multimodal_active = False

        try:
            text_latent = self._text_to_latent(user_turn.user_text)
        except self.NullLatentError:
            text_latent = None

        visual_latent = None
        if visual_signals:
            visual_latent, visual_error = self._encode_visual(visual_signals[0])
            if visual_error is not None:
                degraded_reason = visual_error
                perception_status = PerceptionStatus.DEGRADED

        if text_latent is not None and visual_latent is not None:
            latent = self._merge_modalities(text_latent, visual_latent)
            multimodal_active = True
        elif text_latent is not None:
            latent = text_latent
            if self._health != ComponentHealth.HEALTHY:
                degraded_reason = degraded_reason or "checkpoint_missing_text_only"
                perception_status = PerceptionStatus.DEGRADED
        elif visual_latent is not None:
            latent = visual_latent
        else:
            degraded_reason = degraded_reason or "no_latent_signal"
            perception_status = PerceptionStatus.BLIND
            latent = np.zeros(self._latent_size, dtype=np.float32)

        latent_mu, latent_logvar, predictor_uncertainty, prediction_error = self._estimate_distribution(latent)
        if latent_mu is None or latent_logvar is None:
            degraded_reason = degraded_reason or "predictive_distribution_unavailable"
            perception_status = PerceptionStatus.BLIND
            latent_mu = latent
            latent_logvar = np.ones_like(latent, dtype=np.float32) * np.log(2.0)
            prediction_error = self._heuristic_prediction_error(latent, memory_context)

        surprise = max(0.0, min(1.0, float(calculate_surprise(latent, latent_mu, latent_logvar))))
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
        confidence = max(0.0, min(1.0, 1.0 - max(surprise, uncertainty)))
        emotional_labels = self._decode_emotions(latent, user_turn.user_text)
        hierarchy = self._hierarchical_features(latent)
        dense_semantic = float(np.mean([level["energy"] for level in hierarchy]))
        hypotheses = {
            "prediction_error": float(prediction_error),
            "semantic_density": float(np.std(latent) * 4.0),
            "surprise": surprise,
            "interaction_complexity": min(1.0, len(user_turn.user_text) / 240.0),
            "hierarchy_levels": float(self.config.hierarchy_levels),
            "dense_semantic_energy": round(dense_semantic, 4),
            "visual_weight": round(self.config.visual_weight, 3),
        }
        telemetry = {
            "model_name": "vl-jepa-local" if self._health == ComponentHealth.HEALTHY else "vl-jepa-local-fallback",
            "right_backend": "vljepa_local",
            "right_model_name": "vl-jepa-local",
            "right_mode": "predictive_multimodal" if multimodal_active else "predictive_text_only",
            "degraded_reason": degraded_reason,
            "perception_status": perception_status.value,
            "jepa_uncertainty": round(uncertainty, 4),
            "hierarchical_features": hierarchy,
            "checkpoint_loaded": self._health == ComponentHealth.HEALTHY,
            "multimodal_active": multimodal_active,
            "contract_version": self.CONTRACT_VERSION,
        }

        enriched = InputPerceptionState(
            context_id=user_turn.turn_id,
            latent_vector=latent.tolist(),
            latent_mu=latent_mu.tolist(),
            latent_logvar=latent_logvar.tolist(),
            salience=self._calibrate_salience(surprise, emotional_labels),
            emotional_labels=emotional_labels,
            world_hypotheses=hypotheses,
            confidence=confidence,
            surprise_score=surprise,
            perception_status=perception_status,
            telemetry=telemetry,
        )

        if workspace is not None:
            workspace.right_notes.update(
                {
                    "backend": "vljepa_local",
                    "dense_semantic_energy": hypotheses["dense_semantic_energy"],
                    "checkpoint_loaded": telemetry["checkpoint_loaded"],
                    "multimodal_active": telemetry["multimodal_active"],
                }
            )
        return enriched

    async def aperceive(
        self,
        user_turn: UserTurn,
        memory_context: MemoryContext | None = None,
        workspace: CognitiveWorkspace | None = None,
    ) -> InputPerceptionState:
        return await asyncio.to_thread(self.perceive, user_turn, memory_context, workspace)

    def _merge_modalities(self, text_latent: np.ndarray, visual_latent: np.ndarray) -> np.ndarray:
        merged = ((1.0 - self.config.visual_weight) * text_latent) + (self.config.visual_weight * visual_latent)
        norm = np.linalg.norm(merged)
        return merged if norm == 0 else merged / norm

    def _hierarchical_features(self, latent: np.ndarray) -> list[dict[str, float]]:
        """Multi-scale spatial pyramid pooling over the latent vector.

        Each level doubles the resolution:
        Level 1: global (full vector) — coarse semantic energy
        Level 2: split into 2 regions — mid-level structure
        Level 3: split into 4 regions — fine-grained details
        ...and so on up to hierarchy_levels.
        """
        levels = max(1, self.config.hierarchy_levels)
        out: list[dict[str, float]] = []
        for level_idx in range(levels):
            n_regions = 2 ** level_idx
            region_size = max(1, latent.size // n_regions)
            level_energies: list[float] = []
            level_variances: list[float] = []
            for r in range(n_regions):
                start = r * region_size
                end = min(start + region_size, latent.size)
                segment = latent[start:end]
                if segment.size == 0:
                    continue
                level_energies.append(float(np.mean(np.abs(segment))))
                level_variances.append(float(np.var(segment)))
            if level_energies:
                out.append({
                    "level": float(level_idx + 1),
                    "energy": float(np.mean(level_energies)),
                    "variance": float(np.mean(level_variances)),
                    "regions": float(n_regions),
                    "contrast": float(np.std(level_energies)) if len(level_energies) > 1 else 0.0,
                })
        return out
