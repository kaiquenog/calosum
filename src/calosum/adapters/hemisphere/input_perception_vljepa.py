from __future__ import annotations

import asyncio
from dataclasses import dataclass

import numpy as np

from calosum.adapters.hemisphere.input_perception_vjepa21 import VJepa21Config, VJepa21RightHemisphereAdapter
from calosum.shared.models.types import CognitiveWorkspace, MemoryContext, InputPerceptionState, UserTurn


@dataclass(slots=True)
class VLJepaConfig(VJepa21Config):
    hierarchy_levels: int = 3
    visual_weight: float = 0.25


class VLJepaRightHemisphereAdapter(VJepa21RightHemisphereAdapter):
    """Multimodal extension with hierarchical dense latent features."""

    def __init__(self, config: VLJepaConfig | None = None, vision_adapter=None) -> None:
        super().__init__(config=config or VLJepaConfig(), vision_adapter=vision_adapter)
        self.config: VLJepaConfig

    def perceive(
        self,
        user_turn: UserTurn,
        memory_context: MemoryContext | None = None,
        workspace: CognitiveWorkspace | None = None,
    ) -> InputPerceptionState:
        state = super().perceive(user_turn, memory_context, workspace)

        latent = np.asarray(state.latent_vector, dtype=np.float32)
        hierarchy = self._hierarchical_features(latent)
        dense_semantic = float(np.mean([level["energy"] for level in hierarchy]))

        hypotheses = dict(state.world_hypotheses)
        hypotheses.update(
            {
                "hierarchy_levels": float(self.config.hierarchy_levels),
                "dense_semantic_energy": round(dense_semantic, 4),
                "visual_weight": round(self.config.visual_weight, 3),
            }
        )

        telemetry = dict(state.telemetry)
        telemetry.update(
            {
                "right_backend": "vljepa_local",
                "right_model_name": "vl-jepa-local",
                "right_mode": "predictive_multimodal",
                "hierarchical_features": hierarchy,
            }
        )

        enriched = InputPerceptionState(
            context_id=state.context_id,
            latent_vector=state.latent_vector,
            salience=state.salience,
            emotional_labels=state.emotional_labels,
            world_hypotheses=hypotheses,
            confidence=state.confidence,
            surprise_score=state.surprise_score,
            telemetry=telemetry,
        )

        if workspace is not None:
            workspace.right_notes.update(
                {
                    "backend": "vljepa_local",
                    "dense_semantic_energy": hypotheses["dense_semantic_energy"],
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
