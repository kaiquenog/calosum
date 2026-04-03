from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import Any

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

from calosum.shared.models.types import CognitiveWorkspace, MemoryContext, InputPerceptionState, UserTurn


@dataclass(slots=True)
class SimpleDistanceConfig:
    ema_alpha: float = 0.2
    distance_threshold: float = 0.5


class SimpleDistanceSurpriseAdapter:
    """
    Simplificação do Right Hemisphere: substitui Active Inference por distância vetorial pura (EMA).
    Reduz latência e remove dependência de inferactively-pymdp.
    """

    def __init__(self, base_adapter: Any, config: SimpleDistanceConfig | None = None) -> None:
        self.base_adapter = base_adapter
        self.config = config or SimpleDistanceConfig()
        self._ema_latent: np.ndarray | None = None

    def perceive(
        self,
        user_turn: UserTurn,
        memory_context: MemoryContext | None = None,
        workspace: CognitiveWorkspace | None = None,
    ) -> InputPerceptionState:
        base_state = self._invoke_sync_perception(user_turn, memory_context, workspace)
        return self._attach_distance_surprise(base_state, memory_context)

    async def aperceive(
        self,
        user_turn: UserTurn,
        memory_context: MemoryContext | None = None,
        workspace: CognitiveWorkspace | None = None,
    ) -> InputPerceptionState:
        if hasattr(self.base_adapter, "aperceive"):
            try:
                base_state = await self.base_adapter.aperceive(user_turn, memory_context, workspace)
            except TypeError:
                base_state = await self.base_adapter.aperceive(user_turn, memory_context)
        else:
            base_state = await asyncio.to_thread(self._invoke_sync_perception, user_turn, memory_context, workspace)
        return self._attach_distance_surprise(base_state, memory_context)

    def _invoke_sync_perception(
        self,
        user_turn: UserTurn,
        memory_context: MemoryContext | None = None,
        workspace: CognitiveWorkspace | None = None,
    ) -> InputPerceptionState:
        try:
            return self.base_adapter.perceive(user_turn, memory_context, workspace)
        except TypeError:
            return self.base_adapter.perceive(user_turn, memory_context)

    def _attach_distance_surprise(
        self,
        base_state: InputPerceptionState,
        memory_context: MemoryContext | None,
    ) -> InputPerceptionState:
        if np is None:
            return base_state

        current_latent = np.asarray(base_state.latent_vector, dtype=float)
        
        if self._ema_latent is None:
            if memory_context and memory_context.recent_episodes:
                # Inicializa com a média dos episódios recentes se disponível
                vectors = [np.asarray(ep.right_state.latent_vector) for ep in memory_context.recent_episodes if ep.right_state.latent_vector]
                if vectors:
                    self._ema_latent = np.mean(vectors, axis=0)
                else:
                    self._ema_latent = current_latent
            else:
                self._ema_latent = current_latent
            
            surprise_score = base_state.surprise_score
        else:
            # Calcula distância de cosseno entre o estado atual e a média móvel (EMA)
            norm_ema = np.linalg.norm(self._ema_latent)
            norm_curr = np.linalg.norm(current_latent)
            
            if norm_ema > 1e-9 and norm_curr > 1e-9:
                cosine_sim = np.dot(self._ema_latent, current_latent) / (norm_ema * norm_curr)
                # Distância 1 - Similitude
                surprise_score = float(1.0 - max(0.0, min(1.0, cosine_sim)))
            else:
                surprise_score = 0.5
                
            # Atualiza EMA
            self._ema_latent = (1.0 - self.config.ema_alpha) * self._ema_latent + self.config.ema_alpha * current_latent

        merged_telemetry = dict(base_state.telemetry)
        merged_telemetry.update({
            "surprise_backend": "simple_distance_ema",
            "surprise_engine": "cosine_similarity_ema",
            "ema_alpha": self.config.ema_alpha,
        })

        return InputPerceptionState(
            context_id=base_state.context_id,
            latent_vector=base_state.latent_vector,
            salience=base_state.salience,
            emotional_labels=base_state.emotional_labels,
            world_hypotheses=base_state.world_hypotheses,
            confidence=base_state.confidence,
            surprise_score=round(surprise_score, 4),
            telemetry=merged_telemetry,
        )
