from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque

import numpy as np

from calosum.shared.models.types import (
    InputPerceptionState,
    CognitiveWorkspace,
    MemoryContext,
    UserTurn,
)
from calosum.shared.utils.math_cognitive import calculate_efe, calculate_surprise, kl_divergence_gaussian


@dataclass(slots=True)
class ActiveInferenceConfig:
    history_size: int = 50
    min_history_for_zscore: int = 5
    zscore_threshold: float = 2.0


class ActiveInferenceSurpriseAdapter:
    """
    Adapter que implementa Active Inference real para o Hemisfério Direito.
    Substitui a heurística de distância pura por Expected Free Energy (EFE)
    e Z-score móvel para detecção de surpresa.
    """

    def __init__(
        self, base_adapter: Any, config: ActiveInferenceConfig | None = None
    ) -> None:
        self.base_adapter = base_adapter
        self.config = config or ActiveInferenceConfig()
        self._surprise_history: Deque[float] = deque(maxlen=self.config.history_size)

    def perceive(
        self,
        user_turn: UserTurn,
        memory_context: MemoryContext | None = None,
        workspace: CognitiveWorkspace | None = None,
    ) -> InputPerceptionState:
        base_state = self._invoke_sync_perception(user_turn, memory_context, workspace)
        return self._apply_active_inference(base_state, memory_context)

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
            base_state = await asyncio.to_thread(
                self._invoke_sync_perception, user_turn, memory_context, workspace
            )
        return self._apply_active_inference(base_state, memory_context)

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

    def _apply_active_inference(
        self,
        base_state: InputPerceptionState,
        memory_context: MemoryContext | None,
    ) -> InputPerceptionState:
        mu = np.array(base_state.latent_mu) if base_state.latent_mu else np.array(base_state.latent_vector)
        logvar = (
            np.array(base_state.latent_logvar)
            if base_state.latent_logvar
            else np.ones_like(mu) * -2.0
        )
        prior_mu, prior_logvar = self._get_prior(memory_context, mu.shape)
        raw_surprise = calculate_surprise(mu, prior_mu, prior_logvar)
        self._surprise_history.append(raw_surprise)
        calibrated_surprise = max(
            0.0, min(1.0, float(self._calibrate_surprise(raw_surprise)))
        )
        base_confidence = max(0.0, min(1.0, float(base_state.confidence)))
        ambiguity = max(0.0, min(1.0, 1.0 - base_confidence))
        risk = max(0.0, kl_divergence_gaussian(mu, logvar, prior_mu, prior_logvar))
        efe = calculate_efe(mu, logvar, prior_mu, prior_logvar, ambiguity=ambiguity)
        context_novelty = self._context_novelty(mu, logvar, memory_context)
        calibrated_salience = self._calibrate_salience(
            base_state.salience,
            calibrated_surprise,
            ambiguity,
            context_novelty,
        )
        merged_world = dict(base_state.world_hypotheses)
        merged_world.update(
            {
                "prediction_uncertainty": ambiguity,
                "context_novelty": context_novelty,
                "active_inference_risk": float(round(risk, 4)),
            }
        )

        merged_telemetry = dict(base_state.telemetry)
        merged_telemetry.update(
            {
                "surprise_backend": "active_inference_efe",
                "raw_surprise": round(raw_surprise, 4),
                "efe_score": round(efe, 4),
                "efe_risk": round(risk, 4),
                "efe_ambiguity": round(ambiguity, 4),
                "z_score_surprise": calibrated_surprise,
                "context_novelty": round(context_novelty, 4),
                "history_count": len(self._surprise_history),
            }
        )

        return InputPerceptionState(
            context_id=base_state.context_id,
            latent_vector=base_state.latent_vector,
            salience=calibrated_salience,
            emotional_labels=base_state.emotional_labels,
            world_hypotheses=merged_world,
            confidence=base_confidence,
            surprise_score=calibrated_surprise,
            perception_status=base_state.perception_status,
            latent_mu=mu.tolist(),
            latent_logvar=logvar.tolist(),
            telemetry=merged_telemetry,
        )

    def _get_prior(self, memory_context: MemoryContext | None, shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
        if memory_context and memory_context.recent_episodes:
            last_ep = memory_context.recent_episodes[-1]
            if last_ep.right_state.latent_mu:
                if last_ep.right_state.latent_logvar:
                    return np.array(last_ep.right_state.latent_mu), np.array(last_ep.right_state.latent_logvar)
                return np.array(last_ep.right_state.latent_mu), np.ones(shape) * -2.0
            if last_ep.right_state.latent_vector:
                return np.array(last_ep.right_state.latent_vector), np.ones(shape) * -2.0
        return np.zeros(shape), np.zeros(shape)

    def _calibrate_surprise(self, raw_surprise: float) -> float:
        if len(self._surprise_history) < self.config.min_history_for_zscore:
            # tanh ∈ (-1, 1); map to [0, 1] for InputPerceptionState.surprise_score
            t = float(np.tanh(raw_surprise))
            return float(max(0.0, min(1.0, (t + 1.0) / 2.0)))

        history = np.array(self._surprise_history)
        mean = np.mean(history[:-1]) if len(history) > 1 else np.mean(history)
        std = np.std(history[:-1]) + 1e-6 if len(history) > 1 else np.std(history) + 1e-6
        z_score = (raw_surprise - mean) / std
        normalized = 1.0 / (1.0 + np.exp(-(z_score - self.config.zscore_threshold / 2.0)))
        return float(round(max(0.0, min(1.0, normalized)), 4))

    def _context_novelty(
        self,
        mu: np.ndarray,
        logvar: np.ndarray,
        memory_context: MemoryContext | None,
    ) -> float:
        if not memory_context or not memory_context.recent_episodes:
            return 0.0
        prior_mus: list[np.ndarray] = []
        prior_logvars: list[np.ndarray] = []
        for episode in memory_context.recent_episodes:
            right_state = episode.right_state
            if right_state.latent_mu:
                prior_mus.append(np.array(right_state.latent_mu))
                if right_state.latent_logvar:
                    prior_logvars.append(np.array(right_state.latent_logvar))
                else:
                    prior_logvars.append(np.ones_like(mu) * -2.0)
            elif right_state.latent_vector:
                prior_mus.append(np.array(right_state.latent_vector))
                prior_logvars.append(np.ones_like(mu) * -2.0)
        if not prior_mus:
            return 0.0
        hist_mu = np.mean(np.stack(prior_mus, axis=0), axis=0)
        hist_logvar = np.log(np.mean(np.exp(np.stack(prior_logvars, axis=0)), axis=0) + 1e-6)
        novelty = kl_divergence_gaussian(mu, logvar, hist_mu, hist_logvar)
        return float(round(np.tanh(max(0.0, novelty) / 6.0), 4))

    def _calibrate_salience(
        self,
        base_salience: float,
        surprise_score: float,
        ambiguity: float,
        context_novelty: float,
    ) -> float:
        signal = (0.45 * surprise_score) + (0.35 * ambiguity) + (0.20 * context_novelty)
        return float(round(min(1.0, max(0.0, (0.35 * base_salience) + (0.65 * signal))), 3))
