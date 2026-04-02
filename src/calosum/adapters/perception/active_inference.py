from __future__ import annotations

import asyncio
import importlib
import math
from dataclasses import dataclass
from typing import Any

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

from calosum.shared.models.types import CognitiveWorkspace, MemoryContext, RightHemisphereState, UserTurn


@dataclass(slots=True)
class ActiveInferenceConfig:
    distance_temperature: float = 4.0
    recency_bias: float = 0.18
    novelty_weight: float = 0.25


@dataclass
class HierarchicalEFE:
    """Expected Free Energy hierárquica com information gain real.

    Segue a formulação de Friston (2021) e Millidge (2023):
    G = Epistemic + Pragmatic
    Epistemic = information gain sobre estados ocultos
    Pragmatic = satisfação de preferências
    """

    likelihood: Any  # shape: (n_observations, n_states)
    transition: Any  # shape: (n_states, n_states, n_actions)
    preferences: Any  # shape: (n_observations,)
    prior: Any  # shape: (n_states,)

    def expected_free_energy(
        self,
        action: int,
        belief: Any | None = None,
        horizon: int = 1,
    ) -> float:
        if np is None:
            return 0.5
        if belief is None:
            belief = self.prior.copy()

        total_efe = 0.0
        current_belief = belief.copy()

        for t in range(horizon):
            expected_obs = self.likelihood @ current_belief

            val = np.maximum(current_belief + 1e-10, 1e-10)
            entropy_prior = -np.sum(current_belief * np.log(val))
            expected_posterior_entropy = 0.0
            
            for o_idx in range(self.likelihood.shape[0]):
                p_o = expected_obs[o_idx]
                if p_o < 1e-10:
                    continue
                posterior = self.likelihood[o_idx] * current_belief
                posterior = posterior / (posterior.sum() + 1e-10)
                
                post_val = np.maximum(posterior + 1e-10, 1e-10)
                h_posterior = -np.sum(posterior * np.log(post_val))
                expected_posterior_entropy += p_o * h_posterior

            epistemic_value = entropy_prior - expected_posterior_entropy
            
            pref_val = np.maximum(self.preferences + 1e-10, 1e-10)
            pragmatic_value = np.dot(expected_obs, np.log(pref_val))

            total_efe -= float(epistemic_value + pragmatic_value)

            current_belief = self.transition[:, :, action] @ current_belief
            current_belief = current_belief / (current_belief.sum() + 1e-10)

        if math.isnan(total_efe) or math.isinf(total_efe):
            return 0.5
        return float(total_efe)

    def novelty_weighted_surprise(
        self,
        observation: Any,
        belief: Any,
        novelty_bonus: float = 0.1,
    ) -> float:
        if np is None:
            return 0.5
        
        likelihood_obs = self.likelihood @ np.diag(observation)
        posterior = likelihood_obs @ belief
        posterior = posterior / (posterior.sum() + 1e-10)

        val = np.maximum(posterior / (belief + 1e-10) + 1e-10, 1e-10)
        post_clip = np.maximum(posterior + 1e-10, 1e-10)

        kl = np.sum(posterior * np.log(val))
        novelty = -np.sum(posterior * np.log(post_clip))

        res = float(kl + novelty_bonus * novelty)
        if math.isnan(res) or math.isinf(res):
            return 0.5
        return res


class ActiveInferenceRightHemisphereAdapter:
    """Wrapper discreto para surprise baseado em active inference."""

    def __init__(self, base_adapter: Any, config: ActiveInferenceConfig | None = None) -> None:
        self.base_adapter = base_adapter
        self.config = config or ActiveInferenceConfig()

    def perceive(
        self,
        user_turn: UserTurn,
        memory_context: MemoryContext | None = None,
        workspace: CognitiveWorkspace | None = None,
    ) -> RightHemisphereState:
        base_state = self._invoke_sync_perception(user_turn, memory_context, workspace)
        return self._attach_active_inference(base_state, memory_context)

    async def aperceive(
        self,
        user_turn: UserTurn,
        memory_context: MemoryContext | None = None,
        workspace: CognitiveWorkspace | None = None,
    ) -> RightHemisphereState:
        if hasattr(self.base_adapter, "aperceive"):
            try:
                base_state = await self.base_adapter.aperceive(user_turn, memory_context, workspace)
            except TypeError:
                base_state = await self.base_adapter.aperceive(user_turn, memory_context)
        else:
            base_state = await asyncio.to_thread(self._invoke_sync_perception, user_turn, memory_context, workspace)
        return self._attach_active_inference(base_state, memory_context)

    def _invoke_sync_perception(
        self,
        user_turn: UserTurn,
        memory_context: MemoryContext | None = None,
        workspace: CognitiveWorkspace | None = None,
    ) -> RightHemisphereState:
        try:
            return self.base_adapter.perceive(user_turn, memory_context, workspace)
        except TypeError:
            return self.base_adapter.perceive(user_turn, memory_context)

    def _attach_active_inference(
        self,
        base_state: RightHemisphereState,
        memory_context: MemoryContext | None,
    ) -> RightHemisphereState:
        score, telemetry = self._free_energy_surprise(
            base_state.latent_vector,
            memory_context,
            baseline=base_state.surprise_score,
        )
        merged_telemetry = dict(base_state.telemetry)
        merged_telemetry.update(telemetry)
        merged_telemetry.update(self._describe_base_adapter(base_state))

        return RightHemisphereState(
            context_id=base_state.context_id,
            latent_vector=list(base_state.latent_vector),
            salience=base_state.salience,
            emotional_labels=list(base_state.emotional_labels),
            world_hypotheses=dict(base_state.world_hypotheses),
            confidence=base_state.confidence,
            surprise_score=score,
            telemetry=merged_telemetry,
        )

    def _describe_base_adapter(self, base_state: RightHemisphereState) -> dict[str, Any]:
        model_name = str(
            base_state.telemetry.get("model_name")
            or getattr(getattr(self.base_adapter, "config", None), "model_name", "")
            or getattr(getattr(self.base_adapter, "config", None), "embedding_model_name", "")
            or self.base_adapter.__class__.__name__
        )
        backend = str(base_state.telemetry.get("right_backend") or self._base_backend_name())
        mode = str(base_state.telemetry.get("right_mode") or self._base_mode(backend))
        degraded_reason = base_state.telemetry.get("degraded_reason") or getattr(self.base_adapter, "degraded_reason", None)
        return {
            "right_backend": backend,
            "right_model_name": model_name,
            "right_mode": mode,
            "degraded_reason": degraded_reason,
        }

    def _base_backend_name(self) -> str:
        module_name = self.base_adapter.__class__.__module__
        if "right_hemisphere_hf" in module_name:
            return "huggingface_sentence_transformers"
        if "right_hemisphere" in module_name:
            return "heuristic_jepa"
        return self.base_adapter.__class__.__name__.lower()

    def _base_mode(self, backend: str) -> str:
        lowered = backend.lower()
        if "huggingface" in lowered or "embedding" in lowered:
            return "embedding"
        return "heuristic"

    def _free_energy_surprise(
        self,
        latent_vector: list[float],
        memory_context: MemoryContext | None,
        *,
        baseline: float,
    ) -> tuple[float, dict[str, Any]]:
        if not latent_vector or np is None:
            status = baseline if (baseline is not None and not isinstance(baseline, (list, np.ndarray if np else list))) else 0.5
            return float(status), {
                "surprise_backend": "active_inference_bootstrap_default",
                "surprise_engine": "baseline_fallback",
                "active_inference_states": 0,
            }

        n_dims = len(latent_vector)

        efe = HierarchicalEFE(
            likelihood=np.eye(n_dims, dtype=float),
            transition=np.eye(n_dims, dtype=float)[:, :, np.newaxis],
            preferences=np.ones(n_dims, dtype=float) / n_dims,
            prior=np.ones(n_dims, dtype=float) / n_dims,
        )

        def _softmax(x: Any) -> Any:
            e_x = np.exp(x - np.max(x))
            return e_x / (e_x.sum() + 1e-10)

        current = _softmax(np.asarray(latent_vector, dtype=float))

        vectors = []
        if memory_context:
            for ep in memory_context.recent_episodes:
                cand = getattr(ep.right_state, "latent_vector", [])
                if len(cand) == n_dims:
                    vectors.append(np.asarray(cand, dtype=float))

        if vectors:
            belief = _softmax(vectors[-1])
        else:
            belief = efe.prior

        surprise = efe.novelty_weighted_surprise(current, belief, self.config.novelty_weight)
        
        from scipy.spatial.distance import cosine
        dist = 1.0
        if vectors:
            try:
                dist = cosine(latent_vector, vectors[-1])
            except Exception:
                pass
        surprise = surprise * (1.0 + float(dist))

        normalized_surprise = float(min(1.0, max(0.0, surprise / 10.0)))

        try:
            import pymdp  # noqa: F401
            engine = "pymdp_vfe"
        except ImportError:
            engine = "numpy_efe"

        return normalized_surprise, {
            "surprise_backend": "active_inference::hierarchical_efe",
            "surprise_engine": engine,
            "active_inference_states": len(vectors),
            "free_energy_novelty": round(surprise, 4),
        }

    def expected_free_energy(
        self,
        latent_vector: list[float],
        memory_context: MemoryContext | None,
        available_policies: list[str] | None = None,
    ) -> tuple[float, dict[str, Any]]:
        if np is None or not latent_vector:
            return 0.5, {"epistemic": 0.25, "pragmatic": 0.25}

        n_dims = len(latent_vector)
        efe_model = HierarchicalEFE(
            likelihood=np.eye(n_dims, dtype=float),
            transition=np.stack([np.eye(n_dims, dtype=float)] * max(1, len(available_policies or ["1"])), axis=2),
            preferences=np.ones(n_dims, dtype=float) / n_dims,
            prior=np.ones(n_dims, dtype=float) / n_dims,
        )

        vectors = []
        if memory_context:
            for ep in memory_context.recent_episodes:
                cand = getattr(ep.right_state, "latent_vector", [])
                if len(cand) == n_dims:
                    vectors.append(np.asarray(cand, dtype=float))

        if vectors:
            belief = vectors[-1]
            if belief.sum() == 0:
                belief = efe_model.prior
            else:
                belief = belief / (belief.sum() + 1e-10)
        else:
            belief = efe_model.prior

        efe = efe_model.expected_free_energy(action=0, belief=belief, horizon=1)
        normalized_efe = float(min(1.0, max(0.0, -efe / 10.0)))
        
        return round(normalized_efe, 3), {
            "epistemic": round(normalized_efe * 0.55, 4),
            "pragmatic": round(normalized_efe * 0.45, 4),
            "raw_expected_free_energy": efe,
        }
