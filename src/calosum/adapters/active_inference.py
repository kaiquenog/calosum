from __future__ import annotations

import asyncio
import importlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from calosum.shared.types import MemoryContext, RightHemisphereState, UserTurn


@dataclass(slots=True)
class ActiveInferenceConfig:
    distance_temperature: float = 4.0
    recency_bias: float = 0.18


class ActiveInferenceRightHemisphereAdapter:
    """
    Wrapper discreto para surprise baseado em active inference.

    O adapter preserva o backbone de percepção já escolhido no bootstrap e
    substitui apenas o cálculo de surprise por uma estimativa de free energy:
    - se `pymdp` estiver disponível, usa as rotinas numéricas do pacote;
    - caso contrário, aplica a mesma decomposição (complexidade + ambiguidade)
      em NumPy para manter o contrato do `RightHemispherePort`.
    """

    def __init__(self, base_adapter: Any, config: ActiveInferenceConfig | None = None) -> None:
        self.base_adapter = base_adapter
        self.config = config or ActiveInferenceConfig()

    def perceive(
        self,
        user_turn: UserTurn,
        memory_context: MemoryContext | None = None,
    ) -> RightHemisphereState:
        base_state = self.base_adapter.perceive(user_turn, memory_context)
        return self._attach_active_inference(base_state, memory_context)

    async def aperceive(
        self,
        user_turn: UserTurn,
        memory_context: MemoryContext | None = None,
    ) -> RightHemisphereState:
        if hasattr(self.base_adapter, "aperceive"):
            base_state = await self.base_adapter.aperceive(user_turn, memory_context)
        else:
            base_state = await asyncio.to_thread(
                self.base_adapter.perceive,
                user_turn,
                memory_context,
            )
        return self._attach_active_inference(base_state, memory_context)

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

    def _free_energy_surprise(
        self,
        latent_vector: list[float],
        memory_context: MemoryContext | None,
        *,
        baseline: float,
    ) -> tuple[float, dict[str, Any]]:
        vectors = _recent_vectors(memory_context, expected_size=len(latent_vector))
        if not latent_vector or not vectors:
            return baseline or 0.5, {
                "surprise_backend": "active_inference_bootstrap_default",
                "surprise_engine": "baseline_fallback",
                "active_inference_states": len(vectors),
            }

        current = np.asarray(latent_vector, dtype=float)
        distances = np.asarray([_cosine_distance(current, past) for past in vectors], dtype=float)
        prior = _recency_prior(len(distances), self.config.recency_bias)

        backend = "numpy_vfe_fallback"
        maths = _load_pymdp_math()
        if maths is not None:
            softmax, log_stable = maths
            posterior = np.asarray(softmax(-self.config.distance_temperature * distances), dtype=float)
            posterior = _normalize_distribution(posterior)
            complexity = float(np.sum(posterior * (log_stable(posterior) - log_stable(prior))))
            backend = "pymdp_vfe"
        else:
            posterior = _softmax(-self.config.distance_temperature * distances)
            posterior = _normalize_distribution(posterior)
            complexity = float(np.sum(posterior * (_safe_log(posterior) - _safe_log(prior))))

        ambiguity = float(np.sum(posterior * distances))
        free_energy = max(0.0, complexity + ambiguity)
        scale = max(1.0, 1.0 + np.log(len(distances) + 1))
        normalized = round(float(min(1.0, free_energy / scale)), 3)

        return normalized, {
            "surprise_backend": f"active_inference::{backend}",
            "surprise_engine": backend,
            "active_inference_states": len(distances),
            "free_energy": round(free_energy, 4),
            "free_energy_complexity": round(complexity, 4),
            "free_energy_ambiguity": round(ambiguity, 4),
            "posterior_peak": round(float(np.max(posterior)), 4),
            "memory_alignment": round(float(1.0 - (np.min(distances) / 2.0)), 4),
        }


def _recent_vectors(
    memory_context: MemoryContext | None,
    *,
    expected_size: int,
) -> list[np.ndarray]:
    if expected_size <= 0 or memory_context is None:
        return []

    vectors: list[np.ndarray] = []
    for episode in memory_context.recent_episodes:
        candidate = getattr(episode.right_state, "latent_vector", [])
        if len(candidate) != expected_size:
            continue
        vectors.append(np.asarray(candidate, dtype=float))
    return vectors


def _cosine_distance(current: np.ndarray, past: np.ndarray) -> float:
    current_norm = np.linalg.norm(current)
    past_norm = np.linalg.norm(past)
    if current_norm == 0 or past_norm == 0:
        return 1.0
    similarity = float(np.dot(current, past) / (current_norm * past_norm))
    similarity = max(-1.0, min(1.0, similarity))
    return 1.0 - similarity


def _recency_prior(size: int, recency_bias: float) -> np.ndarray:
    weights = np.asarray(
        [1.0 + recency_bias * (size - index - 1) for index in range(size)],
        dtype=float,
    )
    return _normalize_distribution(weights)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exps = np.exp(shifted)
    return _normalize_distribution(exps)


def _normalize_distribution(values: np.ndarray) -> np.ndarray:
    total = float(np.sum(values))
    if total <= 0:
        return np.full_like(values, 1.0 / len(values))
    return values / total


def _safe_log(values: np.ndarray) -> np.ndarray:
    return np.log(np.clip(values, 1e-9, 1.0))


def _load_pymdp_math() -> tuple[Any, Any] | None:
    try:
        maths = importlib.import_module("pymdp.maths")
    except Exception:
        return None

    softmax = getattr(maths, "softmax", None) or getattr(maths, "spm_softmax", None)
    log_stable = getattr(maths, "spm_log_single", None)
    if softmax is None or log_stable is None:
        return None
    return softmax, log_stable
