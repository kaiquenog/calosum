from __future__ import annotations

import asyncio
import importlib
from dataclasses import dataclass
from typing import Any

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

from calosum.shared.types import CognitiveWorkspace, MemoryContext, RightHemisphereState, UserTurn


@dataclass(slots=True)
class ActiveInferenceConfig:
    distance_temperature: float = 4.0
    recency_bias: float = 0.18
    novelty_weight: float = 0.25


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
        vectors = _recent_vectors(memory_context, expected_size=len(latent_vector))
        if not latent_vector or not vectors:
            return baseline or 0.5, {
                "surprise_backend": "active_inference_bootstrap_default",
                "surprise_engine": "baseline_fallback",
                "active_inference_states": len(vectors),
            }

        if np is not None:
            current = np.asarray(latent_vector, dtype=float)
            distances = np.asarray([_cosine_distance_np(current, past) for past in vectors], dtype=float)
            prior = _recency_prior_np(len(distances), self.config.recency_bias)

            backend = "numpy_vfe"
            maths = _load_pymdp_math()
            if maths is not None:
                softmax, log_stable = maths
                posterior = np.asarray(softmax(-self.config.distance_temperature * distances), dtype=float)
                posterior = _normalize_distribution_np(posterior)
                complexity = float(np.sum(posterior * (log_stable(posterior) - log_stable(prior))))
                backend = "pymdp_vfe"
            else:
                posterior = _softmax_np(-self.config.distance_temperature * distances)
                posterior = _normalize_distribution_np(posterior)
                complexity = float(np.sum(posterior * (_safe_log_np(posterior) - _safe_log_np(prior))))

            ambiguity = float(np.sum(posterior * distances))
            novelty = float(np.min(distances) / 2.0)
        else:
            # Heuristic Fallback with Pure Python
            backend = "pure_python_vfe_fallback"
            current_vec = list(latent_vector)
            distances_list = [_cosine_distance_py(current_vec, past) for past in vectors]
            prior = _recency_prior_py(len(distances_list), self.config.recency_bias)
            
            # Simple Softmax
            posterior = _softmax_py([-self.config.distance_temperature * d for d in distances_list])
            complexity = sum(p * (math.log(p or 1e-9) - math.log(pr or 1e-9)) for p, pr in zip(posterior, prior))
            ambiguity = sum(p * d for p, d in zip(posterior, distances_list))
            novelty = min(distances_list) / 2.0

        free_energy = max(0.0, complexity + ambiguity + (self.config.novelty_weight * novelty))
        scale = max(1.0, 1.0 + math.log(len(vectors) + 1))
        normalized = round(float(min(1.0, free_energy / scale)), 3)

        return normalized, {
            "surprise_backend": f"active_inference::{backend}",
            "surprise_engine": backend,
            "active_inference_states": len(distances),
            "free_energy": round(free_energy, 4),
            "free_energy_complexity": round(complexity, 4),
            "free_energy_ambiguity": round(ambiguity, 4),
            "free_energy_novelty": round(novelty, 4),
            "posterior_peak": round(float(np.max(posterior)), 4),
            "memory_alignment": round(float(1.0 - (np.min(distances) / 2.0)), 4),
        }


def _recent_vectors(
    memory_context: MemoryContext | None,
    *,
    expected_size: int,
) -> list[Any]:
    if expected_size <= 0 or memory_context is None:
        return []

    vectors: list[Any] = []
    for episode in memory_context.recent_episodes:
        candidate = getattr(episode.right_state, "latent_vector", [])
        if len(candidate) != expected_size:
            continue
        if np is not None:
            vectors.append(np.asarray(candidate, dtype=float))
        else:
            vectors.append(list(candidate))
    return vectors


def _cosine_distance_np(current: np.ndarray, past: np.ndarray) -> float:
    current_norm = np.linalg.norm(current)
    past_norm = np.linalg.norm(past)
    if current_norm == 0 or past_norm == 0:
        return 1.0
    similarity = float(np.dot(current, past) / (current_norm * past_norm))
    similarity = max(-1.0, min(1.0, similarity))
    return 1.0 - similarity


def _cosine_distance_py(current: list[float], past: list[float]) -> float:
    dot = sum(c * p for c, p in zip(current, past))
    norm_c = math.sqrt(sum(c * c for c in current))
    norm_p = math.sqrt(sum(p * p for p in past))
    if norm_c == 0 or norm_p == 0:
        return 1.0
    similarity = dot / (norm_c * norm_p)
    return 1.0 - max(-1.0, min(1.0, similarity))


def _recency_prior_np(size: int, recency_bias: float) -> np.ndarray:
    weights = np.asarray(
        [1.0 + recency_bias * (size - index - 1) for index in range(size)],
        dtype=float,
    )
    return _normalize_distribution_np(weights)


def _recency_prior_py(size: int, recency_bias: float) -> list[float]:
    weights = [1.0 + recency_bias * (size - index - 1) for index in range(size)]
    total = sum(weights)
    return [w / total if total > 0 else 1.0 / size for w in weights]


def _softmax_np(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exps = np.exp(shifted)
    return _normalize_distribution_np(exps)


def _softmax_py(values: list[float]) -> list[float]:
    if not values:
        return []
    max_val = max(values)
    exps = [math.exp(v - max_val) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


def _normalize_distribution_np(values: np.ndarray) -> np.ndarray:
    total = float(np.sum(values))
    if total <= 0:
        return np.full_like(values, 1.0 / len(values))
    return values / total


def _safe_log_np(values: np.ndarray) -> np.ndarray:
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
