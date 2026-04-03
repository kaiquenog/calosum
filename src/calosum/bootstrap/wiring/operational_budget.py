from __future__ import annotations

import os
from typing import Any, Mapping


_COMPONENT_ENV_PREFIX = {
    "right_hemisphere": "RIGHT",
    "left_hemisphere": "LEFT",
    "bridge": "BRIDGE",
}

_BUDGET_REQUIREMENTS: dict[str, dict[str, dict[str, float | str | None]]] = {
    "right_hemisphere": {
        "heuristic_literal": {"cpu_cores": 0.25, "memory_mb": 256.0, "fallback_backend": None},
        "heuristic_literal_fallback": {"cpu_cores": 0.25, "memory_mb": 256.0, "fallback_backend": None},
        "predictive_checkpoint": {"cpu_cores": 2.0, "memory_mb": 4096.0, "fallback_backend": "heuristic_literal"},
        "vjepa21_local": {"cpu_cores": 3.0, "memory_mb": 6144.0, "fallback_backend": "heuristic_literal"},
        "vljepa_local": {"cpu_cores": 4.0, "memory_mb": 8192.0, "fallback_backend": "heuristic_literal"},
        "jepars_local": {"cpu_cores": 1.5, "memory_mb": 2048.0, "fallback_backend": "heuristic_literal"},
        "distance_huggingface": {"cpu_cores": 1.0, "memory_mb": 1536.0, "fallback_backend": "heuristic_literal"},
    },
    "left_hemisphere": {
        "rlm_recursive_adapter": {"cpu_cores": 2.0, "memory_mb": 3072.0, "fallback_backend": None},
        "rlm_recursive_adapter_default": {"cpu_cores": 2.0, "memory_mb": 3072.0, "fallback_backend": None},
        "openai_responses_adapter": {"cpu_cores": 0.5, "memory_mb": 512.0, "fallback_backend": None},
        "openai_compatible_chat_adapter": {"cpu_cores": 0.5, "memory_mb": 512.0, "fallback_backend": None},
        "resilient_failover_adapter": {"cpu_cores": 0.75, "memory_mb": 768.0, "fallback_backend": None},
    },
    "bridge": {
        "heuristic_projection": {"cpu_cores": 0.25, "memory_mb": 256.0, "fallback_backend": None},
        "cross_attention": {"cpu_cores": 1.0, "memory_mb": 1024.0, "fallback_backend": "heuristic_projection"},
    },
}


def evaluate_backend_budget(
    component: str,
    backend: str,
    *,
    requested_backend: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    env = environ if environ is not None else os.environ
    component_key = component.strip().lower()
    backend_key = backend.strip().lower()
    requested_key = (requested_backend or backend).strip().lower()
    profile = _BUDGET_REQUIREMENTS.get(component_key, {})
    required = profile.get(backend_key, {"cpu_cores": 1.0, "memory_mb": 1024.0, "fallback_backend": None})
    cpu_limit = _resolve_limit(env, component_key, "CPU_CORES")
    memory_limit = _resolve_limit(env, component_key, "MEMORY_MB")
    cpu_required = float(required["cpu_cores"])
    memory_required = float(required["memory_mb"])
    exceeded = (
        (cpu_limit is not None and cpu_required > cpu_limit)
        or (memory_limit is not None and memory_required > memory_limit)
    )
    return {
        "component": component_key,
        "backend": backend_key,
        "requested_backend": requested_key,
        "cpu_cores_required": cpu_required,
        "memory_mb_required": memory_required,
        "cpu_cores_limit": cpu_limit,
        "memory_mb_limit": memory_limit,
        "status": "exceeded" if exceeded else "within_budget",
        "exceeded": exceeded,
        "fallback_backend": required.get("fallback_backend"),
    }


def operational_budget_snapshot(
    *,
    right_backend: str,
    left_backend: str,
    bridge_backend: str,
    requested_right_backend: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    return {
        "right_hemisphere": evaluate_backend_budget(
            "right_hemisphere",
            right_backend,
            requested_backend=requested_right_backend,
            environ=environ,
        ),
        "left_hemisphere": evaluate_backend_budget("left_hemisphere", left_backend, environ=environ),
        "bridge": evaluate_backend_budget("bridge", bridge_backend, environ=environ),
    }


def _resolve_limit(
    env: Mapping[str, str],
    component: str,
    suffix: str,
) -> float | None:
    prefix = _COMPONENT_ENV_PREFIX.get(component)
    scoped = env.get(f"CALOSUM_{prefix}_BUDGET_{suffix}") if prefix else None
    global_value = env.get(f"CALOSUM_BUDGET_{suffix}")
    raw = scoped if scoped not in {None, ""} else global_value
    if raw in {None, ""}:
        return None
    try:
        return float(raw)
    except ValueError:
        return None
