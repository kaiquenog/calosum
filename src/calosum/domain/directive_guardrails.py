from __future__ import annotations

from typing import Any


def apply_controlled_right_hemisphere_params(
    right_hemisphere: Any,
    changes: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    specs = {
        "salience_smoothing_alpha": {"min": 0.1, "max": 0.9, "max_delta": 0.2},
        "salience_max_step": {"min": 0.05, "max": 0.35, "max_delta": 0.1},
        "salience_window_size": {"min": 2, "max": 12, "max_delta": 4},
        "novelty_weight": {"min": 0.0, "max": 0.6, "max_delta": 0.2},
    }
    applied: dict[str, Any] = {}
    rejected: dict[str, str] = {}

    targets: list[tuple[str, Any]] = []
    right_config = getattr(right_hemisphere, "config", None)
    if right_config is not None:
        targets.append(("right_wrapper", right_config))
    base_adapter = getattr(right_hemisphere, "base_adapter", None)
    base_config = getattr(base_adapter, "config", None)
    if base_config is not None:
        targets.append(("right_base", base_config))

    for key, requested in changes.items():
        if key not in specs:
            rejected[key] = "param_not_allowed"
            continue

        spec = specs[key]
        destination = next((cfg for _, cfg in targets if hasattr(cfg, key)), None)
        if destination is None:
            rejected[key] = "param_unavailable_in_runtime"
            continue

        current = getattr(destination, key)
        if not isinstance(current, (int, float)):
            rejected[key] = "param_non_numeric_current_value"
            continue
        if not isinstance(requested, (int, float)):
            rejected[key] = "param_non_numeric_requested_value"
            continue

        bounded = min(spec["max"], max(spec["min"], float(requested)))
        max_delta = float(spec["max_delta"])
        lower = float(current) - max_delta
        upper = float(current) + max_delta
        safe_value = min(upper, max(lower, bounded))

        final_value: Any = int(round(safe_value)) if isinstance(current, int) else round(float(safe_value), 4)
        setattr(destination, key, final_value)
        applied[key] = {
            "from": current,
            "requested": requested,
            "applied": final_value,
            "target": destination.__class__.__name__,
        }

    return applied, rejected
