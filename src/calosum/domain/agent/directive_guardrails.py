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


def apply_runtime_contract_audit_directive(action_runtime: Any, directive: Any) -> bool:
    if getattr(directive, "target_component", None) != "action_runtime":
        return False

    changes = getattr(directive, "proposed_change", {})
    if not isinstance(changes, dict):
        return False
    if str(changes.get("action", "")).strip() != "audit_runtime_contracts":
        return False

    raw_failure_types = changes.get("failure_types", {})
    failure_types: dict[str, int] = {}
    if isinstance(raw_failure_types, dict):
        for name, value in raw_failure_types.items():
            if not isinstance(name, str):
                continue
            try:
                failure_types[name] = int(value)
            except (TypeError, ValueError):
                continue

    audit_method = getattr(action_runtime, "audit_runtime_contracts", None)
    if callable(audit_method):
        audit_report = audit_method(failure_types=failure_types)
    else:
        descriptor_loader = getattr(action_runtime, "get_registered_tools", None)
        tool_descriptors = descriptor_loader() if callable(descriptor_loader) else []
        audit_report = {
            "status": "partial",
            "reason": "action_runtime_adapter_does_not_implement_audit_runtime_contracts",
            "registered_tools": len(tool_descriptors),
            "tools": [tool.name for tool in tool_descriptors],
            "validation_failed_recent_count": failure_types.get("validation_failed", 0),
        }

    directive.proposed_change = {
        **changes,
        "failure_types": failure_types,
        "_audit": audit_report,
    }
    directive.status = "applied"
    return True
