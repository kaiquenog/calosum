from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping


def _path(value: str | os.PathLike[str] | None) -> Path | None:
    if value is None or value == "":
        return None
    return Path(value)


def _parse_csv_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_json_mapping(value: str | None) -> dict[str, str]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    normalized: dict[str, str] = {}
    for key, item in parsed.items():
        if isinstance(key, str) and isinstance(item, str):
            if key.strip() and item.strip():
                normalized[key.strip()] = item.strip()
    return normalized


def should_enable_local_persistence_defaults(
    settings: Any,
    *,
    args: object | None = None,
    environ: Mapping[str, str] | None = None,
) -> bool:
    env = dict(environ or os.environ)

    explicit_profile = getattr(args, "infra_profile", None) is not None or "CALOSUM_INFRA_PROFILE" in env
    explicit_memory_dir = getattr(args, "memory_dir", None) is not None or "CALOSUM_MEMORY_DIR" in env
    explicit_telemetry = getattr(args, "otlp_jsonl", None) is not None or "CALOSUM_OTLP_JSONL" in env

    if explicit_profile or explicit_memory_dir or explicit_telemetry:
        return False

    return (
        str(getattr(settings, "profile", "ephemeral")) == "ephemeral"
        and getattr(settings, "memory_dir", None) is None
        and getattr(settings, "otlp_jsonl", None) is None
    )


def with_local_persistence_defaults(settings: Any) -> Any:
    profile_type = type(getattr(settings, "profile"))
    persistent_value = profile_type("persistent")
    return replace(settings, profile=persistent_value).with_profile_defaults()


def _default_bridge_state_dir(settings: Any) -> Path:
    if getattr(settings, "memory_dir", None) is not None:
        return getattr(settings, "memory_dir").parent / "state"
    if getattr(settings, "duckdb_path", None) is not None:
        return getattr(settings, "duckdb_path").parent
    return Path(".calosum-runtime/state")


def _default_evolution_archive_path(settings: Any) -> Path:
    if getattr(settings, "memory_dir", None) is not None:
        return getattr(settings, "memory_dir").parent / "evolution" / "archive.jsonl"
    if getattr(settings, "otlp_jsonl", None) is not None:
        telemetry_parent = getattr(settings, "otlp_jsonl").parent
        if telemetry_parent.name == "telemetry":
            return telemetry_parent.parent / "evolution" / "archive.jsonl"
        return telemetry_parent / "evolution" / "archive.jsonl"
    return Path(".calosum-runtime/evolution/archive.jsonl")


def _default_gea_experience_store_path(settings: Any) -> Path:
    if getattr(settings, "memory_dir", None) is not None:
        return getattr(settings, "memory_dir").parent / "evolution" / "gea.sqlite"
    if getattr(settings, "otlp_jsonl", None) is not None:
        telemetry_parent = getattr(settings, "otlp_jsonl").parent
        if telemetry_parent.name == "telemetry":
            return telemetry_parent.parent / "evolution" / "gea.sqlite"
        return telemetry_parent / "evolution" / "gea.sqlite"
    return Path(".calosum-runtime/evolution/gea.sqlite")
