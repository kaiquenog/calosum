from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping
class InfrastructureProfile(StrEnum):
    EPHEMERAL = "ephemeral"
    PERSISTENT = "persistent"
    DOCKER = "docker"


@dataclass(slots=True)
class InfrastructureSettings:
    profile: InfrastructureProfile = InfrastructureProfile.EPHEMERAL
    memory_dir: Path | None = None
    otlp_jsonl: Path | None = None
    vector_db_url: str | None = None
    duckdb_path: Path | None = None
    bridge_state_dir: Path | None = None
    evolution_archive_path: Path | None = None
    awareness_interval_turns: int = 1
    api_port: int = 8000
    otel_collector_endpoint: str | None = None
    jaeger_ui_url: str | None = None
    right_hemisphere_endpoint: str | None = None
    right_hemisphere_backend: str | None = None
    right_model_path: Path | None = None
    right_action_conditioned: bool = True
    right_horizon: int = 4
    right_jepars_binary: str | None = None
    left_hemisphere_endpoint: str | None = None
    left_hemisphere_api_key: str | None = None
    left_hemisphere_model: str | None = None
    left_hemisphere_provider: str | None = None
    left_hemisphere_reasoning_effort: str | None = None
    left_hemisphere_backend: str | None = None
    left_rlm_runtime_command: str | None = None
    left_rlm_path: Path | None = None
    left_rlm_max_depth: int = 3
    event_bus: Any | None = None

    left_hemisphere_fallback_endpoint: str | None = None
    left_hemisphere_fallback_api_key: str | None = None
    left_hemisphere_fallback_model: str | None = None
    left_hemisphere_fallback_provider: str | None = None
    left_hemisphere_fallback_reasoning_effort: str | None = None

    # Routing Policy
    perception_model: str | None = None
    reason_model: str | None = None
    reflection_model: str | None = None
    verifier_model: str | None = None

    # Embedding settings
    embedding_endpoint: str | None = None
    embedding_api_key: str | None = None
    embedding_model: str | None = None
    embedding_provider: str | None = None
    bridge_backend: str | None = None
    gea_experience_store_path: Path | None = None
    gea_sharing_enabled: bool = False
    telegram_bot_token: str | None = None
    telegram_dm_policy: str = "open"
    telegram_allowlist_ids: list[str] = field(default_factory=list)
    vault: dict[str, str] | None = None

    # Vector quantization (TurboQuant)
    vector_quantization: str = "none"   # "none" | "turboquant"
    turboquant_bits: int = 4
    qdrant_scalar_quantization: bool = False
    mcp_enabled: bool = False
    mcp_servers: dict[str, str] = field(default_factory=dict)
    mcp_allowlist: list[str] = field(default_factory=list)

    @classmethod
    def from_sources(
        cls,
        *,
        args: object | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> "InfrastructureSettings":
        env = dict(environ or os.environ)
        
        # Parseia o .env localmente caso estejamos num terminal sem python-dotenv
        env_path = Path(__file__).resolve().parent.parent.parent.parent / ".env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                doc = line.strip()
                if doc and not doc.startswith("#") and "=" in doc:
                    k, v = doc.split("=", 1)
                    if k.strip() not in env:
                        env[k.strip()] = v.strip().strip("'\"")

        def _arg(name: str):
            return getattr(args, name, None) if args is not None else None

        profile_raw = _arg("infra_profile") or env.get(
            "CALOSUM_INFRA_PROFILE",
            InfrastructureProfile.EPHEMERAL.value,
        )
        profile = InfrastructureProfile(profile_raw)

        memory_dir = _path(_arg("memory_dir") or env.get("CALOSUM_MEMORY_DIR"))
        otlp_jsonl = _path(_arg("otlp_jsonl") or env.get("CALOSUM_OTLP_JSONL"))
        explicit_profile = _arg("infra_profile") is not None or "CALOSUM_INFRA_PROFILE" in env
        if not explicit_profile and (memory_dir is not None or otlp_jsonl is not None):
            profile = InfrastructureProfile.PERSISTENT

        # Vault para APIs externas e credenciais sigilosas
        vault = {}
        for key, value in env.items():
            if key.startswith("CALOSUM_VAULT_"):
                secret_name = key.replace("CALOSUM_VAULT_", "").lower()
                vault[secret_name] = value

        settings = cls(
            profile=profile,
            memory_dir=memory_dir,
            otlp_jsonl=otlp_jsonl,
            vector_db_url=env.get("CALOSUM_VECTORDB_URL"),
            duckdb_path=_path(env.get("CALOSUM_DUCKDB_PATH")),
            bridge_state_dir=_path(_arg("bridge_state_dir") or env.get("CALOSUM_BRIDGE_STATE_DIR")),
            evolution_archive_path=_path(
                _arg("evolution_archive_path") or env.get("CALOSUM_EVOLUTION_ARCHIVE_PATH")
            ),
            awareness_interval_turns=max(
                1,
                int(_arg("awareness_interval_turns") or env.get("CALOSUM_AWARENESS_INTERVAL_TURNS", 1)),
            ),
            api_port=int(env.get("CALOSUM_API_PORT", 8000)),
            otel_collector_endpoint=env.get("CALOSUM_OTEL_COLLECTOR_ENDPOINT"),
            jaeger_ui_url=env.get("CALOSUM_JAEGER_UI_URL"),
            right_hemisphere_endpoint=env.get("CALOSUM_RIGHT_ENDPOINT"),
            right_hemisphere_backend=env.get("CALOSUM_RIGHT_BACKEND"),
            right_model_path=_path(env.get("CALOSUM_RIGHT_MODEL_PATH")),
            right_action_conditioned=_parse_bool(env.get("CALOSUM_RIGHT_ACTION_CONDITIONED"), True),
            right_horizon=max(1, int(env.get("CALOSUM_RIGHT_HORIZON", 4))),
            right_jepars_binary=env.get("CALOSUM_RIGHT_JEPARS_BINARY"),
            left_hemisphere_endpoint=env.get("CALOSUM_LEFT_ENDPOINT"),
            left_hemisphere_api_key=env.get("CALOSUM_LEFT_API_KEY"),
            left_hemisphere_model=env.get("CALOSUM_LEFT_MODEL"),
            left_hemisphere_provider=env.get("CALOSUM_LEFT_PROVIDER"),
            left_hemisphere_reasoning_effort=env.get("CALOSUM_LEFT_REASONING_EFFORT"),
            left_hemisphere_backend=env.get("CALOSUM_LEFT_BACKEND"),
            left_rlm_runtime_command=env.get("CALOSUM_LEFT_RLM_RUNTIME_COMMAND"),
            left_rlm_path=_path(env.get("CALOSUM_LEFT_RLM_PATH")),
            left_rlm_max_depth=max(1, int(env.get("CALOSUM_LEFT_RLM_MAX_DEPTH", 3))),
            left_hemisphere_fallback_endpoint=env.get("CALOSUM_LEFT_FALLBACK_ENDPOINT"),
            left_hemisphere_fallback_api_key=env.get("CALOSUM_LEFT_FALLBACK_API_KEY"),
            left_hemisphere_fallback_model=env.get("CALOSUM_LEFT_FALLBACK_MODEL"),
            left_hemisphere_fallback_provider=env.get("CALOSUM_LEFT_FALLBACK_PROVIDER"),
            left_hemisphere_fallback_reasoning_effort=env.get("CALOSUM_LEFT_FALLBACK_REASONING_EFFORT"),
            perception_model=env.get("CALOSUM_PERCEPTION_MODEL"),
            reason_model=env.get("CALOSUM_REASON_MODEL"),
            reflection_model=env.get("CALOSUM_REFLECTION_MODEL"),
            verifier_model=env.get("CALOSUM_VERIFIER_MODEL"),
            embedding_endpoint=env.get("CALOSUM_EMBEDDING_ENDPOINT"),
            embedding_api_key=env.get("CALOSUM_EMBEDDING_API_KEY"),
            embedding_model=env.get("CALOSUM_EMBEDDING_MODEL"),
            embedding_provider=env.get("CALOSUM_EMBEDDING_PROVIDER"),
            bridge_backend=env.get("CALOSUM_BRIDGE_BACKEND"),
            gea_experience_store_path=_path(env.get("CALOSUM_GEA_EXPERIENCE_STORE_PATH")),
            gea_sharing_enabled=_parse_bool(env.get("CALOSUM_GEA_SHARING_ENABLED"), False),
            telegram_bot_token=env.get("TELEGRAM_BOT_TOKEN"),
            telegram_dm_policy=env.get("CALOSUM_TELEGRAM_DM_POLICY", "open"),
            telegram_allowlist_ids=_parse_csv_list(env.get("CALOSUM_TELEGRAM_ALLOWLIST")),
            vault=vault if vault else None,
            vector_quantization=env.get("CALOSUM_VECTOR_QUANTIZATION", "none"),
            turboquant_bits=max(1, int(env.get("CALOSUM_TURBOQUANT_BITS", 4))),
            qdrant_scalar_quantization=_parse_bool(env.get("CALOSUM_QDRANT_SCALAR_QUANTIZATION"), False),
            mcp_enabled=_parse_bool(env.get("CALOSUM_MCP_ENABLED"), False),
            mcp_servers=_parse_json_mapping(env.get("CALOSUM_MCP_SERVERS")),
            mcp_allowlist=_parse_csv_list(env.get("CALOSUM_MCP_ALLOWLIST")),
        )
        return settings.with_profile_defaults()

    def with_profile_defaults(self) -> "InfrastructureSettings":
        if self.profile == InfrastructureProfile.PERSISTENT:
            return InfrastructureSettings(
                profile=self.profile,
                memory_dir=self.memory_dir or Path(".calosum-runtime/memory"),
                otlp_jsonl=self.otlp_jsonl or Path(".calosum-runtime/telemetry/events.jsonl"),
                vector_db_url=self.vector_db_url,
                duckdb_path=self.duckdb_path or Path(".calosum-runtime/state/semantic.duckdb"),
                bridge_state_dir=self.bridge_state_dir or _default_bridge_state_dir(self),
                evolution_archive_path=self.evolution_archive_path or _default_evolution_archive_path(self),
                awareness_interval_turns=max(1, self.awareness_interval_turns),
                api_port=self.api_port,
                otel_collector_endpoint=self.otel_collector_endpoint,
                jaeger_ui_url=self.jaeger_ui_url,
                right_hemisphere_endpoint=self.right_hemisphere_endpoint,
                right_hemisphere_backend=self.right_hemisphere_backend,
                right_model_path=self.right_model_path,
                right_action_conditioned=self.right_action_conditioned,
                right_horizon=self.right_horizon,
                right_jepars_binary=self.right_jepars_binary,
                left_hemisphere_endpoint=self.left_hemisphere_endpoint,
                left_hemisphere_api_key=self.left_hemisphere_api_key,
                left_hemisphere_model=self.left_hemisphere_model,
                left_hemisphere_provider=self.left_hemisphere_provider,
                left_hemisphere_reasoning_effort=self.left_hemisphere_reasoning_effort,
                left_hemisphere_backend=self.left_hemisphere_backend,
                left_rlm_runtime_command=self.left_rlm_runtime_command,
                left_rlm_path=self.left_rlm_path,
                left_rlm_max_depth=self.left_rlm_max_depth,
                left_hemisphere_fallback_endpoint=self.left_hemisphere_fallback_endpoint,
                left_hemisphere_fallback_api_key=self.left_hemisphere_fallback_api_key,
                left_hemisphere_fallback_model=self.left_hemisphere_fallback_model,
                left_hemisphere_fallback_provider=self.left_hemisphere_fallback_provider,
                left_hemisphere_fallback_reasoning_effort=self.left_hemisphere_fallback_reasoning_effort,
                perception_model=self.perception_model,
                reason_model=self.reason_model,
                reflection_model=self.reflection_model,
                verifier_model=self.verifier_model,
                embedding_endpoint=self.embedding_endpoint,
                embedding_api_key=self.embedding_api_key,
                embedding_model=self.embedding_model,
                embedding_provider=self.embedding_provider,
                bridge_backend=self.bridge_backend,
                gea_experience_store_path=self.gea_experience_store_path or _default_gea_experience_store_path(self),
                gea_sharing_enabled=self.gea_sharing_enabled,
                telegram_bot_token=self.telegram_bot_token,
                telegram_dm_policy=self.telegram_dm_policy,
                telegram_allowlist_ids=list(self.telegram_allowlist_ids or []),
                vault=self.vault,
                vector_quantization=self.vector_quantization,
                turboquant_bits=self.turboquant_bits,
                qdrant_scalar_quantization=self.qdrant_scalar_quantization,
                mcp_enabled=self.mcp_enabled,
                mcp_servers=dict(self.mcp_servers or {}),
                mcp_allowlist=list(self.mcp_allowlist or []),
            )

        if self.profile == InfrastructureProfile.DOCKER:
            return InfrastructureSettings(
                profile=self.profile,
                memory_dir=self.memory_dir or Path("/data/memory"),
                otlp_jsonl=self.otlp_jsonl or Path("/data/telemetry/events.jsonl"),
                vector_db_url=self.vector_db_url or "http://qdrant:6333",
                duckdb_path=self.duckdb_path or Path("/data/state/semantic.duckdb"),
                bridge_state_dir=self.bridge_state_dir or Path("/data/state"),
                evolution_archive_path=self.evolution_archive_path or Path("/data/evolution/archive.jsonl"),
                awareness_interval_turns=max(1, self.awareness_interval_turns),
                api_port=self.api_port,
                otel_collector_endpoint=self.otel_collector_endpoint
                or "http://otel-collector:4318",
                jaeger_ui_url=self.jaeger_ui_url or "http://jaeger:16686",
                right_hemisphere_endpoint=self.right_hemisphere_endpoint
                or "http://right-hemisphere:8081",
                right_hemisphere_backend=self.right_hemisphere_backend,
                right_model_path=self.right_model_path,
                right_action_conditioned=self.right_action_conditioned,
                right_horizon=self.right_horizon,
                right_jepars_binary=self.right_jepars_binary,
                left_hemisphere_endpoint=self.left_hemisphere_endpoint,
                left_hemisphere_api_key=self.left_hemisphere_api_key,
                left_hemisphere_model=self.left_hemisphere_model,
                left_hemisphere_provider=self.left_hemisphere_provider,
                left_hemisphere_reasoning_effort=self.left_hemisphere_reasoning_effort,
                left_hemisphere_backend=self.left_hemisphere_backend,
                left_rlm_runtime_command=self.left_rlm_runtime_command,
                left_rlm_path=self.left_rlm_path,
                left_rlm_max_depth=self.left_rlm_max_depth,
                left_hemisphere_fallback_endpoint=self.left_hemisphere_fallback_endpoint,
                left_hemisphere_fallback_api_key=self.left_hemisphere_fallback_api_key,
                left_hemisphere_fallback_model=self.left_hemisphere_fallback_model,
                left_hemisphere_fallback_provider=self.left_hemisphere_fallback_provider,
                left_hemisphere_fallback_reasoning_effort=self.left_hemisphere_fallback_reasoning_effort,
                perception_model=self.perception_model,
                reason_model=self.reason_model,
                reflection_model=self.reflection_model,
                verifier_model=self.verifier_model,
                embedding_endpoint=self.embedding_endpoint,
                embedding_api_key=self.embedding_api_key,
                embedding_model=self.embedding_model,
                embedding_provider=self.embedding_provider,
                bridge_backend=self.bridge_backend,
                gea_experience_store_path=self.gea_experience_store_path or Path("/data/evolution/gea.sqlite"),
                gea_sharing_enabled=self.gea_sharing_enabled,
                telegram_bot_token=self.telegram_bot_token,
                telegram_dm_policy=self.telegram_dm_policy,
                telegram_allowlist_ids=list(self.telegram_allowlist_ids or []),
                vault=self.vault,
                vector_quantization=self.vector_quantization,
                turboquant_bits=self.turboquant_bits,
                qdrant_scalar_quantization=self.qdrant_scalar_quantization,
                mcp_enabled=self.mcp_enabled,
                mcp_servers=dict(self.mcp_servers or {}),
                mcp_allowlist=list(self.mcp_allowlist or []),
            )

        return replace(
            self,
            awareness_interval_turns=max(1, self.awareness_interval_turns),
            telegram_dm_policy=self.telegram_dm_policy or "open",
            telegram_allowlist_ids=list(self.telegram_allowlist_ids or []),
            mcp_enabled=self.mcp_enabled,
            mcp_servers=dict(self.mcp_servers or {}),
            mcp_allowlist=list(self.mcp_allowlist or []),
        )


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
    settings: InfrastructureSettings,
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
        settings.profile == InfrastructureProfile.EPHEMERAL
        and settings.memory_dir is None
        and settings.otlp_jsonl is None
    )


def with_local_persistence_defaults(settings: InfrastructureSettings) -> InfrastructureSettings:
    return replace(settings, profile=InfrastructureProfile.PERSISTENT).with_profile_defaults()


def _default_bridge_state_dir(settings: InfrastructureSettings) -> Path:
    if settings.memory_dir is not None:
        return settings.memory_dir.parent / "state"
    if settings.duckdb_path is not None:
        return settings.duckdb_path.parent
    return Path(".calosum-runtime/state")


def _default_evolution_archive_path(settings: InfrastructureSettings) -> Path:
    if settings.memory_dir is not None:
        return settings.memory_dir.parent / "evolution" / "archive.jsonl"
    if settings.otlp_jsonl is not None:
        telemetry_parent = settings.otlp_jsonl.parent
        if telemetry_parent.name == "telemetry":
            return telemetry_parent.parent / "evolution" / "archive.jsonl"
        return telemetry_parent / "evolution" / "archive.jsonl"
    return Path(".calosum-runtime/evolution/archive.jsonl")


def _default_gea_experience_store_path(settings: InfrastructureSettings) -> Path:
    if settings.memory_dir is not None:
        return settings.memory_dir.parent / "evolution" / "gea.sqlite"
    if settings.otlp_jsonl is not None:
        telemetry_parent = settings.otlp_jsonl.parent
        if telemetry_parent.name == "telemetry":
            return telemetry_parent.parent / "evolution" / "gea.sqlite"
        return telemetry_parent / "evolution" / "gea.sqlite"
    return Path(".calosum-runtime/evolution/gea.sqlite")
