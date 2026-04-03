from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping

from calosum.bootstrap.infrastructure.helpers import (
    _default_bridge_state_dir,
    _default_evolution_archive_path,
    _default_gea_experience_store_path,
    _parse_bool,
    _parse_csv_list,
    _parse_json_mapping,
    _path,
    should_enable_local_persistence_defaults,
    with_local_persistence_defaults,
)
class InfrastructureProfile(StrEnum):
    EPHEMERAL = "ephemeral"
    PERSISTENT = "persistent"
    DOCKER = "docker"


class RuntimeDependencyMode(StrEnum):
    AUTO = "auto"
    LOCAL = "local"
    API = "api"


class CalosumMode(StrEnum):
    API = "api"
    LOCAL = "local"

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
    dependency_mode: RuntimeDependencyMode = RuntimeDependencyMode.AUTO
    mode: CalosumMode = CalosumMode.API

    @classmethod
    def from_sources(
        cls,
        *,
        args: object | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> "InfrastructureSettings":
        env = dict(environ if environ is not None else os.environ)
        
        # Parseia o .env localmente apenas se não estiver em modo de ignorar (testes)
        ignore_dotenv = _parse_bool(env.get("CALOSUM_IGNORE_DOTENV"), False)
        if not ignore_dotenv:
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
        dependency_mode_raw = env.get(
            "CALOSUM_DEPENDENCY_MODE",
            env.get("CALOSUM_INSTALL_MODE", RuntimeDependencyMode.AUTO.value),
        )
        dependency_mode = RuntimeDependencyMode(dependency_mode_raw.strip().lower())
        mode_raw = env.get("CALOSUM_MODE")
        if mode_raw:
            mode = CalosumMode(mode_raw.strip().lower())
        elif dependency_mode == RuntimeDependencyMode.LOCAL:
            mode = CalosumMode.LOCAL
        else:
            mode = CalosumMode.API

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
            dependency_mode=dependency_mode,
            mode=mode,
        )
        profile_enabled_settings = settings.with_profile_defaults()
        profile_enabled_settings.validate_consistency()
        return profile_enabled_settings

    def validate_consistency(self) -> None:
        missing_local_deps = _missing_local_dependency_stack()
        local_features = _configured_local_features(self)
        if self.mode == CalosumMode.API and self.dependency_mode == RuntimeDependencyMode.LOCAL:
            raise RuntimeError(
                "INCOHERENT CONFIGURATION: CALOSUM_MODE=api conflicts with "
                "CALOSUM_DEPENDENCY_MODE=local. Use API mode + dependency_mode=api/auto."
            )
        if self.mode == CalosumMode.LOCAL and self.dependency_mode == RuntimeDependencyMode.API:
            raise RuntimeError(
                "INCOHERENT CONFIGURATION: CALOSUM_MODE=local conflicts with "
                "CALOSUM_DEPENDENCY_MODE=api. Use local mode + dependency_mode=local/auto."
            )

        if self.dependency_mode == RuntimeDependencyMode.LOCAL and missing_local_deps:
            raise RuntimeError(
                "INCOHERENT CONFIGURATION: CALOSUM_DEPENDENCY_MODE=local requires optional local dependencies "
                f"({', '.join(missing_local_deps)}), but they are not installed. "
                "Install with 'pip install calosum[local]'."
            )

        if self.dependency_mode == RuntimeDependencyMode.API and local_features:
            raise RuntimeError(
                "INCOHERENT CONFIGURATION: CALOSUM_DEPENDENCY_MODE=api is API-only, but local-only runtime options "
                f"were configured ({', '.join(local_features)}). "
                "Use API-compatible settings only, or switch to local mode with "
                "'CALOSUM_DEPENDENCY_MODE=local' and install 'pip install calosum[local]'."
            )
        if self.mode == CalosumMode.API and local_features:
            raise RuntimeError(
                "INCOHERENT CONFIGURATION: CALOSUM_MODE=api is API-only, but local-only runtime options "
                f"were configured ({', '.join(local_features)}). "
                "Use CALOSUM_MODE=local for local model stacks."
            )

        if local_features and missing_local_deps:
            raise RuntimeError(
                "INCOHERENT CONFIGURATION: local-only runtime options were configured "
                f"({', '.join(local_features)}), but missing optional dependencies were detected "
                f"({', '.join(missing_local_deps)}). "
                "Install with 'pip install calosum[local]' or remove local-only options."
            )

    def with_profile_defaults(self) -> "InfrastructureSettings":
        if self.profile == InfrastructureProfile.PERSISTENT:
            return InfrastructureSettings(
                profile=self.profile,
                memory_dir=self.memory_dir or Path(".calosum-runtime/memory"),
                otlp_jsonl=self.otlp_jsonl or Path(".calosum-runtime/telemetry/events.jsonl"),
                vector_db_url=self.vector_db_url,
                duckdb_path=self.duckdb_path,
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
                dependency_mode=self.dependency_mode,
                mode=self.mode,
            )

        if self.profile == InfrastructureProfile.DOCKER:
            return InfrastructureSettings(
                profile=self.profile,
                memory_dir=self.memory_dir or Path("/data/memory"),
                otlp_jsonl=self.otlp_jsonl or Path("/data/telemetry/events.jsonl"),
                vector_db_url=self.vector_db_url or "http://qdrant:6333",
                duckdb_path=self.duckdb_path,
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
                dependency_mode=self.dependency_mode,
                mode=self.mode,
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
def _missing_local_dependency_stack() -> list[str]:
    required = ("torch", "peft", "transformers")
    return [package for package in required if importlib.util.find_spec(package) is None]
def _configured_local_features(settings: InfrastructureSettings) -> list[str]:
    features: list[str] = []
    right_backend = (settings.right_hemisphere_backend or "").strip().lower()
    if right_backend in {"huggingface", "vjepa21", "vljepa", "trained_jepa"}:
        features.append(f"right_hemisphere_backend={right_backend}")
    if settings.vector_quantization.strip().lower() == "turboquant":
        features.append("vector_quantization=turboquant")
    return features
