from __future__ import annotations

import os
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Mapping


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
    api_port: int = 8000
    otel_collector_endpoint: str | None = None
    jaeger_ui_url: str | None = None
    right_hemisphere_endpoint: str | None = None
    left_hemisphere_endpoint: str | None = None
    left_hemisphere_api_key: str | None = None
    left_hemisphere_model: str | None = None
    left_hemisphere_provider: str | None = None
    left_hemisphere_reasoning_effort: str | None = None
    vault: dict[str, str] | None = None

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
            api_port=int(env.get("CALOSUM_API_PORT", 8000)),
            otel_collector_endpoint=env.get("CALOSUM_OTEL_COLLECTOR_ENDPOINT"),
            jaeger_ui_url=env.get("CALOSUM_JAEGER_UI_URL"),
            right_hemisphere_endpoint=env.get("CALOSUM_RIGHT_ENDPOINT"),
            left_hemisphere_endpoint=env.get("CALOSUM_LEFT_ENDPOINT"),
            left_hemisphere_api_key=env.get("CALOSUM_LEFT_API_KEY"),
            left_hemisphere_model=env.get("CALOSUM_LEFT_MODEL"),
            left_hemisphere_provider=env.get("CALOSUM_LEFT_PROVIDER"),
            left_hemisphere_reasoning_effort=env.get("CALOSUM_LEFT_REASONING_EFFORT"),
            vault=vault if vault else None,
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
                api_port=self.api_port,
                otel_collector_endpoint=self.otel_collector_endpoint,
                jaeger_ui_url=self.jaeger_ui_url,
                right_hemisphere_endpoint=self.right_hemisphere_endpoint,
                left_hemisphere_endpoint=self.left_hemisphere_endpoint,
                left_hemisphere_api_key=self.left_hemisphere_api_key,
                left_hemisphere_model=self.left_hemisphere_model,
                left_hemisphere_provider=self.left_hemisphere_provider,
                left_hemisphere_reasoning_effort=self.left_hemisphere_reasoning_effort,
                vault=self.vault,
            )

        if self.profile == InfrastructureProfile.DOCKER:
            return InfrastructureSettings(
                profile=self.profile,
                memory_dir=self.memory_dir or Path("/data/memory"),
                otlp_jsonl=self.otlp_jsonl or Path("/data/telemetry/events.jsonl"),
                vector_db_url=self.vector_db_url or "http://qdrant:6333",
                duckdb_path=self.duckdb_path or Path("/data/state/semantic.duckdb"),
                api_port=self.api_port,
                otel_collector_endpoint=self.otel_collector_endpoint
                or "http://otel-collector:4318",
                jaeger_ui_url=self.jaeger_ui_url or "http://jaeger:16686",
                right_hemisphere_endpoint=self.right_hemisphere_endpoint
                or "http://right-hemisphere:8081",
                left_hemisphere_endpoint=self.left_hemisphere_endpoint,
                left_hemisphere_api_key=self.left_hemisphere_api_key,
                left_hemisphere_model=self.left_hemisphere_model,
                left_hemisphere_provider=self.left_hemisphere_provider,
                left_hemisphere_reasoning_effort=self.left_hemisphere_reasoning_effort,
                vault=self.vault,
            )

        return self


def _path(value: str | os.PathLike[str] | None) -> Path | None:
    if value is None or value == "":
        return None
    return Path(value)


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
