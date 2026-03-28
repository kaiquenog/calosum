from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from calosum.adapters.action_runtime import ConcreteActionRuntime
from calosum.adapters.llm_qwen import QwenAdapterConfig, QwenLeftHemisphereAdapter
from calosum.adapters.memory_qdrant import QdrantAdapterConfig, QdrantDualMemoryAdapter
from calosum.adapters.right_hemisphere_hf import HuggingFaceRightHemisphereAdapter
from calosum.domain.memory import DualMemorySystem
from calosum.domain.orchestrator import CalosumAgent
from calosum.domain.persistent_memory import PersistentDualMemorySystem
from calosum.domain.right_hemisphere import RightHemisphereJEPA
from calosum.bootstrap.settings import InfrastructureProfile, InfrastructureSettings
from calosum.domain.telemetry import CognitiveTelemetryBus, InMemoryTelemetrySink, OTLPJsonlTelemetrySink

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CalosumAgentBuilder:
    settings: InfrastructureSettings
    _last_right_hemisphere_backend: str | None = field(default=None, init=False, repr=False)

    def build(self) -> CalosumAgent:
        left_hemisphere = None
        if self.settings.left_hemisphere_endpoint:
            left_hemisphere = QwenLeftHemisphereAdapter(
                QwenAdapterConfig(
                    api_url=self.settings.left_hemisphere_endpoint,
                    api_key=self.settings.left_hemisphere_api_key or "empty",
                    model_name=self.settings.left_hemisphere_model or "Qwen/Qwen-3.5-9B-Instruct"
                )
            )
        else:
            left_hemisphere = QwenLeftHemisphereAdapter()

        right_hemisphere = self.build_right_hemisphere()

        return CalosumAgent(
            right_hemisphere=right_hemisphere,
            left_hemisphere=left_hemisphere,
            action_runtime=ConcreteActionRuntime(vault=self.settings.vault),
            memory_system=self.build_memory_system(),
            telemetry_bus=self.build_telemetry_bus(),
        )

    def build_right_hemisphere(self):
        try:
            right_hemisphere = HuggingFaceRightHemisphereAdapter()
        except Exception as exc:
            logger.warning(
                "Falling back to heuristic right hemisphere adapter because the optional "
                "HuggingFace stack is unavailable: %s",
                exc,
            )
            self._last_right_hemisphere_backend = "heuristic_jepa_fallback"
            return RightHemisphereJEPA()

        self._last_right_hemisphere_backend = "huggingface_embeddings"
        return right_hemisphere

    def build_memory_system(self):
        if self.settings.vector_db_url:
            return QdrantDualMemoryAdapter(QdrantAdapterConfig(url=self.settings.vector_db_url))
        
        if self.settings.profile in {
            InfrastructureProfile.PERSISTENT,
            InfrastructureProfile.DOCKER,
        }:
            if self.settings.memory_dir is None:
                raise ValueError("persistent profiles require a resolved memory_dir")
            return PersistentDualMemorySystem.from_directory(self.settings.memory_dir)
        return DualMemorySystem()

    def build_telemetry_bus(self) -> CognitiveTelemetryBus:
        if self.settings.otlp_jsonl is not None:
            return CognitiveTelemetryBus(OTLPJsonlTelemetrySink(self.settings.otlp_jsonl))
        return CognitiveTelemetryBus(InMemoryTelemetrySink())

    def describe(self) -> dict[str, Any]:
        return {
            "pattern": "ports_and_adapters_with_builder_factory",
            "profile": self.settings.profile.value,
            "memory_backend": self._memory_backend_name(),
            "telemetry_backend": self._telemetry_backend_name(),
            "right_hemisphere_backend": self._right_hemisphere_backend_name(),
            "left_hemisphere_backend": "qwen3.5_9b_adapter",
            "action_runtime": "concrete_adapter",
            "memory_dir": str(self.settings.memory_dir) if self.settings.memory_dir else None,
            "otlp_jsonl": str(self.settings.otlp_jsonl) if self.settings.otlp_jsonl else None,
            "vector_db_url": self.settings.vector_db_url,
            "duckdb_path": str(self.settings.duckdb_path) if self.settings.duckdb_path else None,
            "otel_collector_endpoint": self.settings.otel_collector_endpoint,
            "jaeger_ui_url": self.settings.jaeger_ui_url,
            "right_hemisphere_endpoint": self.settings.right_hemisphere_endpoint,
            "left_hemisphere_endpoint": self.settings.left_hemisphere_endpoint,
        }

    def _memory_backend_name(self) -> str:
        if self.settings.vector_db_url:
            return "qdrant_vector_memory"
        if self.settings.profile in {
            InfrastructureProfile.PERSISTENT,
            InfrastructureProfile.DOCKER,
        }:
            return "persistent_jsonl"
        return "in_memory"

    def _telemetry_backend_name(self) -> str:
        if self.settings.otlp_jsonl is not None:
            return "otlp_jsonl"
        return "in_memory"

    def _right_hemisphere_backend_name(self) -> str:
        return self._last_right_hemisphere_backend or "optional_huggingface_with_fallback"
