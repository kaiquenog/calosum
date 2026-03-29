from __future__ import annotations

import logging
from dataclasses import dataclass, field
import importlib.util
from pathlib import Path
from typing import Any

from calosum.adapters.active_inference import ActiveInferenceRightHemisphereAdapter
from calosum.adapters.action_runtime import ConcreteActionRuntime
from calosum.adapters.knowledge_graph_nanorag import NanoGraphRAGKnowledgeGraphStore
from calosum.adapters.llm_failover import ResilientLeftHemisphereAdapter
from calosum.adapters.llm_qwen import QwenAdapterConfig, QwenLeftHemisphereAdapter
from calosum.adapters.memory_qdrant import QdrantAdapterConfig, QdrantDualMemoryAdapter
from calosum.adapters.right_hemisphere_hf import HuggingFaceRightHemisphereAdapter
from calosum.adapters.text_embeddings import TextEmbeddingAdapter, TextEmbeddingAdapterConfig
from calosum.domain.memory import DualMemorySystem
from calosum.domain.orchestrator import CalosumAgent
from calosum.domain.persistent_memory import (
    JsonlEpisodicStore,
    JsonlSemanticStore,
    PersistentDualMemorySystem,
)
from calosum.domain.right_hemisphere import RightHemisphereJEPA
from calosum.bootstrap.settings import InfrastructureProfile, InfrastructureSettings
from calosum.domain.telemetry import CognitiveTelemetryBus, InMemoryTelemetrySink, OTLPJsonlTelemetrySink
from calosum.shared.types import CapabilityDescriptor, ComponentHealth, ModelDescriptor, ToolDescriptor

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CalosumAgentBuilder:
    settings: InfrastructureSettings
    _last_right_hemisphere_backend: str | None = field(default=None, init=False, repr=False)
    _last_left_hemisphere_backend: str | None = field(default=None, init=False, repr=False)
    _last_embedding_backend: str | None = field(default=None, init=False, repr=False)
    _last_knowledge_graph_backend: str | None = field(default=None, init=False, repr=False)

    def build(self) -> CalosumAgent:
        left_hemisphere = self.build_left_hemisphere()

        right_hemisphere = self.build_right_hemisphere()

        from calosum.domain.bridge import CognitiveTokenizer
        from calosum.adapters.bridge_store import LocalBridgeStateStore
        tokenizer = CognitiveTokenizer(store=LocalBridgeStateStore())
        
        action_runtime = ConcreteActionRuntime(vault=self.settings.vault)
        memory_system = self.build_memory_system()
        telemetry_bus = self.build_telemetry_bus()

        capability_snapshot = self.build_capability_snapshot(action_runtime)

        return CalosumAgent(
            right_hemisphere=right_hemisphere,
            tokenizer=tokenizer,
            left_hemisphere=left_hemisphere,
            action_runtime=action_runtime,
            memory_system=memory_system,
            telemetry_bus=telemetry_bus,
            capability_snapshot=capability_snapshot,
        )

    def build_left_hemisphere(self):
        primary = self._build_single_left_hemisphere(
            endpoint=self.settings.left_hemisphere_endpoint,
            api_key=self.settings.left_hemisphere_api_key,
            model=self.settings.left_hemisphere_model,
            provider=self.settings.left_hemisphere_provider,
            reasoning_effort=self.settings.left_hemisphere_reasoning_effort,
        )

        if self.settings.left_hemisphere_fallback_endpoint:
            fallback = self._build_single_left_hemisphere(
                endpoint=self.settings.left_hemisphere_fallback_endpoint,
                api_key=self.settings.left_hemisphere_fallback_api_key,
                model=self.settings.left_hemisphere_fallback_model or self.settings.left_hemisphere_model,
                provider=self.settings.left_hemisphere_fallback_provider,
                reasoning_effort=(
                    self.settings.left_hemisphere_fallback_reasoning_effort
                    or self.settings.left_hemisphere_reasoning_effort
                ),
            )
            self._last_left_hemisphere_backend = "resilient_failover_adapter"
            return ResilientLeftHemisphereAdapter([primary, fallback])

        self._last_left_hemisphere_backend = self._left_hemisphere_backend_name_from_settings()
        return primary

    def build_right_hemisphere(self):
        try:
            base_adapter = HuggingFaceRightHemisphereAdapter()
        except Exception as exc:
            logger.warning(
                "Falling back to heuristic right hemisphere adapter because the optional "
                "HuggingFace stack is unavailable: %s",
                exc,
            )
            self._last_right_hemisphere_backend = "active_inference_heuristic_fallback"
            return ActiveInferenceRightHemisphereAdapter(RightHemisphereJEPA())

        self._last_right_hemisphere_backend = "active_inference_huggingface"
        return ActiveInferenceRightHemisphereAdapter(base_adapter)

    def build_memory_system(self):
        from calosum.adapters.night_trainer import LocalDatasetExporter
        from calosum.domain.memory import SleepModeConsolidator
        
        exporter = LocalDatasetExporter(Path(".calosum-runtime/nightly_data"))
        graph_store = self.build_graph_store()

        if self.settings.vector_db_url:
            embedder = self.build_text_embedder()
            return QdrantDualMemoryAdapter(
                QdrantAdapterConfig(url=self.settings.vector_db_url),
                embedder=embedder,
                exporter=exporter,
                graph_store=graph_store,
            )
        
        if self.settings.profile in {
            InfrastructureProfile.PERSISTENT,
            InfrastructureProfile.DOCKER,
        }:
            if self.settings.memory_dir is None:
                raise ValueError("persistent profiles require a resolved memory_dir")
            return PersistentDualMemorySystem(
                episodic_store=JsonlEpisodicStore(self.settings.memory_dir / "episodic.jsonl"),
                semantic_store=JsonlSemanticStore(self.settings.memory_dir / "semantic_rules.jsonl"),
                graph_store=graph_store,
                consolidator=SleepModeConsolidator(exporter=exporter)
            )
        return DualMemorySystem(
            graph_store=graph_store,
            consolidator=SleepModeConsolidator(exporter=exporter),
        )

    def build_graph_store(self) -> NanoGraphRAGKnowledgeGraphStore:
        storage_path: Path | None = None
        if self.settings.memory_dir is not None:
            storage_path = self.settings.memory_dir / "knowledge_graph.jsonl"
        elif self.settings.duckdb_path is not None:
            storage_path = self.settings.duckdb_path.parent / "knowledge_graph.jsonl"

        store = NanoGraphRAGKnowledgeGraphStore(storage_path=storage_path)
        self._last_knowledge_graph_backend = store.backend_name
        return store

    def build_telemetry_bus(self) -> CognitiveTelemetryBus:
        if self.settings.otlp_jsonl is not None:
            return CognitiveTelemetryBus(OTLPJsonlTelemetrySink(self.settings.otlp_jsonl))
        return CognitiveTelemetryBus(InMemoryTelemetrySink())

    def build_text_embedder(self) -> TextEmbeddingAdapter:
        endpoint = self.settings.embedding_endpoint
        api_key = self.settings.embedding_api_key
        provider = self.settings.embedding_provider
        model = self.settings.embedding_model

        if endpoint is None and self._left_endpoint_supports_embeddings():
            endpoint = self.settings.left_hemisphere_endpoint
            api_key = api_key or self.settings.left_hemisphere_api_key
            provider = provider or self._default_embedding_provider()
            model = model or "text-embedding-3-small"

        embedder = TextEmbeddingAdapter(
            TextEmbeddingAdapterConfig(
                provider=provider or "auto",
                api_url=endpoint,
                api_key=api_key,
                model_name=model or "text-embedding-3-small",
            )
        )
        self._last_embedding_backend = embedder.backend_name()
        return embedder

    def _build_single_left_hemisphere(
        self,
        *,
        endpoint: str | None,
        api_key: str | None,
        model: str | None,
        provider: str | None,
        reasoning_effort: str | None,
    ) -> QwenLeftHemisphereAdapter:
        if endpoint:
            return QwenLeftHemisphereAdapter(
                QwenAdapterConfig(
                    api_url=endpoint,
                    api_key=api_key or "empty",
                    model_name=model or "Qwen/Qwen-3.5-9B-Instruct",
                    provider=provider or "auto",
                    reasoning_effort=reasoning_effort,
                )
            )
        return QwenLeftHemisphereAdapter()

    def build_capability_snapshot(self, action_runtime: ConcreteActionRuntime | None = None) -> CapabilityDescriptor:
        right_model = ModelDescriptor(
            provider="local",
            model_name="jepa",
            backend=self._right_hemisphere_backend_name(),
            health=ComponentHealth.HEALTHY,
        )

        left_model = ModelDescriptor(
            provider=self.settings.left_hemisphere_provider or "auto",
            model_name=self.settings.left_hemisphere_model or "Qwen/Qwen-3.5-9B-Instruct",
            backend=self._left_hemisphere_backend_name(),
            health=ComponentHealth.HEALTHY,
        )

        embedding_model = None
        if self._embedding_backend_name():
            embedding_model = ModelDescriptor(
                provider=self._derived_embedding_provider() or "auto",
                model_name=self._derived_embedding_model() or "auto",
                backend=self._embedding_backend_name() or "auto",
                health=ComponentHealth.HEALTHY,
            )

        kg_model = ModelDescriptor(
            provider="local",
            model_name="nanorag",
            backend=self._knowledge_graph_backend_name(),
            health=ComponentHealth.HEALTHY,
        )

        tools = []
        if action_runtime:
            tools = action_runtime.get_registered_tools()

        return CapabilityDescriptor(
            right_hemisphere=right_model,
            left_hemisphere=left_model,
            embeddings=embedding_model,
            knowledge_graph=kg_model,
            tools=tools,
            health=ComponentHealth.HEALTHY,
        )

    def describe(self) -> dict[str, Any]:
        from dataclasses import asdict
        snapshot = self.build_capability_snapshot()
        return {
            "pattern": "ports_and_adapters_with_builder_factory",
            "profile": self.settings.profile.value,
            "capabilities": asdict(snapshot),
            "memory_backend": self._memory_backend_name(),
            "telemetry_backend": self._telemetry_backend_name(),
            "right_hemisphere_backend": self._right_hemisphere_backend_name(),
            "left_hemisphere_backend": self._left_hemisphere_backend_name(),
            "action_runtime": "concrete_adapter",
            "memory_dir": str(self.settings.memory_dir) if self.settings.memory_dir else None,
            "otlp_jsonl": str(self.settings.otlp_jsonl) if self.settings.otlp_jsonl else None,
            "vector_db_url": self.settings.vector_db_url,
            "duckdb_path": str(self.settings.duckdb_path) if self.settings.duckdb_path else None,
            "otel_collector_endpoint": self.settings.otel_collector_endpoint,
            "jaeger_ui_url": self.settings.jaeger_ui_url,
            "right_hemisphere_endpoint": self.settings.right_hemisphere_endpoint,
            "left_hemisphere_endpoint": self.settings.left_hemisphere_endpoint,
            "left_hemisphere_model": self.settings.left_hemisphere_model,
            "left_hemisphere_provider": self.settings.left_hemisphere_provider,
            "left_hemisphere_reasoning_effort": self.settings.left_hemisphere_reasoning_effort,
            "left_hemisphere_failover_enabled": bool(self.settings.left_hemisphere_fallback_endpoint),
            "left_hemisphere_fallback_endpoint": self.settings.left_hemisphere_fallback_endpoint,
            "left_hemisphere_fallback_model": self.settings.left_hemisphere_fallback_model,
            "left_hemisphere_fallback_provider": self.settings.left_hemisphere_fallback_provider,
            "knowledge_graph_backend": self._knowledge_graph_backend_name(),
            "embedding_backend": self._embedding_backend_name(),
            "embedding_endpoint": self.settings.embedding_endpoint or self._derived_embedding_endpoint(),
            "embedding_model": self.settings.embedding_model or self._derived_embedding_model(),
            "embedding_provider": self.settings.embedding_provider or self._derived_embedding_provider(),
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
        return self._last_right_hemisphere_backend or "active_inference_with_optional_huggingface"

    def _left_hemisphere_backend_name(self) -> str:
        return self._last_left_hemisphere_backend or self._left_hemisphere_backend_name_from_settings()

    def _embedding_backend_name(self) -> str | None:
        if not self.settings.vector_db_url:
            return None
        return self._last_embedding_backend or self._derived_embedding_provider() or "auto"

    def _knowledge_graph_backend_name(self) -> str:
        if self._last_knowledge_graph_backend:
            return self._last_knowledge_graph_backend
        if importlib.util.find_spec("nano_graphrag"):
            return "nanorag_compatible_networkx"
        return "networkx_graph_rag_fallback"

    def _left_hemisphere_backend_name_from_settings(self) -> str:
        if self.settings.left_hemisphere_fallback_endpoint:
            return "resilient_failover_adapter"
        endpoint = (self.settings.left_hemisphere_endpoint or "").lower()
        provider = (self.settings.left_hemisphere_provider or "auto").lower()

        if provider in {"openai_responses", "openai", "responses"}:
            return "openai_responses_adapter"
        if provider in {"openai_chat", "chat"}:
            return "openai_chat_adapter"
        if "api.openai.com" in endpoint:
            if endpoint.rstrip("/").endswith("/chat/completions"):
                return "openai_chat_adapter"
            return "openai_responses_adapter"
        if endpoint:
            return "openai_compatible_chat_adapter"
        return "openai_compatible_chat_adapter_default"

    def _left_endpoint_supports_embeddings(self) -> bool:
        endpoint = (self.settings.left_hemisphere_endpoint or "").lower()
        provider = (self.settings.left_hemisphere_provider or "").lower()
        if "api.openai.com" in endpoint:
            return True
        return provider in {"openai", "openai_responses", "responses"}

    def _default_embedding_provider(self) -> str:
        endpoint = (self.settings.left_hemisphere_endpoint or "").lower()
        if "api.openai.com" in endpoint:
            return "openai"
        return "openai_compatible"

    def _derived_embedding_endpoint(self) -> str | None:
        if self.settings.embedding_endpoint:
            return self.settings.embedding_endpoint
        if self._left_endpoint_supports_embeddings():
            return self.settings.left_hemisphere_endpoint
        return None

    def _derived_embedding_model(self) -> str | None:
        if self.settings.embedding_model:
            return self.settings.embedding_model
        if self._left_endpoint_supports_embeddings():
            return "text-embedding-3-small"
        return None

    def _derived_embedding_provider(self) -> str | None:
        if self.settings.embedding_provider:
            return self.settings.embedding_provider
        if self._left_endpoint_supports_embeddings():
            return self._default_embedding_provider()
        return None
