from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
import importlib.util
from pathlib import Path
from typing import Any

from calosum.adapters.active_inference import ActiveInferenceRightHemisphereAdapter
from calosum.adapters.action_runtime import ConcreteActionRuntime
from calosum.adapters.llm_failover import ResilientLeftHemisphereAdapter
from calosum.adapters.llm_qwen import QwenAdapterConfig, QwenLeftHemisphereAdapter
from calosum.adapters.telemetry_otlp import OTLPHTTPTraceSink
from calosum.adapters.text_embeddings import TextEmbeddingAdapter, TextEmbeddingAdapterConfig
from calosum.domain.evolution import JsonlEvolutionArchive
from calosum.domain.memory import DualMemorySystem, InMemorySemanticGraphStore
from calosum.domain.orchestrator import CalosumAgent, CalosumAgentConfig
from calosum.domain.persistent_memory import (
    JsonlEpisodicStore,
    JsonlSemanticStore,
    PersistentDualMemorySystem,
)
from calosum.domain.right_hemisphere import RightHemisphereJEPA
from calosum.bootstrap.settings import InfrastructureProfile, InfrastructureSettings
from calosum.domain.telemetry import CognitiveTelemetryBus, CompositeTelemetrySink, InMemoryTelemetrySink, OTLPJsonlTelemetrySink
from calosum.shared.types import CapabilityDescriptor, ComponentHealth, ModelDescriptor

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CalosumAgentBuilder:
    settings: InfrastructureSettings
    _last_right_hemisphere_backend: str | None = field(default=None, init=False, repr=False)
    _last_left_hemisphere_backend: str | None = field(default=None, init=False, repr=False)
    _last_embedding_backend: str | None = field(default=None, init=False, repr=False)
    _last_knowledge_graph_backend: str | None = field(default=None, init=False, repr=False)
    _last_right_hemisphere_model_name: str | None = field(default=None, init=False, repr=False)

    def build(self) -> CalosumAgent:
        left_hemisphere = self.build_left_hemisphere()
        right_hemisphere = self.build_right_hemisphere()
        from calosum.domain.bridge import CognitiveTokenizer
        from calosum.adapters.bridge_store import LocalBridgeStateStore
        bridge_store = None
        if self.settings.bridge_state_dir is not None:
            bridge_store = LocalBridgeStateStore(base_dir=self.settings.bridge_state_dir)
        tokenizer = CognitiveTokenizer(store=bridge_store)
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
            config=self._build_agent_config(),
            capability_snapshot=capability_snapshot,
            evolution_archive=JsonlEvolutionArchive(self.settings.evolution_archive_path),
        )

    def build_left_hemisphere(self):
        primary = self._build_single_left_hemisphere(
            endpoint=self.settings.left_hemisphere_endpoint,
            api_key=self.settings.left_hemisphere_api_key,
            model=self._reason_model_name(),
            provider=self.settings.left_hemisphere_provider,
            reasoning_effort=self.settings.left_hemisphere_reasoning_effort,
        )
        if self.settings.left_hemisphere_fallback_endpoint:
            fallback = self._build_single_left_hemisphere(
                endpoint=self.settings.left_hemisphere_fallback_endpoint,
                api_key=self.settings.left_hemisphere_fallback_api_key,
                model=self.settings.left_hemisphere_fallback_model or self._reason_model_name(),
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
        requested_model = self.settings.perception_model
        if requested_model and requested_model.lower() == "jepa":
            self._last_right_hemisphere_backend = "active_inference_jepa_policy"
            self._last_right_hemisphere_model_name = "jepa"
            return ActiveInferenceRightHemisphereAdapter(RightHemisphereJEPA())

        try:
            from calosum.adapters.right_hemisphere_hf import (
                HuggingFaceRightHemisphereAdapter,
                HuggingFaceRightHemisphereConfig,
            )
            config = None
            if requested_model and requested_model.lower() != "jepa":
                config = HuggingFaceRightHemisphereConfig(embedding_model_name=requested_model)
                self._last_right_hemisphere_model_name = requested_model
            base_adapter = HuggingFaceRightHemisphereAdapter(config)
        except Exception as exc:
            logger.warning(
                "Falling back to heuristic right hemisphere adapter because the optional "
                "HuggingFace stack is unavailable: %s",
                exc,
            )
            self._last_right_hemisphere_backend = "active_inference_heuristic_fallback"
            self._last_right_hemisphere_model_name = "jepa"
            return ActiveInferenceRightHemisphereAdapter(RightHemisphereJEPA())

        self._last_right_hemisphere_backend = "active_inference_huggingface"
        if self._last_right_hemisphere_model_name is None:
            self._last_right_hemisphere_model_name = getattr(
                getattr(base_adapter, "config", None),
                "embedding_model_name",
                requested_model or "paraphrase-multilingual-MiniLM-L12-v2",
            )
        return ActiveInferenceRightHemisphereAdapter(base_adapter)

    def build_memory_system(self):
        from calosum.adapters.night_trainer import LocalDatasetExporter
        from calosum.domain.memory import SleepModeConsolidator
        exporter = LocalDatasetExporter(self._runtime_root() / "nightly_data")
        graph_store = self.build_graph_store()
        if self.settings.vector_db_url:
            try:
                from calosum.adapters.memory_qdrant import QdrantAdapterConfig, QdrantDualMemoryAdapter
                embedder = self.build_text_embedder()
                return QdrantDualMemoryAdapter(
                    QdrantAdapterConfig(url=self.settings.vector_db_url),
                    embedder=embedder,
                    exporter=exporter,
                    graph_store=graph_store,
                )
            except Exception as exc:
                logger.warning(
                    "Falling back from Qdrant memory adapter because the optional stack is unavailable: %s",
                    exc,
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

    def build_graph_store(self):
        storage_path: Path | None = None
        if self.settings.memory_dir is not None:
            storage_path = self.settings.memory_dir / "knowledge_graph.jsonl"
        elif self.settings.duckdb_path is not None:
            storage_path = self.settings.duckdb_path.parent / "knowledge_graph.jsonl"
        try:
            from calosum.adapters.knowledge_graph_nanorag import NanoGraphRAGKnowledgeGraphStore
            store = NanoGraphRAGKnowledgeGraphStore(storage_path=storage_path)
            self._last_knowledge_graph_backend = store.backend_name
            return store
        except Exception as exc:
            logger.warning(
                "Falling back to in-memory knowledge graph store because the optional stack is unavailable: %s",
                exc,
            )
            self._last_knowledge_graph_backend = "in_memory_graph_fallback"
            return InMemorySemanticGraphStore()

    def build_telemetry_bus(self) -> CognitiveTelemetryBus:
        query_sink = OTLPJsonlTelemetrySink(self.settings.otlp_jsonl) if self.settings.otlp_jsonl else InMemoryTelemetrySink()
        sinks = [query_sink]
        if self.settings.otel_collector_endpoint is not None:
            sinks.append(OTLPHTTPTraceSink(self.settings.otel_collector_endpoint))
        sink = query_sink if len(sinks) == 1 else CompositeTelemetrySink(sinks=sinks, query_sink=query_sink)
        return CognitiveTelemetryBus(sink)

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

    def _build_agent_config(self) -> CalosumAgentConfig:
        return CalosumAgentConfig(awareness_interval_turns=self.settings.awareness_interval_turns)

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

    def build_capability_snapshot(self, action_runtime: Any | None = None) -> CapabilityDescriptor:
        from calosum.shared.types import RoutingPolicy
        right_health = self._right_hemisphere_health()
        right_model = ModelDescriptor(
            provider="huggingface_local" if "huggingface" in self._right_hemisphere_backend_name() else "local",
            model_name=self._right_hemisphere_model_name(),
            backend=self._right_hemisphere_backend_name(),
            health=right_health,
        )
        left_model = ModelDescriptor(
            provider=self.settings.left_hemisphere_provider or "auto",
            model_name=self._reason_model_name(),
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
            health=self._knowledge_graph_health(),
        )
        tools = []
        if action_runtime:
            tools = action_runtime.get_registered_tools()
        routing_policy = RoutingPolicy(
            perception_model=self.settings.perception_model or right_model.model_name,
            reason_model=self.settings.reason_model or left_model.model_name,
            reflection_model=self.settings.reflection_model or left_model.model_name,
            verifier_model=self.settings.verifier_model,
        )
        healths = {right_health, left_model.health, kg_model.health}
        overall_health = ComponentHealth.HEALTHY
        if ComponentHealth.UNAVAILABLE in healths:
            overall_health = ComponentHealth.UNAVAILABLE
        elif ComponentHealth.DEGRADED in healths:
            overall_health = ComponentHealth.DEGRADED

        return CapabilityDescriptor(
            right_hemisphere=right_model,
            left_hemisphere=left_model,
            embeddings=embedding_model,
            knowledge_graph=kg_model,
            tools=tools,
            routing_policy=routing_policy,
            health=overall_health,
        )

    def build_night_trainer(self) -> Any:
        import os
        from calosum.adapters.night_trainer import NightTrainer

        return NightTrainer(
            model_name=self.settings.left_hemisphere_model or "Qwen/Qwen-3.5-9B-Instruct",
            dataset_path=self._runtime_root() / "nightly_data" / "dspy_dataset.jsonl",
            output_dir=self._runtime_root() / "dspy_artifacts" / "latest",
            api_url=self.settings.left_hemisphere_endpoint,
            api_key=self.settings.left_hemisphere_api_key,
            provider=self.settings.left_hemisphere_provider,
            reasoning_effort=self.settings.left_hemisphere_reasoning_effort,
            backend=os.getenv("CALOSUM_NIGHT_TRAINER_BACKEND", "dspy"),
        )

    def describe(self, agent: CalosumAgent | None = None) -> dict[str, Any]:
        action_runtime = getattr(agent, "action_runtime", None) if agent is not None else None
        snapshot = agent.capability_snapshot if agent is not None else None
        if snapshot is None:
            snapshot = self.build_capability_snapshot(action_runtime)
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
            "bridge_state_dir": str(self.settings.bridge_state_dir) if self.settings.bridge_state_dir else None,
            "evolution_archive_path": (
                str(self.settings.evolution_archive_path)
                if self.settings.evolution_archive_path
                else None
            ),
            "awareness_interval_turns": self.settings.awareness_interval_turns,
            "otel_collector_endpoint": self.settings.otel_collector_endpoint,
            "jaeger_ui_url": self.settings.jaeger_ui_url,
            "right_hemisphere_endpoint": self.settings.right_hemisphere_endpoint,
            "left_hemisphere_endpoint": self.settings.left_hemisphere_endpoint,
            "left_hemisphere_model": self._reason_model_name(),
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
            "routing_resolution": self._routing_resolution(snapshot),
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

    def _right_hemisphere_health(self) -> ComponentHealth:
        return ComponentHealth.DEGRADED if self._last_right_hemisphere_backend == "active_inference_heuristic_fallback" else ComponentHealth.HEALTHY

    def _right_hemisphere_backend_name(self) -> str:
        return self._last_right_hemisphere_backend or "active_inference_with_optional_huggingface"

    def _right_hemisphere_model_name(self) -> str:
        return self._last_right_hemisphere_model_name or self.settings.perception_model or "jepa"

    def _left_hemisphere_backend_name(self) -> str:
        return self._last_left_hemisphere_backend or self._left_hemisphere_backend_name_from_settings()

    def _reason_model_name(self) -> str:
        return self.settings.reason_model or self.settings.left_hemisphere_model or "Qwen/Qwen-3.5-9B-Instruct"

    def _embedding_backend_name(self) -> str | None:
        return self._last_embedding_backend or self._derived_embedding_provider() or "auto" if self.settings.vector_db_url else None

    def _knowledge_graph_health(self) -> ComponentHealth:
        return ComponentHealth.DEGRADED if self._knowledge_graph_backend_name() == "in_memory_graph_fallback" else ComponentHealth.HEALTHY

    def _knowledge_graph_backend_name(self) -> str:
        if self._last_knowledge_graph_backend: return self._last_knowledge_graph_backend
        return "nanorag_compatible_networkx" if importlib.util.find_spec("nano_graphrag") else "networkx_graph_rag_fallback"

    def _runtime_root(self) -> Path:
        if self.settings.memory_dir is not None:
            return self.settings.memory_dir.parent
        if self.settings.bridge_state_dir is not None:
            return self.settings.bridge_state_dir.parent
        if self.settings.evolution_archive_path is not None:
            evolution_parent = self.settings.evolution_archive_path.parent
            if evolution_parent.name == "evolution":
                return evolution_parent.parent
            return evolution_parent
        if self.settings.otlp_jsonl is not None:
            telemetry_parent = self.settings.otlp_jsonl.parent
            if telemetry_parent.name == "telemetry":
                return telemetry_parent.parent
            return telemetry_parent
        return Path(".calosum-runtime")

    def _routing_resolution(self, snapshot: CapabilityDescriptor) -> dict[str, dict[str, Any]]:
        right_model = snapshot.right_hemisphere
        left_model = snapshot.left_hemisphere
        reflection_requested = self.settings.reflection_model or left_model.model_name if left_model else None
        reflection_shared = left_model is not None and reflection_requested == left_model.model_name
        verifier_requested = self.settings.verifier_model
        return {
            "perception": {
                "requested_model": self.settings.perception_model or right_model.model_name if right_model else None,
                "active_model": right_model.model_name if right_model else None,
                "backend": right_model.backend if right_model else None,
                "available": right_model.health != ComponentHealth.UNAVAILABLE if right_model else False,
            },
            "reason": {
                "requested_model": self.settings.reason_model or left_model.model_name if left_model else None,
                "active_model": left_model.model_name if left_model else None,
                "backend": left_model.backend if left_model else None,
                "available": left_model.health != ComponentHealth.UNAVAILABLE if left_model else False,
            },
            "reflection": {
                "requested_model": reflection_requested,
                "active_model": left_model.model_name if reflection_shared and left_model else None,
                "backend": left_model.backend if reflection_shared and left_model else None,
                "available": reflection_shared and left_model is not None and left_model.health != ComponentHealth.UNAVAILABLE,
                "shared_with_reasoning": reflection_shared,
                "note": (
                    "shared reasoning backend"
                    if reflection_shared
                    else "dedicated reflection route not implemented"
                ),
            },
            "verifier": {
                "requested_model": verifier_requested,
                "active_model": "heuristic_verifier" if verifier_requested is None else None,
                "backend": "heuristic_verifier",
                "available": verifier_requested is None,
                "shared_with_reasoning": False,
                "note": (
                    "heuristic verifier active"
                    if verifier_requested is None
                    else "dedicated verifier model route not implemented"
                ),
            },
        }

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
        return self.settings.embedding_endpoint or (self.settings.left_hemisphere_endpoint if self._left_endpoint_supports_embeddings() else None)

    def _derived_embedding_model(self) -> str | None:
        return self.settings.embedding_model or ("text-embedding-3-small" if self._left_endpoint_supports_embeddings() else None)

    def _derived_embedding_provider(self) -> str | None:
        return self.settings.embedding_provider or (self._default_embedding_provider() if self._left_endpoint_supports_embeddings() else None)
