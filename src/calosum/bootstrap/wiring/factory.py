from __future__ import annotations

import logging
from dataclasses import dataclass, field
import importlib.util
from pathlib import Path
from typing import Any

from calosum.adapters.action_runtime import ConcreteActionRuntime
from calosum.adapters.tools.mcp_client import HttpMcpClientAdapter, McpServerEndpoint
from calosum.adapters.communication.telemetry_otlp import OTLPHTTPTraceSink
from calosum.adapters.memory.text_embeddings import TextEmbeddingAdapter, TextEmbeddingAdapterConfig
from calosum.domain.infrastructure.event_bus import InternalEventBus
from calosum.domain.agent.evolution import JsonlEvolutionArchive
from calosum.domain.infrastructure.interceptors import AuditLogInterceptor, InterceptorManager
from calosum.domain.memory.memory import DualMemorySystem, InMemorySemanticGraphStore
from calosum.domain.agent.orchestrator import CalosumAgent, CalosumAgentConfig
from calosum.domain.metacognition.introspection_capabilities import CalosumSystemIntrospector
from calosum.domain.memory.persistent_memory import (
    JsonlEpisodicStore,
    JsonlSemanticStore,
    PersistentDualMemorySystem,
)
from calosum.bootstrap.infrastructure.settings import InfrastructureProfile, InfrastructureSettings
from calosum.bootstrap.wiring.backend_resolvers import (
    resolve_bridge_fusion,
    resolve_left_hemisphere,
    resolve_reflection_controller,
    resolve_right_hemisphere,
    resolve_vision_adapter,
)
from calosum.domain.infrastructure.telemetry import CognitiveTelemetryBus, CompositeTelemetrySink, InMemoryTelemetrySink, OTLPJsonlTelemetrySink
from calosum.adapters.communication.latent_exchange import InternalLatentExchangeAdapter
from calosum.shared.models.types import CapabilityDescriptor, ComponentHealth

logger = logging.getLogger(__name__)


def _build_codec(settings: InfrastructureSettings):
    """Instantiate a VectorCodecPort if vector_quantization flag is set."""
    if settings.vector_quantization == "turboquant":
        from calosum.adapters.perception.quantized_embeddings import TurboQuantVectorCodec
        return TurboQuantVectorCodec(bits=settings.turboquant_bits)
    return None


@dataclass(slots=True)
class CalosumAgentBuilder:
    settings: InfrastructureSettings
    _last_right_hemisphere_backend: str | None = field(default=None, init=False, repr=False)
    _last_left_hemisphere_backend: str | None = field(default=None, init=False, repr=False)
    _last_embedding_backend: str | None = field(default=None, init=False, repr=False)
    _last_knowledge_graph_backend: str | None = field(default=None, init=False, repr=False)
    _last_right_hemisphere_model_name: str | None = field(default=None, init=False, repr=False)

    def build(self, agent_accessor: Any | None = None) -> CalosumAgent:
        interceptor_manager = InterceptorManager([AuditLogInterceptor()])
        left_hemisphere = self.build_left_hemisphere()
        right_hemisphere = self.build_right_hemisphere()
        from calosum.domain.cognition.bridge import CognitiveTokenizer
        from calosum.adapters.bridge.bridge_store import LocalBridgeStateStore
        bridge_store = None
        if self.settings.bridge_state_dir is not None:
            bridge_store = LocalBridgeStateStore(base_dir=self.settings.bridge_state_dir)
        tokenizer = CognitiveTokenizer(
            store=bridge_store,
            fusion=resolve_bridge_fusion(self.settings),
        )
        action_runtime = ConcreteActionRuntime(
            vault=self.settings.vault,
            agent_accessor=agent_accessor,
            mcp_client=self.build_mcp_client(),
            interceptor_manager=interceptor_manager,
        )
        memory_system = self.build_memory_system()
        telemetry_bus = self.build_telemetry_bus()
        capability_snapshot = self.build_capability_snapshot(action_runtime)
        night_trainer = self.build_night_trainer()

        latent_exchange = InternalLatentExchangeAdapter(event_bus=self.settings.event_bus or InternalEventBus())
        reflection_controller = resolve_reflection_controller(self.settings)
        agent = CalosumAgent(
            right_hemisphere=right_hemisphere,
            tokenizer=tokenizer,
            left_hemisphere=left_hemisphere,
            action_runtime=action_runtime,
            memory_system=memory_system,
            telemetry_bus=telemetry_bus,
            config=self._build_agent_config(),
            capability_snapshot=capability_snapshot,
            evolution_archive=JsonlEvolutionArchive(self.settings.evolution_archive_path),
            night_trainer=night_trainer,
            latent_exchange=latent_exchange,
            reflection_controller=reflection_controller,
        )
        interceptor_manager.attach_event_bus(agent.event_bus)
        setattr(agent, "interceptor_manager", interceptor_manager)
        return agent

    def build_left_hemisphere(self):
        try:
            adapter, backend = resolve_left_hemisphere(self.settings, self._reason_model_name())
            self._last_left_hemisphere_backend = backend
            return adapter
        except Exception as exc:
            logger.warning("Falling back to default left hemisphere adapter: %s", exc)
            from calosum.adapters.llm.llm_qwen import QwenLeftHemisphereAdapter

            self._last_left_hemisphere_backend = "openai_compatible_chat_adapter_default"
            return QwenLeftHemisphereAdapter()

    def build_right_hemisphere(self):
        vision_adapter = resolve_vision_adapter()
        try:
            codec = _build_codec(self.settings)
            adapter, backend, model_name = resolve_right_hemisphere(
                self.settings,
                vision_adapter=vision_adapter,
                codec=codec,
            )
            self._last_right_hemisphere_backend = backend
            self._last_right_hemisphere_model_name = model_name
            return adapter
        except Exception as exc:
            logger.warning("Falling back to heuristic right hemisphere adapter: %s", exc)
            from calosum.adapters.perception.active_inference import ActiveInferenceRightHemisphereAdapter
            from calosum.domain.cognition.right_hemisphere import RightHemisphereJEPA

            base_adapter = RightHemisphereJEPA(vision_adapter=vision_adapter)
            setattr(base_adapter, "degraded_reason", f"resolver_fallback:{exc.__class__.__name__}")
            self._last_right_hemisphere_backend = "active_inference_heuristic_fallback"
            self._last_right_hemisphere_model_name = "jepa"
            return ActiveInferenceRightHemisphereAdapter(base_adapter)

    def build_memory_system(self):
        from calosum.adapters.night_trainer.night_trainer import LocalDatasetExporter
        from calosum.domain.memory.memory import SleepModeConsolidator
        exporter = LocalDatasetExporter(self._runtime_root() / "nightly_data")
        graph_store = self.build_graph_store()
        if self.settings.vector_db_url:
            try:
                from calosum.adapters.memory.memory_qdrant import QdrantAdapterConfig, QdrantDualMemoryAdapter
                embedder = self.build_text_embedder()
                codec = _build_codec(self.settings)
                return QdrantDualMemoryAdapter(
                    QdrantAdapterConfig(
                        url=self.settings.vector_db_url,
                        scalar_quantization=self.settings.qdrant_scalar_quantization,
                    ),
                    embedder=embedder,
                    exporter=exporter,
                    graph_store=graph_store,
                    codec=codec,
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
            from calosum.adapters.knowledge.knowledge_graph_nanorag import NanoGraphRAGKnowledgeGraphStore
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
        return CalosumAgentConfig(
            awareness_interval_turns=self.settings.awareness_interval_turns,
            episode_volume_threshold=50 # V3 Default
        )

    def build_capability_snapshot(self, action_runtime: Any | None = None) -> CapabilityDescriptor:
        return CalosumSystemIntrospector.build_capability_snapshot(self, action_runtime)

    def build_night_trainer(self) -> Any:
        import os
        from calosum.adapters.night_trainer.night_trainer import NightTrainer

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

    def build_mcp_client(self) -> HttpMcpClientAdapter | None:
        if not self.settings.mcp_enabled:
            return None
        servers: dict[str, McpServerEndpoint] = {}
        for name, url in self.settings.mcp_servers.items():
            servers[name] = McpServerEndpoint(name=name, url=url)
        allowlisted_servers = set(self.settings.mcp_allowlist or []) or None
        return HttpMcpClientAdapter(
            servers=servers,
            allowlisted_servers=allowlisted_servers,
        )

    def describe(self, agent: CalosumAgent | None = None) -> dict[str, Any]:
        return CalosumSystemIntrospector.describe(self, agent)

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
        return (
            ComponentHealth.DEGRADED
            if self._last_right_hemisphere_backend in {"active_inference_heuristic_fallback"}
            else ComponentHealth.HEALTHY
        )

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
        r_m, l_m = snapshot.right_hemisphere, snapshot.left_hemisphere
        refl_req = self.settings.reflection_model or l_m.model_name if l_m else None
        refl_sh = l_m is not None and refl_req == l_m.model_name
        return {
            "perception": {"active": r_m.model_name if r_m else None, "backend": r_m.backend if r_m else None},
            "reason": {"active": l_m.model_name if l_m else None, "backend": l_m.backend if l_m else None},
            "reflection": {"active": l_m.model_name if refl_sh and l_m else None, "shared": refl_sh},
            "verifier": {"active": "heuristic_verifier", "shared": False},
        }

    def _left_hemisphere_backend_name_from_settings(self) -> str:
        if (self.settings.left_hemisphere_backend or "").lower() == "rlm":
            return "rlm_recursive_adapter"
        if self.settings.left_hemisphere_fallback_endpoint:
            return "resilient_failover_adapter"
        endpoint = (self.settings.left_hemisphere_endpoint or "").lower()
        provider = (self.settings.left_hemisphere_provider or "auto").lower()

        if provider in {"openai_responses", "openai", "responses"}:
            return "openai_responses_adapter"
        if provider in {"openai_chat", "chat"}:
            return "openai_chat_adapter"
        if provider == "openrouter":
            return "openrouter_adapter"
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
