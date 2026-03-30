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
from calosum.domain.event_bus import InternalEventBus
from calosum.domain.evolution import JsonlEvolutionArchive
from calosum.domain.memory import DualMemorySystem, InMemorySemanticGraphStore
from calosum.domain.orchestrator import CalosumAgent, CalosumAgentConfig
from calosum.domain.introspection_capabilities import CalosumSystemIntrospector
from calosum.domain.persistent_memory import (
    JsonlEpisodicStore,
    JsonlSemanticStore,
    PersistentDualMemorySystem,
)
from calosum.domain.right_hemisphere import RightHemisphereJEPA
from calosum.bootstrap.settings import InfrastructureProfile, InfrastructureSettings
from calosum.domain.telemetry import CognitiveTelemetryBus, CompositeTelemetrySink, InMemoryTelemetrySink, OTLPJsonlTelemetrySink
from calosum.shared.ports import LatentExchangePort, VisionEmbeddingPort
from calosum.adapters.multimodal_perception import MockVisionAdapter
from calosum.adapters.latent_exchange import InternalLatentExchangeAdapter
from calosum.shared.types import CapabilityDescriptor, ComponentHealth

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CalosumAgentBuilder:
    settings: InfrastructureSettings
    _last_right_hemisphere_backend: str | None = field(default=None, init=False, repr=False)
    _last_left_hemisphere_backend: str | None = field(default=None, init=False, repr=False)
    _last_embedding_backend: str | None = field(default=None, init=False, repr=False)
    _last_knowledge_graph_backend: str | None = field(default=None, init=False, repr=False)
    _last_right_hemisphere_model_name: str | None = field(default=None, init=False, repr=False)

    def build(self, agent_accessor: Any | None = None) -> CalosumAgent:
        left_hemisphere = self.build_left_hemisphere()
        right_hemisphere = self.build_right_hemisphere()
        from calosum.domain.bridge import CognitiveTokenizer
        from calosum.adapters.bridge_store import LocalBridgeStateStore
        bridge_store = None
        if self.settings.bridge_state_dir is not None:
            bridge_store = LocalBridgeStateStore(base_dir=self.settings.bridge_state_dir)
        tokenizer = CognitiveTokenizer(store=bridge_store)
        action_runtime = ConcreteActionRuntime(
            vault=self.settings.vault,
            agent_accessor=agent_accessor
        )
        memory_system = self.build_memory_system()
        telemetry_bus = self.build_telemetry_bus()
        capability_snapshot = self.build_capability_snapshot(action_runtime)
        night_trainer = self.build_night_trainer()
        
        # V3: Latent Exchange & Vision
        latent_exchange = InternalLatentExchangeAdapter(event_bus=self.settings.event_bus or InternalEventBus())
        vision_adapter = MockVisionAdapter()
        
        # Inject vision into right hemisphere if it's the JEPA one
        if isinstance(right_hemisphere, ActiveInferenceRightHemisphereAdapter):
            if hasattr(right_hemisphere.base_adapter, "vision_adapter"):
                right_hemisphere.base_adapter.vision_adapter = vision_adapter
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
            night_trainer=night_trainer,
            latent_exchange=latent_exchange,
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
            base_adapter = RightHemisphereJEPA()
            setattr(base_adapter, "degraded_reason", None)
            return ActiveInferenceRightHemisphereAdapter(base_adapter)

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
            base_adapter = RightHemisphereJEPA()
            setattr(base_adapter, "degraded_reason", f"hf_stack_unavailable:{exc.__class__.__name__}")
            return ActiveInferenceRightHemisphereAdapter(base_adapter)

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
        return CalosumAgentConfig(
            awareness_interval_turns=self.settings.awareness_interval_turns,
            episode_volume_threshold=50 # V3 Default
        )

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
        return CalosumSystemIntrospector.build_capability_snapshot(self, action_runtime)

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
