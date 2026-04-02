"""Public package surface for the Calosum cognitive architecture."""

from calosum.adapters.perception.active_inference import ActiveInferenceRightHemisphereAdapter
from calosum.adapters.bridge.bridge_cross_attention import CrossAttentionBridgeAdapter, CrossAttentionBridgeConfig
from calosum.adapters.hemisphere.left_hemisphere_rlm import RlmAdapterConfig, RlmLeftHemisphereAdapter
from calosum.adapters.perception.multimodal_perception import LocalClipVisionAdapter, LocalClipVisionConfig
from calosum.adapters.hemisphere.right_hemisphere_jepars import JepaRsConfig, JepaRsRightHemisphereAdapter
from calosum.adapters.hemisphere.right_hemisphere_vjepa21 import VJepa21Config, VJepa21RightHemisphereAdapter
from calosum.adapters.hemisphere.right_hemisphere_vljepa import VLJepaConfig, VLJepaRightHemisphereAdapter
from calosum.bootstrap.wiring.factory import CalosumAgentBuilder
from calosum.bootstrap.wiring.agent_baseline import AgentBaseline, AgentBaselineConfig
from calosum.domain.cognition.bridge import ContextCompressor, ContextCompressorConfig, CognitiveTokenizer, CognitiveTokenizerConfig
from calosum.domain.cognition.left_hemisphere import LeftHemisphereLogicalSLM, LeftHemisphereLogicalSLMConfig
from calosum.domain.memory.memory import (
    DualMemorySystem,
    InMemoryEpisodicStore,
    InMemorySemanticGraphStore,
    InMemorySemanticStore,
    SleepModeConsolidator,
)
from calosum.domain.metacognition.metacognition import (
    CognitiveCandidate,
    CognitiveVariantSpec,
    CognitiveVariantSelector,
    GEAReflectionController,
    GroupTurnResult,
    ReflectionOutcome,
    ReflectionScore,
)
from calosum.domain.agent.multiagent import ExecutorRole, PlannerRole, VerifierRole
from calosum.domain.agent.orchestrator import CalosumAgent, CalosumAgentConfig
from calosum.domain.memory.persistent_memory import (
    JsonlEpisodicStore,
    JsonlSemanticGraphStore,
    JsonlSemanticStore,
    PersistentDualMemorySystem,
)
from calosum.shared.models.ports import (
    ActionRuntimePort,
    BridgeFusionPort,
    ContextCompressorPort,
    CognitiveTokenizerPort,
    ExperienceStorePort,
    LeftHemispherePort,
    MemorySystemPort,
    ReflectionControllerPort,
    RightHemispherePort,
    TelemetryBusPort,
)
from calosum.domain.cognition.right_hemisphere import RightHemisphereJEPA, RightHemisphereJEPAConfig
from calosum.domain.execution.runtime import StrictLambdaRuntime, StrictLambdaRuntimeConfig
from calosum.shared.utils.serialization import dump_json, to_json, to_primitive
from calosum.bootstrap.infrastructure.settings import InfrastructureProfile, InfrastructureSettings
from calosum.domain.infrastructure.telemetry import (
    CognitiveTelemetryBus,
    InMemoryTelemetrySink,
    OTLPJsonlTelemetrySink,
    TelemetryEvent,
)
from calosum.shared.models.types import (
    ActionExecutionResult,
    AgentTurnResult,
    ContextDirective,
    CognitiveBridgePacket,
    ConsolidationReport,
    FailureType,
    KnowledgeTriple,
    LeftHemisphereResult,
    MemoryContext,
    MemoryEpisode,
    Modality,
    MultimodalSignal,
    PrimitiveAction,
    RightHemisphereState,
    SemanticRule,
    SoftPromptToken,
    TypedLambdaProgram,
    UserTurn,
)

try:
    from calosum.adapters.knowledge.knowledge_graph_nanorag import NanoGraphRAGKnowledgeGraphStore
except Exception:  # pragma: no cover - optional dependency surface
    NanoGraphRAGKnowledgeGraphStore = None  # type: ignore[assignment]

__all__ = [
    "ActionExecutionResult",
    "ActionRuntimePort",
    "ActiveInferenceRightHemisphereAdapter",
    "AgentTurnResult",
    "AgentBaseline",
    "AgentBaselineConfig",
    "BridgeFusionPort",
    "CalosumAgent",
    "CalosumAgentBuilder",
    "CalosumAgentConfig",
    "CognitiveCandidate",
    "CognitiveBridgePacket",
    "ContextDirective",
    "CognitiveTelemetryBus",
    "ContextCompressor",
    "CognitiveTokenizer",
    "ContextCompressorPort",
    "CognitiveTokenizerPort",
    "ContextCompressorConfig",
    "CognitiveTokenizerConfig",
    "CognitiveVariantSpec",
    "CognitiveVariantSelector",
    "ConsolidationReport",
    "CrossAttentionBridgeAdapter",
    "CrossAttentionBridgeConfig",
    "DualMemorySystem",
    "dump_json",
    "ExperienceStorePort",
    "FailureType",
    "GEAReflectionController",
    "GroupTurnResult",
    "InMemoryEpisodicStore",
    "InMemorySemanticGraphStore",
    "InMemorySemanticStore",
    "InMemoryTelemetrySink",
    "InfrastructureProfile",
    "InfrastructureSettings",
    "JepaRsConfig",
    "JepaRsRightHemisphereAdapter",
    "JsonlEpisodicStore",
    "JsonlSemanticGraphStore",
    "JsonlSemanticStore",
    "KnowledgeTriple",
    "NanoGraphRAGKnowledgeGraphStore",
    "LeftHemispherePort",
    "LocalClipVisionAdapter",
    "LocalClipVisionConfig",
    "LeftHemisphereLogicalSLM",
    "LeftHemisphereLogicalSLMConfig",
    "LeftHemisphereResult",
    "MemorySystemPort",
    "MemoryContext",
    "MemoryEpisode",
    "Modality",
    "MultimodalSignal",
    "OTLPJsonlTelemetrySink",
    "PersistentDualMemorySystem",
    "PrimitiveAction",
    "ReflectionControllerPort",
    "RightHemisphereJEPA",
    "RightHemispherePort",
    "RightHemisphereJEPAConfig",
    "RightHemisphereState",
    "RlmAdapterConfig",
    "RlmLeftHemisphereAdapter",
    "ReflectionOutcome",
    "ReflectionScore",
    "SemanticRule",
    "SleepModeConsolidator",
    "SoftPromptToken",
    "StrictLambdaRuntime",
    "StrictLambdaRuntimeConfig",
    "TelemetryBusPort",
    "TelemetryEvent",
    "to_json",
    "to_primitive",
    "TypedLambdaProgram",
    "UserTurn",
    "VJepa21Config",
    "VJepa21RightHemisphereAdapter",
    "VLJepaConfig",
    "VLJepaRightHemisphereAdapter",
]
