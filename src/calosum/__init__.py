"""Public package surface for the Calosum cognitive architecture."""

from calosum.adapters.active_inference import ActiveInferenceRightHemisphereAdapter
from calosum.adapters.bridge_cross_attention import CrossAttentionBridgeAdapter, CrossAttentionBridgeConfig
from calosum.adapters.left_hemisphere_rlm import RlmAdapterConfig, RlmLeftHemisphereAdapter
from calosum.adapters.multimodal_perception import LocalClipVisionAdapter, LocalClipVisionConfig
from calosum.adapters.right_hemisphere_jepars import JepaRsConfig, JepaRsRightHemisphereAdapter
from calosum.adapters.right_hemisphere_vjepa21 import VJepa21Config, VJepa21RightHemisphereAdapter
from calosum.adapters.right_hemisphere_vljepa import VLJepaConfig, VLJepaRightHemisphereAdapter
from calosum.bootstrap.factory import CalosumAgentBuilder
from calosum.domain.bridge import CognitiveTokenizer, CognitiveTokenizerConfig
from calosum.domain.left_hemisphere import LeftHemisphereLogicalSLM, LeftHemisphereLogicalSLMConfig
from calosum.domain.memory import (
    DualMemorySystem,
    InMemoryEpisodicStore,
    InMemorySemanticGraphStore,
    InMemorySemanticStore,
    SleepModeConsolidator,
)
from calosum.domain.metacognition import (
    CognitiveCandidate,
    CognitiveVariantSpec,
    GEAReflectionController,
    GroupTurnResult,
    ReflectionOutcome,
    ReflectionScore,
)
from calosum.domain.multiagent import ExecutorRole, PlannerRole, VerifierRole
from calosum.domain.orchestrator import CalosumAgent, CalosumAgentConfig
from calosum.domain.persistent_memory import (
    JsonlEpisodicStore,
    JsonlSemanticGraphStore,
    JsonlSemanticStore,
    PersistentDualMemorySystem,
)
from calosum.shared.ports import (
    ActionRuntimePort,
    BridgeFusionPort,
    CognitiveTokenizerPort,
    ExperienceStorePort,
    LeftHemispherePort,
    MemorySystemPort,
    ReflectionControllerPort,
    RightHemispherePort,
    TelemetryBusPort,
)
from calosum.domain.right_hemisphere import RightHemisphereJEPA, RightHemisphereJEPAConfig
from calosum.domain.runtime import StrictLambdaRuntime, StrictLambdaRuntimeConfig
from calosum.shared.serialization import dump_json, to_json, to_primitive
from calosum.bootstrap.settings import InfrastructureProfile, InfrastructureSettings
from calosum.domain.telemetry import (
    CognitiveTelemetryBus,
    InMemoryTelemetrySink,
    OTLPJsonlTelemetrySink,
    TelemetryEvent,
)
from calosum.shared.types import (
    ActionExecutionResult,
    AgentTurnResult,
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
    from calosum.adapters.knowledge_graph_nanorag import NanoGraphRAGKnowledgeGraphStore
except Exception:  # pragma: no cover - optional dependency surface
    NanoGraphRAGKnowledgeGraphStore = None  # type: ignore[assignment]

__all__ = [
    "ActionExecutionResult",
    "ActionRuntimePort",
    "ActiveInferenceRightHemisphereAdapter",
    "AgentTurnResult",
    "BridgeFusionPort",
    "CalosumAgent",
    "CalosumAgentBuilder",
    "CalosumAgentConfig",
    "CognitiveCandidate",
    "CognitiveBridgePacket",
    "CognitiveTelemetryBus",
    "CognitiveTokenizer",
    "CognitiveTokenizerPort",
    "CognitiveTokenizerConfig",
    "CognitiveVariantSpec",
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
