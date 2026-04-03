"""Public package surface for the Calosum cognitive architecture."""

from calosum.adapters.perception.simple_distance import SimpleDistanceSurpriseAdapter
from calosum.adapters.bridge.bridge_cross_attention import CrossAttentionBridgeAdapter, CrossAttentionBridgeConfig
from calosum.adapters.hemisphere.left_hemisphere_rlm_ast import RlmAstAdapterConfig, RlmAstLeftHemisphereAdapter
from calosum.adapters.perception.multimodal_perception import LocalClipVisionAdapter, LocalClipVisionConfig
from calosum.adapters.hemisphere.input_perception_jepars import JepaRsConfig, JepaRsRightHemisphereAdapter
from calosum.adapters.hemisphere.input_perception_heuristic_jepa import (
    HeuristicJEPAAdapter,
    HeuristicJEPAConfig,
)
from calosum.adapters.hemisphere.input_perception_trained_jepa import (
    TrainedJEPAAdapter,
    TrainedJEPAConfig,
)
from calosum.adapters.hemisphere.input_perception_vjepa21 import VJepa21Config, VJepa21RightHemisphereAdapter
from calosum.adapters.hemisphere.input_perception_vljepa import VLJepaConfig, VLJepaRightHemisphereAdapter
from calosum.bootstrap.wiring.factory import CalosumAgentBuilder
from calosum.bootstrap.wiring.agent_baseline import AgentBaseline, AgentBaselineConfig
from calosum.domain.cognition.bridge import ContextCompressor, ContextCompressorConfig, CognitiveTokenizer, CognitiveTokenizerConfig
from calosum.domain.cognition.action_planner import ActionPlannerLogicalSLM, ActionPlannerLogicalSLMConfig
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
from calosum.adapters.memory.persistent_sql_memory import PersistentDualMemorySystem
from calosum.shared.models.ports import (
    ToolRuntimePort,
    BridgeFusionPort,
    ContextCompressorPort,
    CognitiveTokenizerPort,
    ExperienceStorePort,
    JEPAInputPerceptionPort,
    ActionPlannerPort,
    MemorySystemPort,
    ReflectionControllerPort,
    InputPerceptionPort,
    TelemetryBusPort,
)
from calosum.shared.models.jepa import ContextEmbedding, ResponsePrediction, SurpriseScore
from calosum.domain.cognition.input_perception import InputPerceptionJEPA, InputPerceptionJEPAConfig
from calosum.domain.execution.tool_runtime import ToolRuntime, ToolRuntimeConfig
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
    PerceptionSummary,
    ConsolidationReport,
    FailureType,
    KnowledgeTriple,
    ActionPlannerResult,
    MemoryContext,
    MemoryEpisode,
    Modality,
    MultimodalSignal,
    PrimitiveAction,
    InputPerceptionState,
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
    "ToolRuntimePort",
    "SimpleDistanceSurpriseAdapter",
    "AgentTurnResult",
    "AgentBaseline",
    "AgentBaselineConfig",
    "BridgeFusionPort",
    "CalosumAgent",
    "CalosumAgentBuilder",
    "CalosumAgentConfig",
    "CognitiveCandidate",
    "PerceptionSummary",
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
    "LinearReflectionController",
    "GroupTurnResult",
    "HeuristicJEPAAdapter",
    "HeuristicJEPAConfig",
    "TrainedJEPAAdapter",
    "TrainedJEPAConfig",
    "InMemoryEpisodicStore",
    "InMemorySemanticGraphStore",
    "InMemorySemanticStore",
    "InMemoryTelemetrySink",
    "InfrastructureProfile",
    "InfrastructureSettings",
    "JepaRsConfig",
    "JepaRsRightHemisphereAdapter",
    "JEPAInputPerceptionPort",
    "KnowledgeTriple",
    "NanoGraphRAGKnowledgeGraphStore",
    "ActionPlannerPort",
    "LocalClipVisionAdapter",
    "LocalClipVisionConfig",
    "ActionPlannerLogicalSLM",
    "ActionPlannerLogicalSLMConfig",
    "ActionPlannerResult",
    "MemorySystemPort",
    "MemoryContext",
    "MemoryEpisode",
    "Modality",
    "MultimodalSignal",
    "OTLPJsonlTelemetrySink",
    "PersistentDualMemorySystem",
    "PrimitiveAction",
    "ResponsePrediction",
    "ReflectionControllerPort",
    "InputPerceptionJEPA",
    "InputPerceptionPort",
    "InputPerceptionJEPAConfig",
    "InputPerceptionState",
    "RlmAstAdapterConfig",
    "RlmAstLeftHemisphereAdapter",
    "ReflectionOutcome",
    "ReflectionScore",
    "SemanticRule",
    "SleepModeConsolidator",
    "SoftPromptToken",
    "ToolRuntime",
    "ToolRuntimeConfig",
    "TelemetryBusPort",
    "TelemetryEvent",
    "SurpriseScore",
    "to_json",
    "to_primitive",
    "TypedLambdaProgram",
    "UserTurn",
    "ContextEmbedding",
    "VJepa21Config",
    "VJepa21RightHemisphereAdapter",
    "VLJepaConfig",
    "VLJepaRightHemisphereAdapter",
]
