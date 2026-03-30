from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import uuid4


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Modality(StrEnum):
    TEXT = "text"
    AUDIO = "audio"
    VIDEO = "video"
    TYPING = "typing"
    SENSOR = "sensor"


class FailureType(StrEnum):
    SCHEMA_VIOLATION = "schema"
    UNSAFE_CONTENT = "safety"
    RUNTIME_REJECTION = "runtime"
    INCOMPLETE_RESULT = "incomplete"


class ComponentHealth(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


@dataclass(slots=True)
class ToolDescriptor:
    name: str
    description: str
    requires_approval: bool
    required_permissions: list[str]
    health: ComponentHealth = ComponentHealth.HEALTHY


@dataclass(slots=True)
class ModelDescriptor:
    provider: str
    model_name: str
    backend: str
    health: ComponentHealth = ComponentHealth.HEALTHY


@dataclass(slots=True)
class RoutingPolicy:
    perception_model: str
    reason_model: str
    reflection_model: str
    verifier_model: str | None = None


@dataclass(slots=True)
class CapabilityDescriptor:
    right_hemisphere: ModelDescriptor | None
    left_hemisphere: ModelDescriptor | None
    embeddings: ModelDescriptor | None
    knowledge_graph: ModelDescriptor | None
    tools: list[ToolDescriptor]
    routing_policy: RoutingPolicy | None = None
    health: ComponentHealth = ComponentHealth.HEALTHY


@dataclass(slots=True)
class ArchitectureComponent:
    component_id: str
    role: str
    adapter_class: str
    health: ComponentHealth


@dataclass(slots=True)
class ComponentConnection:
    source: str
    target: str
    protocol: str


@dataclass(slots=True)
class AdaptationSurface:
    tunable_parameters: list[str]
    supported_directives: list[str]


@dataclass(slots=True)
class CognitiveArchitectureMap:
    components: list[ArchitectureComponent]
    connections: list[ComponentConnection]
    adaptation_surface: AdaptationSurface
    capabilities: CapabilityDescriptor


@dataclass(slots=True)
class MultimodalSignal:
    modality: Modality
    source: str
    payload: Any
    quality: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


class DirectiveType(StrEnum):
    PARAMETER = "parameter"
    PROMPT = "prompt"
    TOPOLOGY = "topology"
    ARCHITECTURE = "architecture"


@dataclass(slots=True)
class CognitiveBottleneck:
    bottleneck_id: str
    description: str
    severity: float
    evidence: list[str]
    affected_components: list[str]


@dataclass(slots=True)
class SessionDiagnostic:
    session_id: str
    analyzed_turns: int
    tool_success_rate: float
    average_retries: float
    average_surprise: float
    bottlenecks: list[CognitiveBottleneck]
    failure_types: dict[str, int] = field(default_factory=dict)
    pending_approval_backlog: int = 0
    pending_directive_count: int = 0
    surprise_trend: float = 0.0
    dominant_variant: str | None = None
    dominant_variant_ratio: float = 0.0
    generated_at: datetime = field(default_factory=utc_now)


@dataclass(slots=True)
class EvolutionDirective:
    directive_id: str
    directive_type: DirectiveType
    target_component: str
    proposed_change: dict[str, Any]
    reasoning: str
    status: str = "pending"  # pending, applied, rejected
    created_at: datetime = field(default_factory=utc_now)


@dataclass(slots=True)
class CognitiveWorkspace:
    """
    Memoria operacional compartilhada de um turno.
    Cada componente preenche sua respectiva secao ao longo do pipeline.
    """
    task_frame: dict[str, Any] = field(default_factory=dict)
    self_model_ref: dict[str, Any] | None = None
    capability_snapshot: CapabilityDescriptor | None = None
    right_notes: dict[str, Any] = field(default_factory=dict)
    bridge_state: dict[str, Any] = field(default_factory=dict)
    left_notes: dict[str, Any] = field(default_factory=dict)
    verifier_feedback: list[dict[str, Any]] = field(default_factory=list)
    runtime_feedback: list[dict[str, Any]] = field(default_factory=list)
    pending_questions: list[str] = field(default_factory=list)


@dataclass(slots=True)
class UserTurn:
    """
    Unidade de entrada do agente.

    `signals` carrega o texto do usuario e pistas nao textuais que o JEPA pode
    usar para compor um estado contextual abstrato.
    """

    session_id: str
    user_text: str
    signals: list[MultimodalSignal] = field(default_factory=list)
    observed_at: datetime = field(default_factory=utc_now)
    turn_id: str = field(default_factory=lambda: str(uuid4()))


@dataclass(slots=True)
class RightHemisphereState:
    """
    Saida primaria do hemisferio direito.

    O vetor latente representa o "sentimento/contexto" continuo do ambiente.
    Nenhum texto e gerado aqui.
    """

    context_id: str
    latent_vector: list[float]
    salience: float
    emotional_labels: list[str]
    world_hypotheses: dict[str, float]
    confidence: float
    surprise_score: float = 0.0
    telemetry: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not (0.0 <= self.salience <= 1.0):
            raise ValueError(f"salience must be between 0.0 and 1.0, got {self.salience}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {self.confidence}")
        if not (0.0 <= self.surprise_score <= 1.0):
            raise ValueError(f"surprise_score must be between 0.0 and 1.0, got {self.surprise_score}")


@dataclass(slots=True)
class SoftPromptToken:
    token: str
    weight: float
    provenance: str


@dataclass(slots=True)
class BridgeControlSignal:
    """
    Sinais de controle emitidos pelo corpo caloso.

    O SLM recebe a sugestao de temperatura e diretivas de sistema ajustadas
    pela saliencia afetiva do contexto.
    """

    target_temperature: float
    empathy_priority: bool
    system_directives: list[str] = field(default_factory=list)
    annotations: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.target_temperature < 0.0:
            raise ValueError(f"target_temperature must be non-negative, got {self.target_temperature}")


@dataclass(slots=True)
class CognitiveBridgePacket:
    """
    Interface de comunicacao entre intuicao continua e raciocinio discreto.
    """

    context_id: str
    soft_prompts: list[SoftPromptToken]
    control: BridgeControlSignal
    salience: float
    latent_vector: list[float] = field(default_factory=list)
    bridge_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PrimitiveAction:
    """
    Acao primitiva tipificada, pronta para ser executada por um runtime seguro.
    """

    action_type: str
    typed_signature: str
    payload: dict[str, Any]
    safety_invariants: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ActionExecutionResult:
    action_type: str
    typed_signature: str
    status: str
    output: dict[str, Any]
    violations: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CritiqueVerdict:
    is_valid: bool
    critique_reasoning: list[str]
    identified_issues: list[str]
    suggested_fixes: list[str]
    confidence: float
    failure_types: list[FailureType] = field(default_factory=list)

    def __post_init__(self):
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {self.confidence}")


@dataclass(slots=True)
class TypedLambdaProgram:
    """
    Representacao simbolica de uma solucao no estilo lambda-recursive.
    """

    signature: str
    expression: str
    expected_effect: str


@dataclass(slots=True)
class SemanticRule:
    rule_id: str
    statement: str
    strength: float
    supporting_episodes: list[str]
    tags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class KnowledgeTriple:
    subject: str
    predicate: str
    object: str
    weight: float = 1.0
    source_rule_id: str | None = None


@dataclass(slots=True)
class MemoryContext:
    recent_episodes: list["MemoryEpisode"] = field(default_factory=list)
    semantic_rules: list[SemanticRule] = field(default_factory=list)
    knowledge_triples: list[KnowledgeTriple] = field(default_factory=list)


@dataclass(slots=True)
class LeftHemisphereResult:
    """
    Resultado do hemisferio esquerdo.

    `lambda_program` representa o plano simbolico.
    `actions` representa a fronteira operacional segura do agente.
    """

    response_text: str
    lambda_program: TypedLambdaProgram
    actions: list[PrimitiveAction]
    reasoning_summary: list[str]
    telemetry: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MemoryEpisode:
    episode_id: str
    recorded_at: datetime
    user_turn: UserTurn
    right_state: RightHemisphereState
    bridge_packet: CognitiveBridgePacket
    left_result: LeftHemisphereResult
    execution_results: list[ActionExecutionResult] = field(default_factory=list)
    runtime_retry_count: int = 0
    critique_revision_count: int = 0


@dataclass(slots=True)
class ConsolidationReport:
    started_at: datetime
    finished_at: datetime
    episodes_considered: int
    promoted_rules: list[SemanticRule]
    lora_adaptation_backlog: list[str]
    graph_updates: list[KnowledgeTriple] = field(default_factory=list)


@dataclass(slots=True)
class CognitiveTelemetrySnapshot:
    """
    Estrutura para dashboards do tipo:
    - o que o JEPA sentiu
    - o que o SLM pensou
    - que decisao final saiu
    """

    felt: dict[str, Any]
    thought: dict[str, Any]
    decision: dict[str, Any]
    capabilities: dict[str, Any] = field(default_factory=dict)
    bridge_config: dict[str, Any] = field(default_factory=dict)
    active_variant: str | None = None


@dataclass(slots=True)
class AgentTurnResult:
    user_turn: UserTurn
    memory_context: MemoryContext
    right_state: RightHemisphereState
    bridge_packet: CognitiveBridgePacket
    left_result: LeftHemisphereResult
    telemetry: CognitiveTelemetrySnapshot
    execution_results: list[ActionExecutionResult] = field(default_factory=list)
    runtime_retry_count: int = 0
    critique_revision_count: int = 0
    latency_ms: float = 0.0
