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


@dataclass(slots=True)
class MultimodalSignal:
    modality: Modality
    source: str
    payload: Any
    quality: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


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


@dataclass(slots=True)
class CognitiveBridgePacket:
    """
    Interface de comunicacao entre intuicao continua e raciocinio discreto.
    """

    context_id: str
    soft_prompts: list[SoftPromptToken]
    control: BridgeControlSignal
    salience: float
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
    latency_ms: float = 0.0
