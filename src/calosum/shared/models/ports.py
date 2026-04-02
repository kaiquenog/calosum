from __future__ import annotations

from typing import Any, Awaitable, Callable, Protocol, TYPE_CHECKING, runtime_checkable

from calosum.shared.models.jepa import ContextEmbedding, ResponsePrediction, SurpriseScore
from calosum.shared.models.types import (
    ActionExecutionResult,
    AgentTurnResult,
    CognitiveBridgePacket,
    CognitiveWorkspace,
    ConsolidationReport,
    CritiqueVerdict,
    LeftHemisphereResult,
    MemoryContext,
    MemoryEpisode,
    RightHemisphereState,
    UserTurn,
)

if TYPE_CHECKING:
    from calosum.domain.metacognition.metacognition import CognitiveCandidate, ReflectionOutcome

@runtime_checkable
class ChannelPort(Protocol):
    async def listen(self, on_message: Callable[[UserTurn], Awaitable[None]]) -> None: ...
    async def send(self, session_id: str, text: str) -> None: ...

@runtime_checkable
class RightHemispherePort(Protocol):
    def perceive(self, user_turn: UserTurn, memory_context: MemoryContext | None = None, workspace: CognitiveWorkspace | None = None) -> RightHemisphereState: ...
    async def aperceive(self, user_turn: UserTurn, memory_context: MemoryContext | None = None, workspace: CognitiveWorkspace | None = None) -> RightHemisphereState: ...


@runtime_checkable
class JEPARightHemispherePort(Protocol):
    async def encode_context(self, turns: list[UserTurn]) -> ContextEmbedding: ...
    async def predict_response_embedding(self, ctx: ContextEmbedding) -> ResponsePrediction: ...
    async def compute_surprise(self, ctx: ContextEmbedding, actual_response: str) -> SurpriseScore: ...


@runtime_checkable
class ContextCompressorPort(Protocol):
    def translate(self, right_state: RightHemisphereState, workspace: CognitiveWorkspace | None = None) -> CognitiveBridgePacket: ...
    async def atranslate(self, right_state: RightHemisphereState, workspace: CognitiveWorkspace | None = None) -> CognitiveBridgePacket: ...


@runtime_checkable
class LeftHemispherePort(Protocol):
    def reason(
        self,
        user_turn: UserTurn,
        bridge_packet: CognitiveBridgePacket,
        memory_context: MemoryContext,
        runtime_feedback: list[str] | None = None,
        attempt: int = 0,
        workspace: CognitiveWorkspace | None = None,
    ) -> LeftHemisphereResult: ...

    async def areason(
        self,
        user_turn: UserTurn,
        bridge_packet: CognitiveBridgePacket,
        memory_context: MemoryContext,
        runtime_feedback: list[str] | None = None,
        attempt: int = 0,
        workspace: CognitiveWorkspace | None = None,
    ) -> LeftHemisphereResult: ...

    def repair(
        self,
        user_turn: UserTurn,
        bridge_packet: CognitiveBridgePacket,
        memory_context: MemoryContext,
        previous_result: LeftHemisphereResult,
        rejected_results: list[ActionExecutionResult],
        attempt: int,
        critique_feedback: list[str] | None = None,
        workspace: CognitiveWorkspace | None = None,
    ) -> LeftHemisphereResult: ...

    async def arepair(
        self,
        user_turn: UserTurn,
        bridge_packet: CognitiveBridgePacket,
        memory_context: MemoryContext,
        previous_result: LeftHemisphereResult,
        rejected_results: list[ActionExecutionResult],
        attempt: int,
        critique_feedback: list[str] | None = None,
        workspace: CognitiveWorkspace | None = None,
    ) -> LeftHemisphereResult: ...


@runtime_checkable
class MemorySystemPort(Protocol):
    def build_context(self, user_turn: UserTurn, episodic_limit: int = 5) -> MemoryContext: ...
    async def abuild_context(
        self, user_turn: UserTurn, episodic_limit: int = 5
    ) -> MemoryContext: ...
    def episode_count(self) -> int: ...
    async def aepisode_count(self) -> int: ...
    def store_episode(self, episode: MemoryEpisode) -> None: ...
    async def astore_episode(self, episode: MemoryEpisode) -> None: ...
    def sleep_mode(self) -> ConsolidationReport: ...
    async def asleep_mode(self) -> ConsolidationReport: ...


@runtime_checkable
class ActionRuntimePort(Protocol):
    def run(self, left_result: LeftHemisphereResult, workspace: CognitiveWorkspace | None = None) -> list[ActionExecutionResult]: ...
    async def arun(self, left_result: LeftHemisphereResult, workspace: CognitiveWorkspace | None = None) -> list[ActionExecutionResult]: ...
    def get_registered_tools(self) -> list["ToolDescriptor"]: ...

@runtime_checkable
class TelemetryBusPort(Protocol):
    def record_turn(self, result: AgentTurnResult) -> None: ...
    async def arecord_turn(self, result: AgentTurnResult) -> None: ...
    def record_reflection(self, session_id: str, turn_id: str, payload: dict[str, Any]) -> None: ...
    async def arecord_reflection(
        self, session_id: str, turn_id: str, payload: dict[str, Any]
    ) -> None: ...
    def dashboard_for_session(self, session_id: str) -> dict[str, list[dict[str, Any]]]: ...


@runtime_checkable
class ReflectionControllerPort(Protocol):
    def evaluate(
        self,
        candidates: list["CognitiveCandidate"],
        base_tokenizer: Any,
    ) -> "ReflectionOutcome": ...

    async def aevaluate(
        self,
        candidates: list["CognitiveCandidate"],
        base_tokenizer: Any,
    ) -> "ReflectionOutcome": ...

    def apply_config_adaptation(self, tokenizer: Any, outcome: "ReflectionOutcome") -> None: ...
    def apply_neuroplasticity(self, tokenizer: Any, outcome: "ReflectionOutcome") -> None: ...


CognitiveTokenizerPort = ContextCompressorPort


@runtime_checkable
class DatasetExporterPort(Protocol):
    def export(self, dataset: list[dict[str, Any]], filename: str) -> str: ...

@runtime_checkable
class BridgeStateStorePort(Protocol):
    def load_weights(self, projection_layer: Any) -> bool: ...
    def load_adaptation_state(self) -> dict[str, Any]: ...
    def persist_adaptation_state(self, state: dict[str, Any]) -> None: ...
    def record_reflection_event(self, payload: dict[str, Any]) -> None: ...

@runtime_checkable
class VerifierPort(Protocol):
    def verify(
        self,
        user_turn: UserTurn,
        left_result: LeftHemisphereResult,
        execution_results: list[ActionExecutionResult],
        workspace: CognitiveWorkspace | None = None,
    ) -> "CritiqueVerdict": ...

    async def averify(
        self,
        user_turn: UserTurn,
        left_result: LeftHemisphereResult,
        execution_results: list[ActionExecutionResult],
        workspace: CognitiveWorkspace | None = None,
    ) -> "CritiqueVerdict": ...


@runtime_checkable
class VisionEmbeddingPort(Protocol):
    def embed_image(self, image_data: bytes) -> list[float]: ...
    async def aembed_image(self, image_data: bytes) -> list[float]: ...


@runtime_checkable
class LatentExchangePort(Protocol):
    async def broadcast_latent(self, session_id: str, latent_vector: list[float]) -> None: ...
    async def get_peer_latents(self, session_id: str) -> list[list[float]]: ...


@runtime_checkable
class BridgeFusionPort(Protocol):
    def fuse_latent(
        self,
        *,
        latent_vector: list[float],
        emotional_labels: list[str],
        surprise: float = 0.0,
        confidence: float = 0.0,
        context_novelty: float = 0.0,
    ) -> tuple[list[float], dict[str, Any]]: ...


@runtime_checkable
class ExperienceStorePort(Protocol):
    def record_experience(
        self,
        *,
        context_type: str,
        variant_id: str,
        score: float,
        reward: float,
        metadata: dict[str, Any] | None = None,
    ) -> None: ...

    def variant_prior(
        self,
        *,
        context_type: str,
        variant_id: str,
        limit: int = 100,
    ) -> float: ...


@runtime_checkable
class DistributedExperiencePort(Protocol):
    """Inter-agent experience sharing port (GEA Sprint 3)."""
    def broadcast_experience(
        self, *, agent_id: str, context_type: str, variant_id: str,
        score: float, metadata: dict[str, Any] | None = None,
    ) -> None: ...

    def collect_peer_experiences(
        self, *, context_type: str, limit: int = 50,
    ) -> list[dict[str, Any]]: ...


@runtime_checkable
class VectorCodecPort(Protocol):
    """Codec de compressão/descompressão de vetores de alta dimensão (TurboQuant)."""

    def encode(self, vector: list[float]) -> bytes: ...
    def decode(self, compressed: bytes) -> list[float]: ...
    def inner_product_approx(self, query: list[float], compressed: bytes) -> float: ...

    @property
    def bits_per_dim(self) -> int: ...


@runtime_checkable
class McpClientPort(Protocol):
    def list_servers(self) -> list[str]: ...
    def call_tool(
        self,
        *,
        server: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]: ...
