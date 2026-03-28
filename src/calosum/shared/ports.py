from __future__ import annotations

from typing import Any, Protocol, TYPE_CHECKING, runtime_checkable

from calosum.shared.types import (
    ActionExecutionResult,
    AgentTurnResult,
    CognitiveBridgePacket,
    ConsolidationReport,
    LeftHemisphereResult,
    MemoryContext,
    MemoryEpisode,
    RightHemisphereState,
    UserTurn,
)

if TYPE_CHECKING:
    from calosum.domain.metacognition import CognitiveCandidate, ReflectionOutcome


@runtime_checkable
class RightHemispherePort(Protocol):
    def perceive(self, user_turn: UserTurn) -> RightHemisphereState: ...
    async def aperceive(self, user_turn: UserTurn) -> RightHemisphereState: ...


@runtime_checkable
class CognitiveTokenizerPort(Protocol):
    def translate(self, right_state: RightHemisphereState) -> CognitiveBridgePacket: ...
    async def atranslate(self, right_state: RightHemisphereState) -> CognitiveBridgePacket: ...


@runtime_checkable
class LeftHemispherePort(Protocol):
    def reason(
        self,
        user_turn: UserTurn,
        bridge_packet: CognitiveBridgePacket,
        memory_context: MemoryContext,
        runtime_feedback: list[str] | None = None,
        attempt: int = 0,
    ) -> LeftHemisphereResult: ...

    async def areason(
        self,
        user_turn: UserTurn,
        bridge_packet: CognitiveBridgePacket,
        memory_context: MemoryContext,
        runtime_feedback: list[str] | None = None,
        attempt: int = 0,
    ) -> LeftHemisphereResult: ...

    def repair(
        self,
        user_turn: UserTurn,
        bridge_packet: CognitiveBridgePacket,
        memory_context: MemoryContext,
        previous_result: LeftHemisphereResult,
        rejected_results: list[ActionExecutionResult],
        attempt: int,
    ) -> LeftHemisphereResult: ...

    async def arepair(
        self,
        user_turn: UserTurn,
        bridge_packet: CognitiveBridgePacket,
        memory_context: MemoryContext,
        previous_result: LeftHemisphereResult,
        rejected_results: list[ActionExecutionResult],
        attempt: int,
    ) -> LeftHemisphereResult: ...


@runtime_checkable
class MemorySystemPort(Protocol):
    def build_context(self, user_turn: UserTurn, episodic_limit: int = 5) -> MemoryContext: ...
    async def abuild_context(
        self, user_turn: UserTurn, episodic_limit: int = 5
    ) -> MemoryContext: ...
    def store_episode(self, episode: MemoryEpisode) -> None: ...
    async def astore_episode(self, episode: MemoryEpisode) -> None: ...
    def sleep_mode(self) -> ConsolidationReport: ...
    async def asleep_mode(self) -> ConsolidationReport: ...


@runtime_checkable
class ActionRuntimePort(Protocol):
    def run(self, left_result: LeftHemisphereResult) -> list[ActionExecutionResult]: ...
    async def arun(self, left_result: LeftHemisphereResult) -> list[ActionExecutionResult]: ...


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

    def apply_neuroplasticity(self, tokenizer: Any, outcome: "ReflectionOutcome") -> None: ...
