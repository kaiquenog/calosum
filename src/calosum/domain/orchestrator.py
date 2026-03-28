from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from uuid import uuid4

from calosum.shared.async_utils import maybe_await, run_sync
from calosum.domain.agent_execution import AgentExecutionEngine
from calosum.domain.bridge import CognitiveTokenizer
from calosum.domain.left_hemisphere import LeftHemisphereLogicalSLM
from calosum.domain.memory import DualMemorySystem
from calosum.domain.metacognition import (
    CognitiveCandidate,
    CognitiveVariantSpec,
    GEAReflectionController,
    GroupTurnResult,
)
from calosum.shared.ports import (
    ActionRuntimePort,
    CognitiveTokenizerPort,
    LeftHemispherePort,
    MemorySystemPort,
    ReflectionControllerPort,
    RightHemispherePort,
    TelemetryBusPort,
)
from calosum.domain.right_hemisphere import RightHemisphereJEPA
from calosum.domain.runtime import StrictLambdaRuntime
from calosum.domain.telemetry import CognitiveTelemetryBus
from calosum.shared.types import (
    AgentTurnResult,
    MemoryEpisode,
    UserTurn,
    utc_now,
)


@dataclass(slots=True)
class CalosumAgentConfig:
    max_runtime_retries: int = 2


class CalosumAgent:
    """
    Orquestrador do ciclo cognitivo.

    O nucleo e assíncrono para suportar I/O concorrente e inferencia remota.
    Os metodos sincronos existentes continuam disponiveis como wrappers para
    facilitar CLI, scripts locais e testes unitarios.
    """

    def __init__(
        self,
        right_hemisphere: RightHemispherePort | None = None,
        tokenizer: CognitiveTokenizerPort | None = None,
        left_hemisphere: LeftHemispherePort | None = None,
        memory_system: MemorySystemPort | None = None,
        action_runtime: ActionRuntimePort | None = None,
        telemetry_bus: TelemetryBusPort | None = None,
        reflection_controller: ReflectionControllerPort | None = None,
        config: CalosumAgentConfig | None = None,
    ) -> None:
        self.right_hemisphere = right_hemisphere or RightHemisphereJEPA()
        self.tokenizer = tokenizer or CognitiveTokenizer()
        self.left_hemisphere = left_hemisphere or LeftHemisphereLogicalSLM()
        self.memory_system = memory_system or DualMemorySystem()
        self.action_runtime = action_runtime or StrictLambdaRuntime()
        self.telemetry_bus = telemetry_bus or CognitiveTelemetryBus()
        self.reflection_controller = reflection_controller or GEAReflectionController()
        self.config = config or CalosumAgentConfig()
        self.execution_engine = AgentExecutionEngine(
            action_runtime=self.action_runtime,
            max_runtime_retries=self.config.max_runtime_retries,
        )

    def process_turn(self, user_turn: UserTurn) -> AgentTurnResult:
        return run_sync(self.aprocess_turn(user_turn))

    async def aprocess_turn(self, user_turn: UserTurn) -> AgentTurnResult:
        started_at = perf_counter()
        memory_context = await maybe_await(
            self.execution_engine.call_component(
                self.memory_system, "abuild_context", "build_context", user_turn
            )
        )
        right_state = await maybe_await(
            self.execution_engine.call_component(
                self.right_hemisphere, "aperceive", "perceive", user_turn
            )
        )
        result = await self.execution_engine.run_variant(
            user_turn=user_turn,
            memory_context=memory_context,
            right_state=right_state,
            tokenizer=self.tokenizer,
            left_hemisphere=self.left_hemisphere,
        )
        result.latency_ms = round((perf_counter() - started_at) * 1000.0, 3)
        await self._store_selected_episode(result)
        await maybe_await(
            self.execution_engine.call_component(
                self.telemetry_bus, "arecord_turn", "record_turn", result
            )
        )
        return result

    def process_group_turn(
        self,
        user_turn: UserTurn,
        variants: list[CognitiveVariantSpec],
    ) -> GroupTurnResult:
        return run_sync(self.aprocess_group_turn(user_turn, variants))

    async def aprocess_group_turn(
        self,
        user_turn: UserTurn,
        variants: list[CognitiveVariantSpec],
    ) -> GroupTurnResult:
        if not variants:
            raise ValueError("process_group_turn requires at least one variant")

        started_at = perf_counter()
        memory_context = await maybe_await(
            self.execution_engine.call_component(
                self.memory_system, "abuild_context", "build_context", user_turn
            )
        )
        right_state = await maybe_await(
            self.execution_engine.call_component(
                self.right_hemisphere, "aperceive", "perceive", user_turn
            )
        )
        candidates: list[CognitiveCandidate] = []

        for variant in variants:
            tokenizer = self._build_variant_tokenizer(variant)
            left_hemisphere = self._build_variant_left_hemisphere(variant)
            turn_result = await self.execution_engine.run_variant(
                user_turn=user_turn,
                memory_context=memory_context,
                right_state=right_state,
                tokenizer=tokenizer,
                left_hemisphere=left_hemisphere,
            )
            candidates.append(CognitiveCandidate(variant=variant, turn_result=turn_result))

        reflection = await maybe_await(
            self.execution_engine.call_component(
                self.reflection_controller,
                "aevaluate",
                "evaluate",
                candidates,
                self.tokenizer,
            )
        )
        self.reflection_controller.apply_neuroplasticity(self.tokenizer, reflection)
        selected_candidate = next(
            item
            for item in candidates
            if item.variant.variant_id == reflection.selected_variant_id
        )
        selected_candidate.turn_result.latency_ms = round((perf_counter() - started_at) * 1000.0, 3)
        await self._store_selected_episode(selected_candidate.turn_result)
        await maybe_await(
            self.execution_engine.call_component(
                self.telemetry_bus,
                "arecord_turn",
                "record_turn",
                selected_candidate.turn_result,
            )
        )
        await maybe_await(
            self.execution_engine.call_component(
                self.telemetry_bus,
                "arecord_reflection",
                "record_reflection",
                user_turn.session_id,
                user_turn.turn_id,
                reflection.as_dict(),
            )
        )

        return GroupTurnResult(
            user_turn=user_turn,
            right_state=right_state,
            candidates=candidates,
            selected_result=selected_candidate.turn_result,
            reflection=reflection,
        )

    def sleep_mode(self):
        return run_sync(self.asleep_mode())

    async def asleep_mode(self):
        return await maybe_await(
            self.execution_engine.call_component(
                self.memory_system, "asleep_mode", "sleep_mode"
            )
        )

    def cognitive_dashboard(self, session_id: str) -> dict[str, list[dict]]:
        return self.telemetry_bus.dashboard_for_session(session_id)

    async def _store_selected_episode(self, result: AgentTurnResult) -> None:
        episode = MemoryEpisode(
            episode_id=str(uuid4()),
            recorded_at=utc_now(),
            user_turn=result.user_turn,
            right_state=result.right_state,
            bridge_packet=result.bridge_packet,
            left_result=result.left_result,
        )
        await maybe_await(
            self.execution_engine.call_component(
                self.memory_system, "astore_episode", "store_episode", episode
            )
        )

    def _build_variant_tokenizer(self, variant: CognitiveVariantSpec) -> CognitiveTokenizerPort:
        return self.execution_engine.clone_component_with_overrides(
            self.tokenizer, variant.tokenizer_overrides
        )

    def _build_variant_left_hemisphere(
        self,
        variant: CognitiveVariantSpec,
    ) -> LeftHemispherePort:
        return self.execution_engine.clone_component_with_overrides(
            self.left_hemisphere, variant.left_overrides
        )
