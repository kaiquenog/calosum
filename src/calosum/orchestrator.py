from __future__ import annotations

from dataclasses import replace
from uuid import uuid4

from .bridge import CognitiveTokenizer
from .left_hemisphere import LeftHemisphereLogicalSLM
from .memory import DualMemorySystem
from .metacognition import (
    CognitiveCandidate,
    CognitiveVariantSpec,
    GEAReflectionController,
    GroupTurnResult,
)
from .right_hemisphere import RightHemisphereJEPA
from .runtime import StrictLambdaRuntime
from .telemetry import CognitiveTelemetryBus
from .types import (
    AgentTurnResult,
    CognitiveTelemetrySnapshot,
    MemoryEpisode,
    UserTurn,
    utc_now,
)


class CalosumAgent:
    """
    Orquestrador do ciclo cognitivo.

    Fluxo:
    1. JEPA percebe o turno e gera um estado latente contextual.
    2. CognitiveTokenizer comprime esse estado em soft prompts e sinais de controle.
    3. O SLM consulta memoria dual e produz plano simbolico + acoes primitivas.
    4. O episodio completo e persistido para recuperacao futura e consolidacao.
    """

    def __init__(
        self,
        right_hemisphere: RightHemisphereJEPA | None = None,
        tokenizer: CognitiveTokenizer | None = None,
        left_hemisphere: LeftHemisphereLogicalSLM | None = None,
        memory_system: DualMemorySystem | None = None,
        action_runtime: StrictLambdaRuntime | None = None,
        telemetry_bus: CognitiveTelemetryBus | None = None,
        reflection_controller: GEAReflectionController | None = None,
    ) -> None:
        self.right_hemisphere = right_hemisphere or RightHemisphereJEPA()
        self.tokenizer = tokenizer or CognitiveTokenizer()
        self.left_hemisphere = left_hemisphere or LeftHemisphereLogicalSLM()
        self.memory_system = memory_system or DualMemorySystem()
        self.action_runtime = action_runtime or StrictLambdaRuntime()
        self.telemetry_bus = telemetry_bus or CognitiveTelemetryBus()
        self.reflection_controller = reflection_controller or GEAReflectionController()

    def process_turn(self, user_turn: UserTurn) -> AgentTurnResult:
        memory_context = self.memory_system.build_context(user_turn)
        right_state = self.right_hemisphere.perceive(user_turn)
        result = self._run_variant(
            user_turn=user_turn,
            memory_context=memory_context,
            right_state=right_state,
            tokenizer=self.tokenizer,
            left_hemisphere=self.left_hemisphere,
        )
        self._store_selected_episode(result)
        self.telemetry_bus.record_turn(result)
        return result

    def process_group_turn(
        self,
        user_turn: UserTurn,
        variants: list[CognitiveVariantSpec],
    ) -> GroupTurnResult:
        if not variants:
            raise ValueError("process_group_turn requires at least one variant")

        memory_context = self.memory_system.build_context(user_turn)
        right_state = self.right_hemisphere.perceive(user_turn)
        candidates: list[CognitiveCandidate] = []

        for variant in variants:
            tokenizer = self._build_variant_tokenizer(variant)
            left_hemisphere = self._build_variant_left_hemisphere(variant)
            turn_result = self._run_variant(
                user_turn=user_turn,
                memory_context=memory_context,
                right_state=right_state,
                tokenizer=tokenizer,
                left_hemisphere=left_hemisphere,
            )
            candidates.append(CognitiveCandidate(variant=variant, turn_result=turn_result))

        reflection = self.reflection_controller.evaluate(
            candidates=candidates,
            base_tokenizer=self.tokenizer,
        )
        self.reflection_controller.apply_neuroplasticity(self.tokenizer, reflection)
        selected_candidate = next(
            item
            for item in candidates
            if item.variant.variant_id == reflection.selected_variant_id
        )
        self._store_selected_episode(selected_candidate.turn_result)
        self.telemetry_bus.record_turn(selected_candidate.turn_result)
        self.telemetry_bus.record_reflection(
            session_id=user_turn.session_id,
            turn_id=user_turn.turn_id,
            payload=reflection.as_dict(),
        )

        return GroupTurnResult(
            user_turn=user_turn,
            right_state=right_state,
            candidates=candidates,
            selected_result=selected_candidate.turn_result,
            reflection=reflection,
        )

    def sleep_mode(self):
        return self.memory_system.sleep_mode()

    def cognitive_dashboard(self, session_id: str) -> dict[str, list[dict]]:
        return self.telemetry_bus.dashboard_for_session(session_id)

    def _run_variant(
        self,
        user_turn: UserTurn,
        memory_context,
        right_state,
        tokenizer: CognitiveTokenizer,
        left_hemisphere: LeftHemisphereLogicalSLM,
    ) -> AgentTurnResult:
        bridge_packet = tokenizer.translate(right_state)
        left_result = left_hemisphere.reason(user_turn, bridge_packet, memory_context)
        execution_results = self.action_runtime.run(left_result)
        telemetry = self._build_telemetry(right_state, left_result)

        return AgentTurnResult(
            user_turn=user_turn,
            memory_context=memory_context,
            right_state=right_state,
            bridge_packet=bridge_packet,
            left_result=left_result,
            telemetry=telemetry,
            execution_results=execution_results,
        )

    def _build_telemetry(self, right_state, left_result) -> CognitiveTelemetrySnapshot:
        return CognitiveTelemetrySnapshot(
            felt={
                "context_id": right_state.context_id,
                "emotional_labels": right_state.emotional_labels,
                "salience": right_state.salience,
                "world_hypotheses": right_state.world_hypotheses,
            },
            thought={
                "lambda_signature": left_result.lambda_program.signature,
                "reasoning_summary": left_result.reasoning_summary,
                "system_directives": left_result.telemetry.get("system_directives", []),
            },
            decision={
                "response_text": left_result.response_text,
                "action_types": [action.action_type for action in left_result.actions],
            },
        )

    def _store_selected_episode(self, result: AgentTurnResult) -> None:
        episode = MemoryEpisode(
            episode_id=str(uuid4()),
            recorded_at=utc_now(),
            user_turn=result.user_turn,
            right_state=result.right_state,
            bridge_packet=result.bridge_packet,
            left_result=result.left_result,
        )
        self.memory_system.store_episode(episode)

    def _build_variant_tokenizer(self, variant: CognitiveVariantSpec) -> CognitiveTokenizer:
        config = replace(self.tokenizer.config)
        for key, value in variant.tokenizer_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return CognitiveTokenizer(config=config)

    def _build_variant_left_hemisphere(
        self,
        variant: CognitiveVariantSpec,
    ) -> LeftHemisphereLogicalSLM:
        config = replace(self.left_hemisphere.config)
        for key, value in variant.left_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return LeftHemisphereLogicalSLM(config=config)
