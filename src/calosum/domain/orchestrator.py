from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from time import perf_counter
from uuid import uuid4

from calosum.shared.async_utils import maybe_await, run_sync
from calosum.domain.directive_guardrails import (
    apply_controlled_right_hemisphere_params,
    apply_runtime_contract_audit_directive,
)
from calosum.domain.agent_execution import AgentExecutionEngine
from calosum.domain.agent_config import CalosumAgentConfig, BranchingBudget
from calosum.domain.bridge import CognitiveTokenizer
from calosum.domain.event_bus import CognitiveEvent, InternalEventBus
from calosum.domain.left_hemisphere import LeftHemisphereLogicalSLM
from calosum.domain.memory import DualMemorySystem
from calosum.domain.evolution import (
    EvolutionManager,
    JsonlEvolutionArchive,
    EvolutionProposer,
)
from calosum.domain.metacognition import (
    CognitiveCandidate,
    CognitiveVariantSpec,
    GEAReflectionController,
    GroupTurnResult,
    default_cognitive_personas,
)
from calosum.shared.ports import (
    ActionRuntimePort,
    CognitiveTokenizerPort,
    LeftHemispherePort,
    MemorySystemPort,
    ReflectionControllerPort,
    RightHemispherePort,
    TelemetryBusPort,
    VerifierPort,
    LatentExchangePort,
    VisionEmbeddingPort,
)
from calosum.domain.right_hemisphere import RightHemisphereJEPA
from calosum.domain.runtime import StrictLambdaRuntime
from calosum.domain.telemetry import CognitiveTelemetryBus
from calosum.domain.verifier import HeuristicVerifier
from calosum.domain.idle_foraging import build_idle_foraging_turn
from calosum.shared.types import (
    AgentTurnResult,
    CapabilityDescriptor,
    CognitiveArchitectureMap,
    CognitiveWorkspace,
    DirectiveType,
    EvolutionDirective,
    MemoryEpisode,
    SessionDiagnostic,
    UserTurn,
    utc_now,
)

class CalosumAgent:
    """
    Orquestrador do ciclo cognitivo (Dual-Hemisphere).
    Suporta I/O concorrente e inferencia remota.
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
        verifier: VerifierPort | None = None,
        config: CalosumAgentConfig | None = None,
        capability_snapshot: CapabilityDescriptor | None = None,
        evolution_archive: "JsonlEvolutionArchive | None" = None,
        night_trainer: Any | None = None,
        latent_exchange: LatentExchangePort | None = None,
    ) -> None:
        self.right_hemisphere = right_hemisphere or RightHemisphereJEPA()
        self.tokenizer = tokenizer or CognitiveTokenizer()
        self.left_hemisphere = left_hemisphere or LeftHemisphereLogicalSLM()
        self.memory_system = memory_system or DualMemorySystem()
        self.action_runtime = action_runtime or StrictLambdaRuntime()
        self.telemetry_bus = telemetry_bus or CognitiveTelemetryBus()
        self.reflection_controller = reflection_controller or GEAReflectionController()
        self.verifier = verifier or HeuristicVerifier()
        self.config = config or CalosumAgentConfig()
        self.capability_snapshot = capability_snapshot
        self.evolution_archive = evolution_archive
        self.night_trainer = night_trainer
        self.latent_exchange = latent_exchange
        self.event_bus = InternalEventBus()
        self.execution_engine = AgentExecutionEngine(
            action_runtime=self.action_runtime,
            max_runtime_retries=self.config.max_runtime_retries,
            verifier=self.verifier,
        )
        self.evolution_manager = EvolutionManager(
            self.evolution_archive,
            EvolutionProposer()
        )
        # Último workspace por sessão
        self.last_workspace_by_session: dict[str, CognitiveWorkspace] = {}
        self.latest_awareness_by_session: dict[str, SessionDiagnostic] = {}
        self._awareness_turn_counts: dict[str, int] = {}

        from calosum.domain.self_model import build_self_model
        self.self_model: CognitiveArchitectureMap = build_self_model(self)

    def process_turn(self, user_turn: UserTurn) -> AgentTurnResult | GroupTurnResult:
        return run_sync(self.aprocess_turn(user_turn))

    async def aprocess_turn(self, user_turn: UserTurn) -> AgentTurnResult | GroupTurnResult:
        started_at = perf_counter()
        
        from calosum.domain.workspace import init_turn_workspace
        workspace = init_turn_workspace(self, user_turn)
        self.last_workspace_by_session[user_turn.session_id] = workspace
        
        await self.event_bus.publish(CognitiveEvent("UserTurnEvent", user_turn, user_turn.turn_id))

        memory_context = await maybe_await(
            self.execution_engine.call_component(
                self.memory_system, "abuild_context", "build_context", user_turn
            )
        )
        
        await self.event_bus.publish(CognitiveEvent("MemoryContextEvent", memory_context, user_turn.turn_id))

        right_state = await maybe_await(
            self.execution_engine.call_component(
                self.right_hemisphere, "aperceive", "perceive", user_turn, memory_context, workspace
            )
        )
        
        await self.event_bus.publish(CognitiveEvent("PerceptionEvent", right_state, user_turn.turn_id))

        # Active Inference V2/V3: Branching based on VFE/EFE
        surprise_score = getattr(right_state, "surprise_score", 0.0)

        # EFE with proper epistemic/pragmatic decomposition
        expected_free_energy = 0.5
        if hasattr(self.right_hemisphere, "expected_free_energy"):
            efe_value, _efe_components = self.right_hemisphere.expected_free_energy(
                right_state.latent_vector, memory_context
            )
            expected_free_energy = efe_value
        else:
            ambiguity_score = right_state.world_hypotheses.get("interaction_complexity", 0.0)
            semantic_density = right_state.world_hypotheses.get("semantic_density", 0.5)
            expected_free_energy = (ambiguity_score * 0.6) + (semantic_density * 0.4)

        needs_branching = (
            self.config.branching_budget.max_depth > 0
            and (surprise_score > self.config.surprise_threshold or expected_free_energy > 0.75)
        )

        if needs_branching:
            max_width = self.config.branching_budget.max_width
            variants = default_cognitive_personas(max_width)
            
            return await self.aprocess_group_turn(user_turn, variants, _precomputed_memory=memory_context, _precomputed_right=right_state, _precomputed_workspace=workspace)

        capabilities_dict = None
        if self.capability_snapshot:
            from dataclasses import asdict
            capabilities_dict = asdict(self.capability_snapshot)

        result = await self.execution_engine.run_variant(
            user_turn=user_turn,
            memory_context=memory_context,
            right_state=right_state,
            tokenizer=self.tokenizer,
            left_hemisphere=self.left_hemisphere,
            bridge_directives=self.evolution_manager.approved_prompt_directives,
            capabilities=capabilities_dict,
            workspace=workspace,
        )
        result.latency_ms = round((perf_counter() - started_at) * 1000.0, 3)
        
        await self.event_bus.publish(CognitiveEvent("ExecutionEvent", result, user_turn.turn_id))
        
        await self._store_selected_episode(result)
        await maybe_await(
            self.execution_engine.call_component(
                self.telemetry_bus, "arecord_turn", "record_turn", result
            )
        )
        
        # Awareness loop for single turn (can be probabilistic or periodic)
        await self._awareness_loop(user_turn.session_id)

        # V3 Collective Intelligence: Share latent state with peers
        if self.latent_exchange:
            await self.latent_exchange.broadcast_latent(
                user_turn.session_id, 
                result.right_state.latent_vector
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
        _precomputed_memory: Any | None = None,
        _precomputed_right: Any | None = None,
        _precomputed_workspace: CognitiveWorkspace | None = None,
    ) -> GroupTurnResult:
        if not variants:
            raise ValueError("process_group_turn requires at least one variant")

        started_at = perf_counter()
        
        workspace = _precomputed_workspace
        if workspace is None:
            from calosum.domain.workspace import init_turn_workspace
            workspace = init_turn_workspace(self, user_turn)
            self.last_workspace_by_session[user_turn.session_id] = workspace
        
        memory_context = _precomputed_memory
        if memory_context is None:
            memory_context = await maybe_await(
                self.execution_engine.call_component(
                    self.memory_system, "abuild_context", "build_context", user_turn
                )
            )
            
        right_state = _precomputed_right
        if right_state is None:
            right_state = await maybe_await(
                self.execution_engine.call_component(
                    self.right_hemisphere, "aperceive", "perceive", user_turn, memory_context, workspace
                )
            )
        candidates: list[CognitiveCandidate] = []

        capabilities_dict = None
        if self.capability_snapshot:
            from dataclasses import asdict
            capabilities_dict = asdict(self.capability_snapshot)

        async def _run_variant(variant: CognitiveVariantSpec) -> CognitiveCandidate:
            variant_started_at = perf_counter()
            tokenizer = self.execution_engine.clone_component_with_overrides(self.tokenizer, variant.tokenizer_overrides)
            left_hemisphere = self.execution_engine.clone_component_with_overrides(self.left_hemisphere, variant.left_overrides)
            turn_result = await self.execution_engine.run_variant(
                user_turn=user_turn,
                memory_context=memory_context,
                right_state=right_state,
                tokenizer=tokenizer,
                left_hemisphere=left_hemisphere,
                variant_label=variant.variant_id,
                bridge_directives=self.evolution_manager.approved_prompt_directives + list(variant.bridge_directives),
                capabilities=capabilities_dict,
                workspace=workspace,
            )
            turn_result.latency_ms = round((perf_counter() - variant_started_at) * 1000.0, 3)
            return CognitiveCandidate(variant=variant, turn_result=turn_result)

        candidates = list(await asyncio.gather(*(_run_variant(variant) for variant in variants)))

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
        
        await self._awareness_loop(user_turn.session_id)

        return GroupTurnResult(
            user_turn=user_turn,
            right_state=right_state,
            candidates=candidates,
            selected_result=selected_candidate.turn_result,
            reflection=reflection,
        )

    async def _awareness_loop(self, session_id: str) -> None:
        self._awareness_turn_counts[session_id] = self._awareness_turn_counts.get(session_id, 0) + 1
        if self._awareness_turn_counts[session_id] % max(1, self.config.awareness_interval_turns) != 0:
            return

        diagnostic = self.analyze_session(session_id, persist=True)
        if not diagnostic.bottlenecks:
            return

        for directive in self.evolution_manager.proposer.propose(diagnostic):
            if directive.directive_type == DirectiveType.PARAMETER:
                self.evolution_manager.apply_directive(directive, self)
                if self.evolution_manager.archive:
                    self.evolution_manager.archive.record_directive(directive, event="auto_applied")
                continue
            self.evolution_manager.queue_directive(directive)

    def analyze_session(self, session_id: str, *, persist: bool = False) -> SessionDiagnostic:
        dashboard = self.cognitive_dashboard(session_id)
        if not dashboard.get("decision"):
            diagnostic = SessionDiagnostic(
                session_id=session_id, analyzed_turns=0, tool_success_rate=1.0,
                average_retries=0.0, average_surprise=0.0, bottlenecks=[],
                pending_approval_backlog=0,
                pending_directive_count=len(self.evolution_manager.pending_directives))
        else:
            from calosum.domain.introspection import IntrospectionEngine
            diagnostic = IntrospectionEngine().analyze(
                session_id, dashboard,
                pending_directive_count=len(self.evolution_manager.pending_directives))
        self.latest_awareness_by_session[session_id] = diagnostic
        if persist:
            if hasattr(self.telemetry_bus, "record_awareness"):
                self.telemetry_bus.record_awareness(session_id, diagnostic)
            if self.evolution_archive:
                self.evolution_archive.record_diagnostic(diagnostic)
        return diagnostic

    def latest_awareness_for_session(self, sid: str | None = None) -> SessionDiagnostic | None:
        if sid: return self.latest_awareness_by_session.get(sid)
        return list(self.latest_awareness_by_session.values())[-1] if self.latest_awareness_by_session else None

    def workspace_for_session(self, sid: str | None = None) -> CognitiveWorkspace | None:
        if sid: return self.last_workspace_by_session.get(sid)
        return list(self.last_workspace_by_session.values())[-1] if self.last_workspace_by_session else None

    @property
    def pending_directives(self) -> list[EvolutionDirective]:
        return self.evolution_manager.pending_directives

    def apply_pending_directive(self, directive_id: str) -> EvolutionDirective | None:
        return self.evolution_manager.apply_pending_directive(directive_id, self)

    def _apply_directive(self, directive: EvolutionDirective) -> None:
        """Compatibility method for tests."""
        self.evolution_manager.apply_directive(directive, self)

    def sleep_mode(self): return run_sync(self.asleep_mode())
    async def asleep_mode(self):
        report = await maybe_await(self.execution_engine.call_component(self.memory_system, "asleep_mode", "sleep_mode"))
        await self.event_bus.publish(CognitiveEvent("SleepModeCompletedEvent", report, "system-loop"))
        if self.night_trainer is not None:
            await maybe_await(self.execution_engine.call_component(self.night_trainer, "arun_training_cycle", "run_training_cycle"))
        return report
    def cognitive_dashboard(self, sid: str | None = None) -> dict[str, list[dict]]: return self.telemetry_bus.dashboard_for_session(sid)

    async def _store_selected_episode(self, result: AgentTurnResult) -> None:
        ep = MemoryEpisode(
            episode_id=str(uuid4()), recorded_at=utc_now(), user_turn=result.user_turn,
            right_state=result.right_state, bridge_packet=result.bridge_packet,
            left_result=result.left_result, execution_results=result.execution_results,
            runtime_retry_count=result.runtime_retry_count, critique_revision_count=result.critique_revision_count)
        await maybe_await(self.execution_engine.call_component(self.memory_system, "astore_episode", "store_episode", ep))
        if self.memory_system and self.config.episode_volume_threshold > 0:
            count = await maybe_await(self.execution_engine.call_component(self.memory_system, "aepisode_count", "episode_count"))
            if count % self.config.episode_volume_threshold == 0: await self.asleep_mode()

    def idle_foraging(self) -> AgentTurnResult | GroupTurnResult | None: return run_sync(self.aidle_foraging())
    async def aidle_foraging(self) -> AgentTurnResult | GroupTurnResult | None:
        return await self.aprocess_turn(build_idle_foraging_turn(self.memory_system))
