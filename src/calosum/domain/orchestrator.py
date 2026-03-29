from __future__ import annotations

import json
from dataclasses import dataclass, field
from time import perf_counter
from uuid import uuid4

from calosum.shared.async_utils import maybe_await, run_sync
from calosum.domain.agent_execution import AgentExecutionEngine
from calosum.domain.bridge import CognitiveTokenizer
from calosum.domain.event_bus import CognitiveEvent, InternalEventBus
from calosum.domain.left_hemisphere import LeftHemisphereLogicalSLM
from calosum.domain.memory import DualMemorySystem
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
)
from calosum.domain.right_hemisphere import RightHemisphereJEPA
from calosum.domain.runtime import StrictLambdaRuntime
from calosum.domain.telemetry import CognitiveTelemetryBus
from calosum.domain.verifier import HeuristicVerifier
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


@dataclass(slots=True)
class BranchingBudget:
    max_width: int = 3
    max_depth: int = 1

@dataclass(slots=True)
class CalosumAgentConfig:
    max_runtime_retries: int = 2
    surprise_threshold: float = 0.6
    awareness_interval_turns: int = 1
    branching_budget: BranchingBudget = field(default_factory=BranchingBudget)


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
        verifier: VerifierPort | None = None,
        config: CalosumAgentConfig | None = None,
        capability_snapshot: CapabilityDescriptor | None = None,
        evolution_archive: "JsonlEvolutionArchive | None" = None,
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
        self.event_bus = InternalEventBus()
        self.execution_engine = AgentExecutionEngine(
            action_runtime=self.action_runtime,
            max_runtime_retries=self.config.max_runtime_retries,
            verifier=self.verifier,
        )
        
        # Último workspace por sessão
        self.last_workspace_by_session: dict[str, CognitiveWorkspace] = {}
        self.latest_awareness_by_session: dict[str, SessionDiagnostic] = {}
        self._awareness_turn_counts: dict[str, int] = {}
        self.approved_prompt_directives: list[str] = (
            self.evolution_archive.load_applied_prompt_directives()
            if self.evolution_archive is not None
            else []
        )
        
        self.pending_directives: list[EvolutionDirective] = (
            self.evolution_archive.load_pending_directives()
            if self.evolution_archive is not None
            else []
        )

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

        # Active Inference: High surprise triggers Metacognition (Group Turn)
        surprise_score = getattr(right_state, "surprise_score", 0.0)
        ambiguity_score = right_state.world_hypotheses.get("interaction_complexity", 0.0)
        
        needs_branching = (
            self.config.branching_budget.max_depth > 0
            and (
                surprise_score > self.config.surprise_threshold
                or ambiguity_score > 0.8
            )
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
            bridge_directives=self.approved_prompt_directives,
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

        for variant in variants:
            variant_started_at = perf_counter()
            tokenizer = self._build_variant_tokenizer(variant)
            left_hemisphere = self._build_variant_left_hemisphere(variant)
            turn_result = await self.execution_engine.run_variant(
                user_turn=user_turn,
                memory_context=memory_context,
                right_state=right_state,
                tokenizer=tokenizer,
                left_hemisphere=left_hemisphere,
                variant_label=variant.variant_id,
                bridge_directives=self.approved_prompt_directives + list(variant.bridge_directives),
                capabilities=capabilities_dict,
                workspace=workspace,
            )
            turn_result.latency_ms = round((perf_counter() - variant_started_at) * 1000.0, 3)
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
        
        # Awareness loop for group turn
        await self._awareness_loop(user_turn.session_id)

        return GroupTurnResult(
            user_turn=user_turn,
            right_state=right_state,
            candidates=candidates,
            selected_result=selected_candidate.turn_result,
            reflection=reflection,
        )

    async def _awareness_loop(self, session_id: str) -> None:
        """
        Gera diagnósticos persistidos e aplica apenas diretivas paramétricas.
        """
        self._awareness_turn_counts[session_id] = self._awareness_turn_counts.get(session_id, 0) + 1
        if self._awareness_turn_counts[session_id] % max(1, self.config.awareness_interval_turns) != 0:
            return

        from calosum.domain.evolution import EvolutionProposer

        diagnostic = self.analyze_session(session_id, persist=True)
        if not diagnostic.bottlenecks:
            return

        proposer = EvolutionProposer()
        directives = proposer.propose(diagnostic)

        for directive in directives:
            if directive.directive_type == DirectiveType.PARAMETER:
                self._apply_directive(directive)
                self._record_directive_event(directive, event="auto_applied")
                continue
            self._queue_directive(directive)

    def analyze_session(self, session_id: str, *, persist: bool = False) -> SessionDiagnostic:
        dashboard = self.cognitive_dashboard(session_id)
        if not dashboard.get("decision"):
            diagnostic = SessionDiagnostic(
                session_id=session_id,
                analyzed_turns=0,
                tool_success_rate=1.0,
                average_retries=0.0,
                average_surprise=0.0,
                bottlenecks=[],
                pending_approval_backlog=0,
                pending_directive_count=len(self.pending_directives),
            )
        else:
            from calosum.domain.introspection import IntrospectionEngine

            engine = IntrospectionEngine()
            diagnostic = engine.analyze(
                session_id,
                dashboard,
                pending_directive_count=len(self.pending_directives),
            )

        self.latest_awareness_by_session[session_id] = diagnostic
        if persist:
            if hasattr(self.telemetry_bus, "record_awareness"):
                self.telemetry_bus.record_awareness(session_id, diagnostic)
            if self.evolution_archive is not None:
                self.evolution_archive.record_diagnostic(diagnostic)
        return diagnostic

    def latest_awareness_for_session(self, session_id: str | None = None) -> SessionDiagnostic | None:
        if session_id is not None:
            return self.latest_awareness_by_session.get(session_id)
        if self.latest_awareness_by_session:
            last_key = list(self.latest_awareness_by_session.keys())[-1]
            return self.latest_awareness_by_session[last_key]
        return None

    def workspace_for_session(self, session_id: str | None = None) -> CognitiveWorkspace | None:
        if session_id is not None:
            return self.last_workspace_by_session.get(session_id)
        if self.last_workspace_by_session:
            last_key = list(self.last_workspace_by_session.keys())[-1]
            return self.last_workspace_by_session[last_key]
        return None

    def apply_pending_directive(self, directive_id: str) -> EvolutionDirective | None:
        directive = next((item for item in self.pending_directives if item.directive_id == directive_id), None)
        if directive is None:
            return None

        self.pending_directives = [
            item for item in self.pending_directives if item.directive_id != directive_id
        ]
        self._apply_directive(directive)
        self._record_directive_event(directive, event="manual_apply")
        return directive

    def _apply_directive(self, directive: EvolutionDirective) -> None:
        target = directive.target_component
        changes = directive.proposed_change
        
        try:
            if target == "orchestrator":
                for k, v in changes.items():
                    if hasattr(self.config, k):
                        setattr(self.config, k, v)
                directive.status = "applied"
            elif target == "orchestrator.branching_budget":
                for k, v in changes.items():
                    if hasattr(self.config.branching_budget, k):
                        setattr(self.config.branching_budget, k, v)
                directive.status = "applied"
            elif target == "bridge":
                if hasattr(self.tokenizer, "config"):
                    for k, v in changes.items():
                        if hasattr(self.tokenizer.config, k):
                            setattr(self.tokenizer.config, k, v)
                    directive.status = "applied"
            elif directive.directive_type == DirectiveType.PROMPT and target in {
                "left_hemisphere",
                "reflection_controller",
            }:
                instruction = str(changes.get("instruction", "")).strip()
                if instruction and instruction not in self.approved_prompt_directives:
                    self.approved_prompt_directives.append(instruction)
                directive.status = "applied"
            else:
                directive.status = "rejected_target_unknown"
        except Exception:
            directive.status = "failed"

    def _queue_directive(self, directive: EvolutionDirective) -> None:
        fingerprint = self._directive_fingerprint(directive)
        existing = {
            self._directive_fingerprint(item)
            for item in self.pending_directives
            if item.status == "pending"
        }
        if fingerprint in existing:
            return
        directive.status = "pending"
        self.pending_directives.append(directive)
        self._record_directive_event(directive, event="queued")

    def _record_directive_event(self, directive: EvolutionDirective, *, event: str) -> None:
        if self.evolution_archive is not None:
            self.evolution_archive.record_directive(directive, event=event)

    def _directive_fingerprint(self, directive: EvolutionDirective) -> str:
        return json.dumps(
            {
                "directive_type": directive.directive_type.value,
                "target_component": directive.target_component,
                "proposed_change": directive.proposed_change,
            },
            sort_keys=True,
        )

    def sleep_mode(self):
        return run_sync(self.asleep_mode())

    async def asleep_mode(self):
        return await maybe_await(
            self.execution_engine.call_component(
                self.memory_system, "asleep_mode", "sleep_mode"
            )
        )

    def cognitive_dashboard(self, session_id: str | None = None) -> dict[str, list[dict]]:
        return self.telemetry_bus.dashboard_for_session(session_id)

    async def _store_selected_episode(self, result: AgentTurnResult) -> None:
        episode = MemoryEpisode(
            episode_id=str(uuid4()),
            recorded_at=utc_now(),
            user_turn=result.user_turn,
            right_state=result.right_state,
            bridge_packet=result.bridge_packet,
            left_result=result.left_result,
            execution_results=result.execution_results,
            runtime_retry_count=result.runtime_retry_count,
            critique_revision_count=result.critique_revision_count,
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
