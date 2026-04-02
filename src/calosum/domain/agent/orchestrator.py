from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

from calosum.shared.utils.async_utils import maybe_await, run_sync
from calosum.domain.agent.directive_guardrails import (
    apply_controlled_right_hemisphere_params,
    apply_runtime_contract_audit_directive,
)
from calosum.domain.execution.agent_execution import AgentExecutionEngine
from calosum.domain.agent.agent_config import CalosumAgentConfig, BranchingBudget
from calosum.domain.cognition.bridge import CognitiveTokenizer
from calosum.domain.infrastructure.event_bus import CognitiveEvent, InternalEventBus
from calosum.domain.cognition.left_hemisphere import LeftHemisphereLogicalSLM
from calosum.domain.memory.memory import DualMemorySystem
from calosum.domain.agent.evolution import (
    EvolutionManager,
    JsonlEvolutionArchive,
    EvolutionProposer,
)
from calosum.domain.metacognition.metacognition import (
    CognitiveCandidate,
    CognitiveVariantSpec,
    GEAReflectionController,
    GroupTurnResult,
    default_cognitive_personas,
)
from calosum.shared.models.ports import (
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
from calosum.domain.cognition.right_hemisphere import RightHemisphereJEPA
from calosum.domain.execution.runtime import StrictLambdaRuntime
from calosum.domain.infrastructure.telemetry import CognitiveTelemetryBus
from calosum.domain.infrastructure.verifier import HeuristicVerifier
from calosum.domain.agent.idle_foraging import build_idle_foraging_turn
from calosum.shared.models.types import (
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
        self.cognitive_diary_path = Path(".calosum-runtime/cognitive_diary.jsonl")
        self.cognitive_diary_path.parent.mkdir(parents=True, exist_ok=True)

        from calosum.domain.metacognition.self_model import build_self_model
        self.self_model: CognitiveArchitectureMap = build_self_model(self)

    def process_turn(self, user_turn: UserTurn) -> AgentTurnResult | GroupTurnResult:
        return run_sync(self.aprocess_turn(user_turn))

    async def aprocess_turn(self, user_turn: UserTurn) -> AgentTurnResult | GroupTurnResult:
        started_at = perf_counter()
        
        from calosum.domain.execution.workspace import init_turn_workspace
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
        uncertainty = float(right_state.telemetry.get("jepa_uncertainty", 0.0))
        ignore_surprise = bool(right_state.telemetry.get("ignore_surprise_for_branching", False)) or uncertainty > 0.7
        effective_surprise = 0.0 if ignore_surprise else surprise_score
        if surprise_score < 0.3:
            surprise_band = "low"
        elif surprise_score <= 0.6:
            surprise_band = "medium"
        else:
            surprise_band = "high"
        right_state.telemetry["surprise_band"] = surprise_band
        right_state.telemetry["ignore_surprise_for_branching"] = ignore_surprise
        workspace.right_notes["surprise_band"] = surprise_band
        workspace.right_notes["ignore_surprise_for_branching"] = ignore_surprise
        workspace.task_frame["session_briefing"] = self.build_session_briefing(
            user_turn.session_id,
            right_state=right_state,
            last_n=10,
        )

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
            and (effective_surprise > 0.6 or expected_free_energy > 0.75)
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
        from calosum.domain.execution.group_turn import process_group_turn as _pgt
        return await _pgt(
            self, user_turn, variants, 
            _precomputed_memory=_precomputed_memory, 
            _precomputed_right=_precomputed_right, 
            _precomputed_workspace=_precomputed_workspace
        )

    async def _awareness_loop(self, session_id: str) -> None:
        from calosum.domain.metacognition.awareness import process_awareness_loop
        await process_awareness_loop(self, session_id)
        diagnostic = self.latest_awareness_for_session(session_id)
        if diagnostic is None:
            return
        if diagnostic.failure_types:
            dominant_failure = max(
                diagnostic.failure_types.items(),
                key=lambda item: item[1],
            )
            self._record_cognitive_diary(
                turn_id=f"{session_id}-{diagnostic.analyzed_turns}",
                observation=(
                    f"Falhas recorrentes detectadas: {dominant_failure[0]} "
                    f"({dominant_failure[1]}x nos últimos turnos)."
                ),
                action="anotado em workspace.left_notes[sandbox_constraints]",
                confidence=max(0.55, min(0.98, 1.0 - diagnostic.average_surprise)),
            )

    def _record_cognitive_diary(
        self,
        *,
        turn_id: str,
        observation: str,
        action: str,
        confidence: float,
    ) -> None:
        payload = {
            "turn_id": turn_id,
            "observation": observation,
            "action": action,
            "confidence": round(max(0.0, min(1.0, confidence)), 3),
            "recorded_at": utc_now().isoformat(),
        }
        with self.cognitive_diary_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def analyze_session(self, session_id: str, *, persist: bool = False) -> SessionDiagnostic:
        from calosum.domain.metacognition.awareness import analyze_session as _analyze_session
        return _analyze_session(self, session_id, persist=persist)

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

    def build_session_briefing(
        self,
        session_id: str,
        *,
        right_state: Any | None = None,
        last_n: int = 10,
    ) -> str:
        dashboard = self.cognitive_dashboard(session_id)
        decisions = list(dashboard.get("decision", []))[-last_n:]
        felt = list(dashboard.get("felt", []))[-last_n:]
        executions = list(dashboard.get("execution", []))[-last_n:]
        turn_number = len(dashboard.get("decision", [])) + 1
        def _avg(rows: list[dict[str, Any]], key: str, default: float) -> float:
            if not rows:
                return default
            return sum(float(item.get(key, default)) for item in rows) / len(rows)
        tool_success_rate = _avg(decisions, "tool_success_rate", 1.0)
        avg_retries = _avg(decisions, "runtime_retry_count", 0.0)
        avg_surprise = _avg(felt, "surprise_score", 0.0)
        uncertainty = None
        if right_state is not None:
            uncertainty = right_state.telemetry.get("jepa_uncertainty")
        if uncertainty is None and felt:
            telemetry_values = [entry.get("telemetry", {}) for entry in felt]
            uncertainty_values = [
                float(item.get("jepa_uncertainty"))
                for item in telemetry_values
                if isinstance(item, dict) and item.get("jepa_uncertainty") is not None
            ]
            if uncertainty_values:
                uncertainty = sum(uncertainty_values) / len(uncertainty_values)
        uncertainty = float(uncertainty or 0.0)
        failures: dict[tuple[str, str], int] = {}
        for event in executions:
            for result in event.get("results", []):
                if result.get("status") != "rejected":
                    continue
                action_type = str(result.get("action_type", "unknown_tool"))
                output = result.get("output", {})
                if isinstance(output, dict):
                    error_type = str(output.get("error_type", "runtime_rejection")).upper()
                else:
                    error_type = "RUNTIME_REJECTION"
                key = (error_type, action_type)
                failures[key] = failures.get(key, 0) + 1
        dominant_failure = "none"
        if failures:
            (error_type, action_type), count = max(failures.items(), key=lambda item: item[1])
            dominant_failure = f"{error_type} in {action_type} ({count}x)"
        pending = [item for item in self.pending_directives if item.status == "pending"]
        pending_summary = ", ".join(
            f"{directive.target_component}:{directive.reasoning[:50].strip()}"
            for directive in pending[:2]
        ) if pending else "none"
        threshold_note = "below 80% threshold" if tool_success_rate < 0.8 else "within healthy band"
        return (
            f"[SESSION BRIEFING - Turn {turn_number} | session: {session_id}]\n"
            f"Tool success rate (last {last_n}): {tool_success_rate:.0%} ({threshold_note})\n"
            f"Avg retries: {avg_retries:.2f}\n"
            f"Dominant failure: {dominant_failure}\n"
            f"Avg surprise: {avg_surprise:.2f} | JEPA uncertainty: {uncertainty:.2f}\n"
            f"Active evolution directives: {pending_summary}"
        )

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
