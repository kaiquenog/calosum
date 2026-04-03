from __future__ import annotations

import asyncio
import copy
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

from calosum.shared.utils.async_utils import maybe_await, run_sync
from calosum.domain.execution.agent_execution import AgentExecutionEngine
from calosum.domain.agent.agent_config import CalosumAgentConfig
from calosum.domain.cognition.bridge import CognitiveTokenizer
from calosum.domain.infrastructure.event_bus import CognitiveEvent, InternalEventBus
from calosum.domain.cognition.action_planner import ActionPlannerLogicalSLM
from calosum.domain.memory.memory import DualMemorySystem
from calosum.domain.agent.evolution import (
    EvolutionManager,
    JsonlEvolutionArchive,
    EvolutionProposer,
)
from calosum.domain.metacognition.metacognition import (
    CognitiveCandidate,
    GEAReflectionController,
    GroupTurnResult,
    default_cognitive_personas,
)
from calosum.domain.metacognition.branching_policy import decide_branching
from calosum.shared.models.ports import (
    ToolRuntimePort,
    CognitiveTokenizerPort,
    ActionPlannerPort,
    MemorySystemPort,
    ReflectionControllerPort,
    InputPerceptionPort,
    TelemetryBusPort,
    VerifierPort,
    LatentExchangePort,
)
from calosum.domain.cognition.input_perception import InputPerceptionJEPA
from calosum.domain.execution.tool_runtime import ToolRuntime
from calosum.domain.infrastructure.telemetry import CognitiveTelemetryBus
from calosum.domain.infrastructure.verifier import HeuristicVerifier
from calosum.domain.agent.idle_foraging import build_idle_foraging_turn
from calosum.shared.models.types import (
    AgentTurnResult,
    CapabilityDescriptor,
    CognitiveArchitectureMap,
    CognitiveWorkspace,
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
        right_hemisphere: InputPerceptionPort | None = None,
        tokenizer: CognitiveTokenizerPort | None = None,
        left_hemisphere: ActionPlannerPort | None = None,
        memory_system: MemorySystemPort | None = None,
        action_runtime: ToolRuntimePort | None = None,
        telemetry_bus: TelemetryBusPort | None = None,
        reflection_controller: ReflectionControllerPort | None = None,
        verifier: VerifierPort | None = None,
        config: CalosumAgentConfig | None = None,
        capability_snapshot: CapabilityDescriptor | None = None,
        evolution_archive: "JsonlEvolutionArchive | None" = None,
        night_trainer: Any | None = None,
        latent_exchange: LatentExchangePort | None = None,
    ) -> None:
        self.right_hemisphere = right_hemisphere or InputPerceptionJEPA()
        self.tokenizer = tokenizer or CognitiveTokenizer()
        self.left_hemisphere = left_hemisphere or ActionPlannerLogicalSLM()
        self.memory_system = memory_system or DualMemorySystem()
        self.action_runtime = action_runtime or ToolRuntime()
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
        self.evolution_manager = EvolutionManager(self.evolution_archive, EvolutionProposer())
        self.cognitive_diary_path = Path(".calosum-runtime/cognitive_diary.jsonl")
        self.cognitive_diary_path.parent.mkdir(parents=True, exist_ok=True)

        from calosum.domain.metacognition.self_model import build_self_model
        self.self_model: CognitiveArchitectureMap = build_self_model(self)

    def process_turn(self, user_turn: UserTurn) -> AgentTurnResult | GroupTurnResult:
        return run_sync(self.aprocess_turn(user_turn))

    async def aprocess_turn(self, user_turn: UserTurn) -> AgentTurnResult | GroupTurnResult:
        started_at = perf_counter()
        from calosum.domain.execution.workspace import init_turn_workspace
        workspace = await maybe_await(self.memory_system.aload_workspace(user_turn.session_id))
        if workspace is None:
            workspace = init_turn_workspace(self, user_turn)
        else:
            workspace.task_frame["current_turn_id"] = user_turn.turn_id

        self.event_bus.publish(CognitiveEvent("UserTurnEvent", user_turn, user_turn.turn_id))
        memory_context = await maybe_await(self.execution_engine.call_component(self.memory_system, "abuild_context", "build_context", user_turn))
        self.event_bus.publish(CognitiveEvent("MemoryContextEvent", memory_context, user_turn.turn_id))
        right_state = await maybe_await(self.execution_engine.call_component(self.right_hemisphere, "aperceive", "perceive", user_turn, memory_context, workspace))
        
        from calosum.domain.cognition.differentiable_logic import apply_active_inference
        right_state = apply_active_inference(right_state, memory_context)
        if self.latent_exchange:
            peer_latents = await self.latent_exchange.get_peer_latents(user_turn.session_id)
            right_state.telemetry["peer_latents_count"] = len(peer_latents)
            if peer_latents:
                right_state.telemetry["peer_latent_alignment"] = round(
                    self._peer_alignment(right_state.latent_vector, peer_latents),
                    4,
                )

        self.event_bus.publish(CognitiveEvent("PerceptionEvent", right_state, user_turn.turn_id))
        
        from calosum.domain.agent.orchestrator_briefing import build_session_briefing
        workspace.task_frame["session_briefing"] = build_session_briefing(self, user_turn.session_id, right_state=right_state)

        capabilities_dict = None
        if self.capability_snapshot:
            capabilities_dict = asdict(self.capability_snapshot)

        branching = decide_branching(user_turn=user_turn, right_state=right_state, workspace=workspace)
        variants = default_cognitive_personas(branching.candidate_count)
        candidate_workspaces = [workspace] if len(variants) == 1 else [copy.deepcopy(workspace) for _ in variants]
        candidate_results = await asyncio.gather(
            *[
                self.execution_engine.run_variant(
                    user_turn=user_turn,
                    memory_context=memory_context,
                    right_state=right_state,
                    tokenizer=self.tokenizer,
                    left_hemisphere=self.left_hemisphere,
                    variant_label=variant.variant_id,
                    bridge_directives=list(self.evolution_manager.approved_prompt_directives) + list(variant.bridge_directives),
                    capabilities=capabilities_dict,
                    workspace=variant_workspace,
                )
                for variant, variant_workspace in zip(variants, candidate_workspaces)
            ]
        )
        selected_result = candidate_results[0]
        result: AgentTurnResult | GroupTurnResult = candidate_results[0]
        if len(candidate_results) > 1:
            candidates = [
                CognitiveCandidate(variant=variant, turn_result=turn_result)
                for variant, turn_result in zip(variants, candidate_results)
            ]
            reflection = await self.reflection_controller.aevaluate(candidates, self.tokenizer)
            selected_result = next(
                candidate.turn_result
                for candidate in candidates
                if candidate.variant.variant_id == reflection.selected_variant_id
            )
            workspace = candidate_workspaces[
                next(index for index, variant in enumerate(variants) if variant.variant_id == reflection.selected_variant_id)
            ]
            self.reflection_controller.apply_config_adaptation(self.tokenizer, reflection)
            self.reflection_controller.apply_neuroplasticity(self.tokenizer, reflection)
            await maybe_await(
                self.execution_engine.call_component(
                    self.telemetry_bus,
                    "arecord_reflection",
                    "record_reflection",
                    user_turn.session_id,
                    user_turn.turn_id,
                    {
                        **reflection.as_dict(),
                        "scoreboard": [asdict(score) for score in reflection.scoreboard],
                        "notes": list(reflection.notes) + branching.reasons,
                        "bridge_adjustments": reflection.bridge_adjustments,
                        "cost_metrics": {
                            **reflection.cost_metrics,
                            "total_latency_ms": round((perf_counter() - started_at) * 1000.0, 3),
                        },
                    },
                )
            )
            result = GroupTurnResult(
                user_turn=user_turn,
                right_state=right_state,
                candidates=candidates,
                selected_result=selected_result,
                reflection=reflection,
            )

        selected_result.latency_ms = round((perf_counter() - started_at) * 1000.0, 3)
        self.event_bus.publish(CognitiveEvent("ExecutionEvent", selected_result, user_turn.turn_id))
        await self._store_selected_episode(selected_result)
        await maybe_await(self.execution_engine.call_component(self.telemetry_bus, "arecord_turn", "record_turn", selected_result))
        
        # Awareness loop remains for observability but now strictly linear
        await self._awareness_loop(user_turn.session_id, workspace)

        if self.latent_exchange:
            await self.latent_exchange.broadcast_latent(user_turn.session_id, selected_result.right_state.latent_vector)

        if workspace.runtime_feedback:
            workspace.task_frame["previous_runtime_feedback"] = list(workspace.runtime_feedback)

        await maybe_await(self.memory_system.asave_workspace(user_turn.session_id, workspace))
        return result

    async def _awareness_loop(self, session_id: str, workspace: CognitiveWorkspace | None = None) -> None:
        try:
            from calosum.domain.metacognition.awareness import process_awareness_loop
            await process_awareness_loop(self, session_id, workspace)
        except Exception:
            pass # Awareness should not crash the main turn

    def analyze_session(self, session_id: str, *, persist: bool = False) -> SessionDiagnostic:
        return run_sync(self.aanalyze_session(session_id, persist=persist))

    async def aanalyze_session(self, session_id: str, *, persist: bool = False) -> SessionDiagnostic:
        from calosum.domain.metacognition.awareness import analyze_session as _analyze_session
        return await _analyze_session(self, session_id, persist=persist)

    def latest_awareness_for_session(self, sid: str | None = None) -> SessionDiagnostic | None:
        return run_sync(self.alatest_awareness_for_session(sid))

    async def alatest_awareness_for_session(self, sid: str | None = None) -> SessionDiagnostic | None:
        return await maybe_await(self.memory_system.aload_diagnostic(sid)) if sid else None

    def workspace_for_session(self, sid: str | None = None) -> CognitiveWorkspace | None:
        return run_sync(self.aload_workspace_for_session(sid))

    async def aload_workspace_for_session(self, sid: str | None = None) -> CognitiveWorkspace | None:
        return await maybe_await(self.memory_system.aload_workspace(sid)) if sid else None

    @property
    def pending_directives(self) -> list[EvolutionDirective]:
        return self.evolution_manager.pending_directives

    def apply_pending_directive(self, directive_id: str) -> EvolutionDirective | None:
        return self.evolution_manager.apply_pending_directive(directive_id, self)

    def _apply_directive(self, directive: EvolutionDirective) -> None:
        self.evolution_manager.apply_directive(directive, self)

    def sleep_mode(self): return run_sync(self.asleep_mode())
    async def asleep_mode(self):
        report = await maybe_await(self.execution_engine.call_component(self.memory_system, "asleep_mode", "sleep_mode"))
        if self.night_trainer:
            await maybe_await(self.execution_engine.call_component(self.night_trainer, "arun_training_cycle", "run_training_cycle"))
        self.event_bus.publish(CognitiveEvent("SleepModeCompletedEvent", report, "system-loop"))
        return report

    def cognitive_dashboard(self, sid: str | None = None) -> dict[str, list[dict]]: 
        return self.telemetry_bus.dashboard_for_session(sid)

    async def _store_selected_episode(self, result: AgentTurnResult) -> None:
        ep = MemoryEpisode(episode_id=str(uuid4()), recorded_at=utc_now(), user_turn=result.user_turn,
            right_state=result.right_state, bridge_packet=result.bridge_packet, left_result=result.left_result, 
            execution_results=result.execution_results, runtime_retry_count=result.runtime_retry_count, 
            critique_revision_count=result.critique_revision_count)
        await maybe_await(self.execution_engine.call_component(self.memory_system, "astore_episode", "store_episode", ep))

    def _peer_alignment(self, latent_vector: list[float], peer_latents: list[list[float]]) -> float:
        if not latent_vector or not peer_latents:
            return 0.0
        aligned: list[float] = []
        base_norm = sum(value * value for value in latent_vector) ** 0.5
        if base_norm == 0:
            return 0.0
        for peer in peer_latents:
            if len(peer) != len(latent_vector):
                continue
            peer_norm = sum(value * value for value in peer) ** 0.5
            if peer_norm == 0:
                continue
            dot = sum(left * right for left, right in zip(latent_vector, peer))
            aligned.append(dot / (base_norm * peer_norm))
        if not aligned:
            return 0.0
        return max(-1.0, min(1.0, sum(aligned) / len(aligned)))

    def idle_foraging(self) -> AgentTurnResult | GroupTurnResult | None: return run_sync(self.aidle_foraging())
    async def aidle_foraging(self) -> AgentTurnResult | GroupTurnResult | None:
        return await self.aprocess_turn(build_idle_foraging_turn(self.memory_system))
