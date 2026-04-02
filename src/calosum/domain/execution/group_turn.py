from __future__ import annotations
import asyncio
from time import perf_counter
from typing import TYPE_CHECKING, Any

from calosum.domain.metacognition.metacognition import CognitiveCandidate, CognitiveVariantSpec, GroupTurnResult
from calosum.shared.utils.async_utils import maybe_await
from calosum.shared.models.types import CognitiveWorkspace, UserTurn

if TYPE_CHECKING:
    from calosum.domain.agent.orchestrator import CalosumAgent

async def process_group_turn(
    agent: CalosumAgent,
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
        from calosum.domain.execution.workspace import init_turn_workspace
        workspace = init_turn_workspace(agent, user_turn)
        agent.last_workspace_by_session[user_turn.session_id] = workspace
    
    memory_context = _precomputed_memory
    if memory_context is None:
        memory_context = await maybe_await(
            agent.execution_engine.call_component(
                agent.memory_system, "abuild_context", "build_context", user_turn
            )
        )
        
    right_state = _precomputed_right
    if right_state is None:
        right_state = await maybe_await(
            agent.execution_engine.call_component(
                agent.right_hemisphere, "aperceive", "perceive", user_turn, memory_context, workspace
            )
        )
    candidates: list[CognitiveCandidate] = []

    capabilities_dict = None
    if agent.capability_snapshot:
        from dataclasses import asdict
        capabilities_dict = asdict(agent.capability_snapshot)

    async def _run_variant(variant: CognitiveVariantSpec) -> CognitiveCandidate:
        variant_started_at = perf_counter()
        tokenizer = agent.execution_engine.clone_component_with_overrides(agent.tokenizer, variant.tokenizer_overrides)
        left_hemisphere = agent.execution_engine.clone_component_with_overrides(agent.left_hemisphere, variant.left_overrides)
        turn_result = await agent.execution_engine.run_variant(
            user_turn=user_turn,
            memory_context=memory_context,
            right_state=right_state,
            tokenizer=tokenizer,
            left_hemisphere=left_hemisphere,
            variant_label=variant.variant_id,
            bridge_directives=agent.evolution_manager.approved_prompt_directives + list(variant.bridge_directives),
            capabilities=capabilities_dict,
            workspace=workspace,
        )
        turn_result.latency_ms = round((perf_counter() - variant_started_at) * 1000.0, 3)
        return CognitiveCandidate(variant=variant, turn_result=turn_result)

    candidates = list(await asyncio.gather(*(_run_variant(variant) for variant in variants)))

    reflection = await maybe_await(
        agent.execution_engine.call_component(
            agent.reflection_controller,
            "aevaluate",
            "evaluate",
            candidates,
            agent.tokenizer,
        )
    )
    agent.reflection_controller.apply_neuroplasticity(agent.tokenizer, reflection)
    selected_candidate = next(
        item
        for item in candidates
        if item.variant.variant_id == reflection.selected_variant_id
    )
    selected_candidate.turn_result.latency_ms = round((perf_counter() - started_at) * 1000.0, 3)
    await agent._store_selected_episode(selected_candidate.turn_result)
    await maybe_await(
        agent.execution_engine.call_component(
            agent.telemetry_bus,
            "arecord_turn",
            "record_turn",
            selected_candidate.turn_result,
        )
    )
    await maybe_await(
        agent.execution_engine.call_component(
            agent.telemetry_bus,
            "arecord_reflection",
            "record_reflection",
            user_turn.session_id,
            user_turn.turn_id,
            reflection.as_dict(),
        )
    )
    
    from calosum.domain.metacognition.awareness import process_awareness_loop
    await process_awareness_loop(agent, user_turn.session_id)

    return GroupTurnResult(
        user_turn=user_turn,
        right_state=right_state,
        candidates=candidates,
        selected_result=selected_candidate.turn_result,
        reflection=reflection,
    )
