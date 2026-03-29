from __future__ import annotations

from dataclasses import replace
import inspect
from typing import Any

from calosum.shared.async_utils import maybe_await
from calosum.shared.ports import (
    ActionRuntimePort,
    CognitiveTokenizerPort,
    LeftHemispherePort,
    VerifierPort,
)
from calosum.shared.types import (
    ActionExecutionResult,
    AgentTurnResult,
    CognitiveBridgePacket,
    CognitiveTelemetrySnapshot,
    CritiqueVerdict,
    LeftHemisphereResult,
    MemoryContext,
    RightHemisphereState,
    UserTurn,
)


class AgentExecutionEngine:
    def __init__(
        self,
        action_runtime: ActionRuntimePort,
        max_runtime_retries: int,
        verifier: VerifierPort | None = None,
    ) -> None:
        self.action_runtime = action_runtime
        self.max_runtime_retries = max_runtime_retries
        self.verifier = verifier

    async def run_variant(
        self,
        *,
        user_turn: UserTurn,
        memory_context: MemoryContext,
        right_state: RightHemisphereState,
        tokenizer: CognitiveTokenizerPort,
        left_hemisphere: LeftHemispherePort,
    ) -> AgentTurnResult:
        bridge_packet = await maybe_await(
            self.call_component(tokenizer, "atranslate", "translate", right_state)
        )
        left_result = await maybe_await(
            self.call_component(
                left_hemisphere,
                "areason",
                "reason",
                user_turn,
                bridge_packet,
                memory_context,
                None,
                0,
            )
        )
        left_result, execution_results, retry_count, critique_revision_count = await self._execute_with_retries(
            user_turn=user_turn,
            bridge_packet=bridge_packet,
            memory_context=memory_context,
            left_hemisphere=left_hemisphere,
            left_result=left_result,
        )
        telemetry = self._build_telemetry(
            right_state=right_state,
            left_result=left_result,
            execution_results=execution_results,
            retry_count=retry_count,
            critique_revision_count=critique_revision_count,
        )
        return AgentTurnResult(
            user_turn=user_turn,
            memory_context=memory_context,
            right_state=right_state,
            bridge_packet=bridge_packet,
            left_result=left_result,
            telemetry=telemetry,
            execution_results=execution_results,
            runtime_retry_count=retry_count,
            critique_revision_count=critique_revision_count,
        )

    async def _execute_with_retries(
        self,
        *,
        user_turn: UserTurn,
        bridge_packet: CognitiveBridgePacket,
        memory_context: MemoryContext,
        left_hemisphere: LeftHemispherePort,
        left_result: LeftHemisphereResult,
    ) -> tuple[LeftHemisphereResult, list[ActionExecutionResult], int, int]:
        retry_count = 0
        critique_revision_count = 0
        current_result = left_result

        while True:
            execution_results = await maybe_await(
                self.call_component(self.action_runtime, "arun", "run", current_result)
            )
            rejected_results = [
                result for result in execution_results if result.status == "rejected"
            ]

            critique_verdict = None
            if self.verifier:
                critique_verdict = await maybe_await(
                    self.call_component(self.verifier, "averify", "verify", user_turn, current_result, execution_results)
                )

            is_valid = critique_verdict.is_valid if critique_verdict else not rejected_results

            if is_valid or retry_count >= self.max_runtime_retries:
                return current_result, execution_results, retry_count, critique_revision_count

            retry_count += 1
            if critique_verdict and not critique_verdict.is_valid:
                critique_revision_count += 1

            current_result = await self._repair_left_result(
                left_hemisphere=left_hemisphere,
                user_turn=user_turn,
                bridge_packet=bridge_packet,
                memory_context=memory_context,
                previous_result=current_result,
                rejected_results=rejected_results,
                attempt=retry_count,
                critique_verdict=critique_verdict,
            )

    async def _repair_left_result(
        self,
        *,
        left_hemisphere: LeftHemispherePort,
        user_turn: UserTurn,
        bridge_packet: CognitiveBridgePacket,
        memory_context: MemoryContext,
        previous_result: LeftHemisphereResult,
        rejected_results: list[ActionExecutionResult],
        attempt: int,
        critique_verdict: CritiqueVerdict | None = None,
    ) -> LeftHemisphereResult:
        feedback = self._format_runtime_feedback(rejected_results, critique_verdict)
        
        if hasattr(left_hemisphere, "arepair") or hasattr(left_hemisphere, "repair"):
            repair_method = getattr(left_hemisphere, "arepair", None)
            if repair_method is None:
                repair_method = getattr(left_hemisphere, "repair")

            repair_args = [
                user_turn,
                bridge_packet,
                memory_context,
                previous_result,
                rejected_results,
                attempt,
            ]
            if self._supports_critique_feedback(repair_method):
                repair_args.append(feedback)

            return await maybe_await(repair_method(*repair_args))

        return await maybe_await(
            self.call_component(
                left_hemisphere,
                "areason",
                "reason",
                user_turn,
                bridge_packet,
                memory_context,
                feedback,
                attempt,
            )
        )

    def _build_telemetry(
        self,
        *,
        right_state: RightHemisphereState,
        left_result: LeftHemisphereResult,
        execution_results: list[ActionExecutionResult],
        retry_count: int,
        critique_revision_count: int,
    ) -> CognitiveTelemetrySnapshot:
        action_count = len(execution_results)
        executed_count = sum(1 for result in execution_results if result.status == "executed")
        rejected_count = sum(1 for result in execution_results if result.status == "rejected")
        tool_success_rate = round(executed_count / action_count, 3) if action_count else 1.0
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
                "runtime_retry_count": retry_count,
                "critique_revision_count": critique_revision_count,
            },
            decision={
                "response_text": left_result.response_text,
                "action_types": [action.action_type for action in left_result.actions],
                "action_count": action_count,
                "tool_success_rate": tool_success_rate,
                "runtime_retry_count": retry_count,
                "runtime_rejected_count": rejected_count,
                "critique_revision_count": critique_revision_count,
            },
        )

    def clone_component_with_overrides(self, component: Any, overrides: dict[str, Any]) -> Any:
        config = getattr(component, "config", None)
        if config is None:
            return component
        cloned_config = replace(config)
        for key, value in overrides.items():
            if hasattr(cloned_config, key):
                setattr(cloned_config, key, value)
        try:
            return component.__class__(config=cloned_config)
        except TypeError:
            return component

    def call_component(
        self,
        component: Any,
        async_method_name: str,
        sync_method_name: str,
        *args: Any,
    ) -> Any:
        method = getattr(component, async_method_name, None)
        if method is None:
            method = getattr(component, sync_method_name)
        return method(*args)

    def _format_runtime_feedback(
        self,
        rejected_results: list[ActionExecutionResult],
        critique_verdict: CritiqueVerdict | None = None,
    ) -> list[str]:
        feedback = [
            f"{result.action_type}:{'; '.join(result.violations)}" for result in rejected_results
        ]
        if critique_verdict and not critique_verdict.is_valid:
            feedback.extend([f"CRITIQUE_ISSUE: {issue}" for issue in critique_verdict.identified_issues])
            feedback.extend([f"SUGGESTED_FIX: {fix}" for fix in critique_verdict.suggested_fixes])
        return feedback

    def _supports_critique_feedback(self, method: Any) -> bool:
        try:
            parameters = inspect.signature(method).parameters.values()
        except (TypeError, ValueError):
            return False

        for parameter in parameters:
            if parameter.name == "critique_feedback":
                return True
            if parameter.kind in {
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            }:
                return True
        return False
