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
        variant_label: str | None = None,
        bridge_directives: list[str] | None = None,
        capabilities: dict[str, Any] | None = None,
        workspace: CognitiveWorkspace | None = None,
    ) -> AgentTurnResult:
        bridge_packet = await maybe_await(
            self.call_component(tokenizer, "atranslate", "translate", right_state, workspace)
        )
        bridge_packet = self._apply_variant_bridge_overrides(
            bridge_packet,
            variant_label=variant_label,
            bridge_directives=bridge_directives,
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
                workspace,
            )
        )
        left_result, execution_results, retry_count, critique_revision_count = await self._execute_with_retries(
            user_turn=user_turn,
            bridge_packet=bridge_packet,
            memory_context=memory_context,
            left_hemisphere=left_hemisphere,
            left_result=left_result,
            workspace=workspace,
        )
        telemetry = self._build_telemetry(
            right_state=right_state,
            bridge_packet=bridge_packet,
            left_result=left_result,
            execution_results=execution_results,
            retry_count=retry_count,
            critique_revision_count=critique_revision_count,
            capabilities=capabilities,
            variant_label=variant_label,
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

    def _apply_variant_bridge_overrides(
        self,
        bridge_packet: CognitiveBridgePacket,
        *,
        variant_label: str | None,
        bridge_directives: list[str] | None,
    ) -> CognitiveBridgePacket:
        merged_directives = list(dict.fromkeys((bridge_directives or []) + bridge_packet.control.system_directives))
        updated_control = replace(
            bridge_packet.control,
            system_directives=merged_directives[:6],
            annotations={
                **bridge_packet.control.annotations,
                "variant_label": variant_label,
            },
        )
        return replace(
            bridge_packet,
            control=updated_control,
            bridge_metadata={
                **bridge_packet.bridge_metadata,
                "variant_label": variant_label,
            },
        )

    async def _execute_with_retries(
        self,
        *,
        user_turn: UserTurn,
        bridge_packet: CognitiveBridgePacket,
        memory_context: MemoryContext,
        left_hemisphere: LeftHemispherePort,
        left_result: LeftHemisphereResult,
        workspace: CognitiveWorkspace | None = None,
    ) -> tuple[LeftHemisphereResult, list[ActionExecutionResult], int, int]:
        retry_count = 0
        critique_revision_count = 0
        current_result = left_result

        while True:
            execution_results = await maybe_await(
                self.call_component(self.action_runtime, "arun", "run", current_result, workspace)
            )
            rejected_results = [
                result for result in execution_results if result.status == "rejected"
            ]

            critique_verdict = None
            if self.verifier:
                critique_verdict = await maybe_await(
                    self.call_component(self.verifier, "averify", "verify", user_turn, current_result, execution_results, workspace)
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
                workspace=workspace,
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
        workspace: CognitiveWorkspace | None = None,
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
                
            if workspace is not None:
                if self._supports_workspace_kwarg(repair_method):
                    return await maybe_await(repair_method(*repair_args, workspace=workspace))

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
                workspace,
            )
        )

    def _build_telemetry(
        self,
        *,
        right_state: RightHemisphereState,
        bridge_packet: CognitiveBridgePacket,
        left_result: LeftHemisphereResult,
        execution_results: list[ActionExecutionResult],
        retry_count: int,
        critique_revision_count: int,
        capabilities: dict[str, Any] | None = None,
        variant_label: str | None = None,
    ) -> CognitiveTelemetrySnapshot:
        action_count = len(execution_results)
        executed_count = sum(1 for result in execution_results if result.status == "executed")
        rejected_count = sum(1 for result in execution_results if result.status == "rejected")
        tool_success_rate = round(executed_count / action_count, 3) if action_count else 1.0
        
        bridge_config = {
            "target_temperature": bridge_packet.control.target_temperature,
            "empathy_priority": bridge_packet.control.empathy_priority,
            "system_directives_count": len(bridge_packet.control.system_directives),
        }
        
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
            capabilities=capabilities or {},
            bridge_config=bridge_config,
            active_variant=variant_label or "default",
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

        # Extract workspace if it was passed as the last argument
        # Check if the method actually supports 'workspace' before passing it
        from calosum.shared.types import CognitiveWorkspace
        
        args_list = list(args)
        workspace_arg = None
        if args_list and isinstance(args_list[-1], CognitiveWorkspace) or args_list and args_list[-1] is None:
            # We assume the last arg might be workspace if we passed it from orchestrator
            # Let's be safer:
            pass

        # Since we are passing *args, we should ideally use kwargs for optional injected params, 
        # but the current architecture passes them as positional. Let's do a smart trim.
        
        # If the last argument is workspace (or None meant to be workspace)
        # we check if the method supports it.
        # However, to be completely safe, we should check the method signature's length.
        try:
            sig = inspect.signature(method)
            # Count how many positional arguments the method can take
            # Exclude 'self'
            params = list(sig.parameters.values())
            max_positional = sum(1 for p in params if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD))
            
            if len(args) > max_positional:
                # We have more args than the method accepts. Trim the excess (which should be the workspace).
                # But wait, what if there's *args in the method?
                has_var_args = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
                if not has_var_args:
                    args = args[:max_positional]
        except (TypeError, ValueError):
            pass

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
            feedback.extend([f"FAILURE_TYPE: {item.value}" for item in critique_verdict.failure_types])
            feedback.extend([f"CRITIQUE_ISSUE: {issue}" for issue in critique_verdict.identified_issues])
            feedback.extend([f"SUGGESTED_FIX: {fix}" for fix in critique_verdict.suggested_fixes])
        return feedback

    def _supports_workspace_kwarg(self, method: Any) -> bool:
        try:
            parameters = inspect.signature(method).parameters.values()
        except (TypeError, ValueError):
            return False

        for parameter in parameters:
            if parameter.name == "workspace":
                return True
            if parameter.kind in {
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            }:
                return True
        return False

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
