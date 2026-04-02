from __future__ import annotations

from dataclasses import replace
import inspect
from typing import Any

from calosum.shared.utils.async_utils import maybe_await
from calosum.shared.models.ports import (
    ActionRuntimePort,
    CognitiveTokenizerPort,
    LeftHemispherePort,
    VerifierPort,
)
from calosum.shared.models.types import (
    ActionExecutionResult,
    AgentTurnResult,
    CognitiveBridgePacket,
    CognitiveTelemetrySnapshot,
    CritiqueVerdict,
    LeftHemisphereResult,
    MemoryContext,
    RightHemisphereState,
    UserTurn,
    CognitiveWorkspace,
)
from calosum.domain.execution.execution_utils import build_execution_telemetry, ensure_response_text


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
        left_result, execution_results, retry_count, critique_revision_count, critique_verdict = await self._execute_with_retries(
            user_turn=user_turn,
            bridge_packet=bridge_packet,
            memory_context=memory_context,
            left_hemisphere=left_hemisphere,
            left_result=left_result,
            workspace=workspace,
        )
        
        # Bidirectional Cognitive Bridge: System 2 overrides System 1
        # Detect if the logical execution engine flagged a cognitive mismatch in its reasoning
        mismatch_detected = any(
            "mismatch" in text.lower() or "override" in text.lower() or "false alarm" in text.lower()
            for text in left_result.reasoning_summary
        )
        if workspace is not None:
            workspace.left_notes["cognitive_override_detected"] = mismatch_detected
        if mismatch_detected and hasattr(tokenizer, "record_reflection_event"):
            event_payload = {
                "turn_id": getattr(user_turn, "turn_id", "unknown"),
                "event": "cognitive_mismatch_override",
                "right_salience": right_state.salience,
                "right_emotional_labels": right_state.emotional_labels,
                "left_reasoning": left_result.reasoning_summary,
                "note": "System 2 logically overrode System 1's heuristic priming."
            }
            await maybe_await(self.call_component(tokenizer, "record_reflection_event", "record_reflection_event", event_payload))

        telemetry = build_execution_telemetry(
            right_state=right_state, bridge_packet=bridge_packet,
            left_result=left_result, execution_results=execution_results,
            retry_count=retry_count, critique_revision_count=critique_revision_count,
            critique_verdict=critique_verdict,
            capabilities=capabilities, variant_label=variant_label,
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
    ) -> tuple[LeftHemisphereResult, list[ActionExecutionResult], int, int, CritiqueVerdict | None]:
        retry_count = 0
        foraging_steps = 0
        critique_revision_count = 0
        last_critique_verdict: CritiqueVerdict | None = None
        current_result = left_result
        all_execution_results = []
        cumulative_feedback = []

        while True:
            execution_results = await maybe_await(
                self.call_component(self.action_runtime, "arun", "run", current_result, workspace)
            )
            all_execution_results.extend(execution_results)
            
            rejected_results = [
                result for result in execution_results if result.status == "rejected"
            ]
            executed_results = [
                result for result in execution_results if result.status == "executed"
            ]

            critique_verdict = None
            if self.verifier:
                critique_verdict = await maybe_await(
                    self.call_component(self.verifier, "averify", "verify", user_turn, current_result, execution_results, workspace)
                )
                last_critique_verdict = critique_verdict

            if workspace is not None:
                workspace.runtime_feedback.append(
                    {
                        "attempt": retry_count,
                        "executed_count": len(executed_results),
                        "rejected_count": len(rejected_results),
                        "tool_success_rate": round(
                            len(executed_results) / max(1, len(execution_results)),
                            3,
                        ),
                        "critique_valid": critique_verdict.is_valid if critique_verdict else None,
                    }
                )

            is_valid = critique_verdict.is_valid if critique_verdict else not rejected_results

            epistemic_actions = {"search_web", "read_file", "execute_bash", "introspect_self", "code_execution", "http_request"}
            has_observations = any(res.action_type in epistemic_actions for res in executed_results)
            
            needs_observation_loop = is_valid and has_observations

            if (is_valid and not needs_observation_loop) or retry_count >= self.max_runtime_retries or foraging_steps >= 5:
                res_text = ensure_response_text(current_result, all_execution_results)
                finalized_summary = list(current_result.reasoning_summary)
                if not current_result.response_text.strip() and res_text.strip():
                    finalized_summary.append("response_text_fallback=runtime_output")
                
                finalized_result = replace(current_result, response_text=res_text, reasoning_summary=finalized_summary)
                return finalized_result, all_execution_results, retry_count, critique_revision_count, last_critique_verdict

            if not is_valid:
                retry_count += 1
            if needs_observation_loop:
                foraging_steps += 1

            if critique_verdict and not critique_verdict.is_valid:
                critique_revision_count += 1

            # Format new feedback
            new_feedback = self._format_runtime_feedback(rejected_results, executed_results, critique_verdict)
            cumulative_feedback.extend(new_feedback)

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
                cumulative_feedback=cumulative_feedback,
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
        cumulative_feedback: list[str] | None = None,
    ) -> LeftHemisphereResult:
        feedback = cumulative_feedback if cumulative_feedback is not None else self._format_runtime_feedback(rejected_results, [], critique_verdict)
        
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

    def clone_component_with_overrides(self, component: Any, overrides: dict[str, Any]) -> Any:
        config = getattr(component, "config", None)
        if config is None: return component
        cloned_config = replace(config)
        for k, v in overrides.items():
            if hasattr(cloned_config, k): setattr(cloned_config, k, v)
        try:
            return component.__class__(config=cloned_config)
        except Exception: return component

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
        from calosum.shared.models.types import CognitiveWorkspace
        
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
        executed_results: list[ActionExecutionResult],
        critique_verdict: CritiqueVerdict | None = None,
    ) -> list[str]:
        feedback = [
            f"{result.action_type}:{'; '.join(result.violations)}" for result in rejected_results
        ]
        
        # Epistemic loop feedback: if we successfully executed a tool, feed the result back
        for res in executed_results:
            if res.action_type in {"execute_bash", "read_file", "search_web", "introspect_self", "code_execution", "http_request"}:
                out = res.output.get("result") or res.output.get("message") or res.output
                feedback.append(f"OBSERVATION from {res.action_type}: {out}")

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
