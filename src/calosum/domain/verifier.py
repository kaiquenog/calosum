from __future__ import annotations

from calosum.shared.schemas import collect_left_result_schema_issues
from calosum.shared.types import ActionExecutionResult, CritiqueVerdict, FailureType, LeftHemisphereResult, UserTurn, CognitiveWorkspace
from calosum.shared.types import FailureType


class HeuristicVerifier:
    """
    Verificador heurístico para avaliar o resultado do Hemisfério Esquerdo
    antes de finalizar a execução (CRITIC-like passo).
    """

    def verify(
        self,
        user_turn: UserTurn,
        left_result: LeftHemisphereResult,
        execution_results: list[ActionExecutionResult],
        workspace: CognitiveWorkspace | None = None,
    ) -> CritiqueVerdict:
        issues: list[str] = []
        fixes: list[str] = []
        reasoning: list[str] = []
        failure_types: list[FailureType] = []

        self._check_response_safety(left_result, issues, fixes, failure_types)
        self._check_result_schema(left_result, issues, fixes, failure_types)
        self._check_result_completeness(left_result, issues, fixes, failure_types)
        self._check_execution_results(execution_results, issues, fixes, failure_types)

        unique_failure_types = list(dict.fromkeys(failure_types))

        is_valid = len(issues) == 0
        if is_valid:
            reasoning.append("Result looks structurally valid and safe.")
            confidence = 1.0
        else:
            categories = ", ".join(item.value for item in unique_failure_types) or "unknown"
            reasoning.append(f"Found {len(issues)} issues during critique.")
            reasoning.append(f"Failure taxonomy: {categories}.")
            confidence = max(0.35, 1.0 - (len(issues) * 0.1) - (len(unique_failure_types) * 0.05))

        verdict = CritiqueVerdict(
            is_valid=is_valid,
            critique_reasoning=reasoning,
            identified_issues=issues,
            suggested_fixes=fixes,
            confidence=confidence,
            failure_types=unique_failure_types,
        )

        if workspace is not None and issues:
            workspace.verifier_feedback.append({
                "issues": issues,
                "fixes": fixes,
                "failure_types": [f.value for f in unique_failure_types],
            })

        return verdict

    async def averify(
        self,
        user_turn: UserTurn,
        left_result: LeftHemisphereResult,
        execution_results: list[ActionExecutionResult],
        workspace: CognitiveWorkspace | None = None,
    ) -> CritiqueVerdict:
        return self.verify(user_turn, left_result, execution_results, workspace)

    def _check_response_safety(
        self,
        left_result: LeftHemisphereResult,
        issues: list[str],
        fixes: list[str],
        failure_types: list[FailureType],
    ) -> None:
        lower_response = left_result.response_text.lower()
        if ("ignor" in lower_response and "instru" in lower_response) or "desconsidere" in lower_response:
            issues.append("Unsafe wording detected (prompt injection or ignore instructions).")
            fixes.append("Rewrite the response to avoid echoing unsafe or user-supplied override commands.")
            failure_types.append(FailureType.UNSAFE_CONTENT)

    def _check_result_schema(
        self,
        left_result: LeftHemisphereResult,
        issues: list[str],
        fixes: list[str],
        failure_types: list[FailureType],
    ) -> None:
        for schema_issue in collect_left_result_schema_issues(left_result):
            issues.append(f"Schema violation: {schema_issue}")
            fixes.append("Return a LeftHemisphereResult that matches the typed contract exactly.")
            failure_types.append(FailureType.SCHEMA_VIOLATION)

        for action in left_result.actions:
            if action.action_type == "unknown":
                issues.append("Schema violation: action_type cannot be 'unknown'.")
                fixes.append("Use an explicit action_type from the runtime vocabulary.")
                failure_types.append(FailureType.SCHEMA_VIOLATION)
            if action.typed_signature == "Any -> Any":
                issues.append(f"Schema violation: action {action.action_type} uses a placeholder typed signature.")
                fixes.append("Provide a specific typed_signature for each planned action.")
                failure_types.append(FailureType.SCHEMA_VIOLATION)

        if left_result.lambda_program.signature == "Any -> Any":
            issues.append("Schema violation: lambda_program.signature must be specific.")
            fixes.append("Use a typed lambda signature that reflects the intended decision flow.")
            failure_types.append(FailureType.SCHEMA_VIOLATION)

    def _check_result_completeness(
        self,
        left_result: LeftHemisphereResult,
        issues: list[str],
        fixes: list[str],
        failure_types: list[FailureType],
    ) -> None:
        epistemic_actions = {"search_web", "read_file", "execute_bash", "introspect_self", "code_execution", "http_request"}
        is_foraging = any(action.action_type in epistemic_actions for action in left_result.actions)

        if not is_foraging and not left_result.response_text.strip():
            issues.append("Incomplete result: response_text is empty and no epistemic foraging action was detected.")
            fixes.append("Produce a user-facing response_text before finalizing the turn or use an epistemic tool to gather data.")
            failure_types.append(FailureType.INCOMPLETE_RESULT)

        if not left_result.lambda_program.expression.strip():
            issues.append("Incomplete result: lambda_program.expression is empty.")
            fixes.append("Populate lambda_program.expression with the symbolic plan that justifies the actions.")
            failure_types.append(FailureType.INCOMPLETE_RESULT)

    def _check_execution_results(
        self,
        execution_results: list[ActionExecutionResult],
        issues: list[str],
        fixes: list[str],
        failure_types: list[FailureType],
    ) -> None:
        for result in execution_results:
            if result.status != "rejected":
                continue

            error_type = result.output.get("error_type")
            detail = "; ".join(result.violations) or result.output.get("error", "runtime rejected the action")
            if error_type in {"validation_failed", "tool_not_found"}:
                issues.append(
                    f"Schema violation: runtime rejected action {result.action_type} due to invalid tool contract ({detail})."
                )
                fixes.append("Correct the action_type and payload to match the registered tool schema.")
                failure_types.append(FailureType.SCHEMA_VIOLATION)
                continue

            issues.append(f"Runtime rejection: action {result.action_type} was rejected ({detail}).")
            fixes.append("Remove or redesign the rejected action so it satisfies runtime safety constraints.")
            failure_types.append(FailureType.RUNTIME_REJECTION)
