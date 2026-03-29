from __future__ import annotations

from typing import Any

from calosum.shared.types import ActionExecutionResult, CritiqueVerdict, LeftHemisphereResult, UserTurn


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
    ) -> CritiqueVerdict:
        issues: list[str] = []
        fixes: list[str] = []
        reasoning: list[str] = []

        # 1. Unsafe wording
        lower_response = left_result.response_text.lower()
        if "ignor" in lower_response and "instruç" in lower_response or "desconsidere" in lower_response:
            issues.append("Unsafe wording detected (prompt injection or ignore instructions)")
            fixes.append("Rewrite response to avoid repeating unsafe user commands.")

        # 2. Tool mismatch / Schema invalid
        for action in left_result.actions:
            if not action.action_type or action.action_type == "unknown":
                issues.append(f"Invalid action type: {action.action_type}")
                fixes.append(f"Use a valid action from the registry instead of {action.action_type}.")
            
            if not action.typed_signature or action.typed_signature == "Any -> Any":
                issues.append(f"Missing typed signature for action {action.action_type}")
                fixes.append("Provide a proper typed signature.")

        # 3. Execution failures that require structural changes
        rejected_count = sum(1 for r in execution_results if r.status == "rejected")
        if rejected_count > 0:
            issues.append(f"{rejected_count} actions were rejected by runtime.")
            fixes.append("Fix the action payload or remove unauthorized actions.")

        is_valid = len(issues) == 0
        if is_valid:
            reasoning.append("Result looks structurally valid and safe.")
            confidence = 1.0
        else:
            reasoning.append(f"Found {len(issues)} issues during critique.")
            # Reduces confidence as issues grow
            confidence = max(0.5, 1.0 - (len(issues) * 0.1))

        return CritiqueVerdict(
            is_valid=is_valid,
            critique_reasoning=reasoning,
            identified_issues=issues,
            suggested_fixes=fixes,
            confidence=confidence,
        )

    async def averify(
        self,
        user_turn: UserTurn,
        left_result: LeftHemisphereResult,
        execution_results: list[ActionExecutionResult],
    ) -> CritiqueVerdict:
        return self.verify(user_turn, left_result, execution_results)
