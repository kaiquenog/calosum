from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

from calosum.shared.types import ActionExecutionResult, LeftHemisphereResult, PrimitiveAction


@dataclass(slots=True)
class StrictLambdaRuntimeConfig:
    allow_external_side_effects: bool = False
    reject_unknown_actions: bool = True
    executable_actions: set[str] = field(
        default_factory=lambda: {
            "respond_text", 
            "load_semantic_rules", 
            "propose_plan",
            "search_web",
            "write_file"
        }
    )


class StrictLambdaRuntime:
    """
    Runtime funcional estrito para a fronteira operacional do hemisferio esquerdo.

    A implementacao real avalia a arvore sintática (AST) restrita
    da string do lambda program e garante que as acoes executadas
    obedecem ao controle de fluxo desenhado.
    """

    def __init__(self, config: StrictLambdaRuntimeConfig | None = None) -> None:
        self.config = config or StrictLambdaRuntimeConfig()

    def run(self, left_result: LeftHemisphereResult) -> list[ActionExecutionResult]:
        # O modelo envia `left_result.lambda_program.expression` que pode ser python pseudo-código.
        # Nós usamos AST para garantir que não haja eval() malicioso e que ele só mapeie
        # para a lista `left_result.actions` que o modelo declarou no JSON.
        
        expression = left_result.lambda_program.expression
        if not expression or not expression.strip():
            # Fallback for empty lambda expressions
            return [self._execute_action(action) for action in left_result.actions]

        try:
            if not expression.lstrip().startswith("("):
                # Sandbox muito basico: garantimos que o código consegue ser parseado.
                # Se for sintaxe invalida, rejeitamos toda a execucao.
                tree = ast.parse(expression, mode='exec')
                
                # Checagem de seguranca de AST (não permitimos imports, por exemplo)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        raise ValueError("Imports are forbidden in lambda sandbox")

            alignment_violations = self._validate_program_alignment(
                expression,
                left_result.actions,
            )
            if alignment_violations:
                validation_violations = [
                    violation
                    for action in left_result.actions
                    for violation in self._validate_action(action)
                ]
                return [
                    ActionExecutionResult(
                        action_type="lambda_evaluation",
                        typed_signature="validate_program_alignment",
                        status="rejected",
                        output={"reason": "lambda_action_mismatch"},
                        violations=alignment_violations + validation_violations,
                    )
                ]
                    
        except Exception as e:
            return [ActionExecutionResult(
                action_type="lambda_evaluation",
                typed_signature="evaluate_lambda",
                status="rejected",
                output={"error": str(e)},
                violations=[f"AST Sandboxing failed: {e}"]
            )]

        # Se o AST for seguro, nós ignoramos a execução imperativa da string (por segurança)
        # e apenas liberamos as PrimitiveActions (que já foram previamente declaradas no JSON).
        # Em uma iteração mais robusta, nós criaríamos um `ast.NodeVisitor` para executar a árvore.
        return [self._execute_action(action) for action in left_result.actions]

    async def arun(self, left_result: LeftHemisphereResult) -> list[ActionExecutionResult]:
        return self.run(left_result)

    def _execute_action(self, action: PrimitiveAction) -> ActionExecutionResult:
        violations = self._validate_action(action)
        if violations:
            return ActionExecutionResult(
                action_type=action.action_type,
                typed_signature=action.typed_signature,
                status="rejected",
                output={"reason": "validation_failed"},
                violations=violations,
            )

        if action.action_type == "respond_text":
            return ActionExecutionResult(
                action_type=action.action_type,
                typed_signature=action.typed_signature,
                status="executed",
                output={
                    "message": action.payload.get("text", ""),
                    "temperature": action.payload.get("temperature"),
                },
            )

        if action.action_type == "load_semantic_rules":
            rules = action.payload.get("rules", [])
            return ActionExecutionResult(
                action_type=action.action_type,
                typed_signature=action.typed_signature,
                status="executed",
                output={"rule_count": len(rules), "rules": rules},
            )

        if action.action_type == "propose_plan":
            steps = action.payload.get("steps", [])
            return ActionExecutionResult(
                action_type=action.action_type,
                typed_signature=action.typed_signature,
                status="executed",
                output={
                    "step_count": len(steps),
                    "steps": steps,
                    "style": action.payload.get("style", "standard"),
                },
            )

        return ActionExecutionResult(
            action_type=action.action_type,
            typed_signature=action.typed_signature,
            status="planned",
            output={"reason": "accepted_but_not_executed_by_skeleton_runtime"},
        )

    def _validate_action(self, action: PrimitiveAction) -> list[str]:
        violations: list[str] = []
        if not action.typed_signature.strip():
            violations.append("typed signature is required")
        if not action.safety_invariants:
            violations.append("at least one safety invariant is required")

        known_action = action.action_type in self.config.executable_actions
        if self.config.reject_unknown_actions and not known_action:
            violations.append(f"unknown action type: {action.action_type}")

        if not self.config.allow_external_side_effects and action.action_type.startswith("call_"):
            violations.append("external side effects are disabled in the strict runtime")

        return violations

    def _validate_program_alignment(
        self,
        expression: str,
        actions: list[PrimitiveAction],
    ) -> list[str]:
        allowed_actions = self._allowed_actions_from_expression(expression)
        if allowed_actions is None:
            return []

        violations: list[str] = []
        action_types = [item.action_type for item in actions]
        undeclared = [item for item in action_types if item not in allowed_actions]
        if undeclared:
            violations.append(
                "lambda program does not reference declared action(s): " + ", ".join(sorted(set(undeclared)))
            )

        if not action_types and allowed_actions:
            violations.append("lambda program references actions but action frontier is empty")

        return violations

    def _allowed_actions_from_expression(self, expression: str) -> set[str] | None:
        normalized = expression.strip()
        if not normalized:
            return set()

        if normalized.startswith("("):
            return self._allowed_actions_from_symbolic_expression(normalized)

        try:
            tree = ast.parse(normalized, mode="exec")
        except SyntaxError:
            return None

        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                called = node.func
                if isinstance(called, ast.Name):
                    names.add(called.id)
                elif isinstance(called, ast.Attribute):
                    names.add(called.attr)
        return self._map_function_names_to_actions(names)

    def _allowed_actions_from_symbolic_expression(self, expression: str) -> set[str] | None:
        if "typed_actions" in expression or "emit typed_actions" in expression:
            return None

        names = set(re.findall(r"[a-z_][a-z0-9_]*", expression))
        return self._map_function_names_to_actions(names)

    def _map_function_names_to_actions(self, names: set[str]) -> set[str]:
        aliases = {
            "respond_text": "respond_text",
            "emit_response": "respond_text",
            "plan": "propose_plan",
            "propose_plan": "propose_plan",
            "search_web": "search_web",
            "write_file": "write_file",
            "load_semantic_rules": "load_semantic_rules",
            "call_external_api": "call_external_api",
        }
        return {
            aliases[name]
            for name in names
            if name in aliases
        }
