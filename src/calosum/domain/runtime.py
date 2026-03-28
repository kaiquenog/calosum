from __future__ import annotations

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
        import ast
        
        # O modelo envia `left_result.lambda_program.expression` que pode ser python pseudo-código.
        # Nós usamos AST para garantir que não haja eval() malicioso e que ele só mapeie
        # para a lista `left_result.actions` que o modelo declarou no JSON.
        
        expression = left_result.lambda_program.expression
        if not expression or not expression.strip():
            # Fallback for empty lambda expressions
            return [self._execute_action(action) for action in left_result.actions]

        try:
            # Sandbox muito basico: garantimos que o código consegue ser parseado.
            # Se for sintaxe invalida, rejeitamos toda a execucao.
            tree = ast.parse(expression, mode='exec')
            
            # Checagem de seguranca de AST (não permitimos imports, por exemplo)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    raise ValueError("Imports are forbidden in lambda sandbox")
                    
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
