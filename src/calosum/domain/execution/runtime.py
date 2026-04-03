import json
from dataclasses import dataclass, field

from calosum.shared.models.types import ActionExecutionResult, LeftHemisphereResult, PrimitiveAction


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
            "write_file",
        }
    )


class StrictLambdaRuntime:
    """
    Runtime seguro que executa planos via Structured Outputs (JSON).

    Abandona a DSL LISP artesanal em favor de validacao JSON nativa
    e sequenciamento direto de acoes tipificadas.
    """

    def __init__(self, config: StrictLambdaRuntimeConfig | None = None) -> None:
        self.config = config or StrictLambdaRuntimeConfig()

    def run(self, left_result: LeftHemisphereResult) -> list[ActionExecutionResult]:
        try:
            # Novo parsing pragmático: JSON em vez de AST
            expression = left_result.lambda_program.expression
            plan_data = json.loads(expression)
            
            # Suporte a fallback para string simples ou lista direta, mas prefere {"plan": [...]}
            if isinstance(plan_data, dict):
                plan = plan_data.get("plan", [])
            elif isinstance(plan_data, list):
                plan = plan_data
            else:
                plan = [str(plan_data)]

            # Validação pragmática de alinhamento
            declared_actions = {a.action_type for a in left_result.actions}
            plan_actions = [p for p in plan if isinstance(p, str)]
            
            # Filtra apenas ações que foram declaradas (alignment check simplificado)
            planned_actions = []
            buckets: dict[str, list[PrimitiveAction]] = {}
            for action in left_result.actions:
                buckets.setdefault(action.action_type, []).append(action)

            for action_type in plan_actions:
                if action_type in buckets and buckets[action_type]:
                    planned_actions.append(buckets[action_type].pop(0))
                else:
                    # Se o plano pede algo não declarado, geramos erro de alinhamento
                    return [
                        ActionExecutionResult(
                            action_type="structured_execution",
                            typed_signature="validate_plan",
                            status="rejected",
                            output={"reason": f"action_not_declared: {action_type}"},
                            violations=[f"plan references undeclared action: {action_type}"],
                        )
                    ]

            # Se restarem ações declaradas não utilizadas, também é uma falha de alinhamento
            unused_actions = [t for t, b in buckets.items() if b]
            if unused_actions:
                return [
                    ActionExecutionResult(
                        action_type="structured_execution",
                        typed_signature="validate_plan",
                        status="rejected",
                        output={"reason": "unused_declared_actions", "types": unused_actions},
                        violations=[f"plan does not reference declared actions: {unused_actions}"],
                    )
                ]

        except (json.JSONDecodeError, TypeError) as exc:
            return [
                ActionExecutionResult(
                    action_type="structured_execution",
                    typed_signature="parse_json_plan",
                    status="rejected",
                    output={"error": str(exc)},
                    violations=[f"structured plan is not valid JSON: {exc}"],
                )
            ]
        except Exception as exc:
            return [
                ActionExecutionResult(
                    action_type="structured_execution",
                    typed_signature="evaluate_plan",
                    status="rejected",
                    output={"error": str(exc)},
                    violations=[f"structured runtime failed: {exc}"],
                )
            ]

        return [self._execute_action(action) for action in planned_actions]

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
