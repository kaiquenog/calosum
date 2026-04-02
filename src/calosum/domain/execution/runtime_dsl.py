from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any

from calosum.shared.models.types import PrimitiveAction
ACTION_ALIASES = {
    "respond_text": "respond_text",
    "emit_response": "respond_text",
    "load_semantic_rules": "load_semantic_rules",
    "plan": "propose_plan",
    "propose_plan": "propose_plan",
    "search_web": "search_web",
    "write_file": "write_file",
    "call_external_api": "call_external_api",
}

PYTHON_SAFE_NODES = (
    ast.Expression,
    ast.Load,
    ast.Lambda,
    ast.arguments,
    ast.arg,
    ast.Call,
    ast.Name,
    ast.Constant,
    ast.List,
    ast.Tuple,
    ast.IfExp,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.UnaryOp,
    ast.Not,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
)

@dataclass(slots=True)
class _RuntimeContext:
    declared_action_types: list[str]

    def has_action(self, action_type: str) -> bool:
        return action_type in self.declared_action_types

    def action_count(self, action_type: str) -> int:
        return self.declared_action_types.count(action_type)


class LambdaExecutionPlanner:
    def build_execution_plan(
        self,
        expression: str,
        actions: list[PrimitiveAction],
    ) -> list[str]:
        if not expression or not expression.strip():
            return [action.action_type for action in actions]
        normalized = expression.strip()
        context = _RuntimeContext(declared_action_types=[action.action_type for action in actions])
        if normalized.startswith("("):
            parsed = self._parse_symbolic_expression(normalized)
            return self._normalize_action_sequence(
                self._evaluate_symbolic_node(parsed, context),
                context,
            )
        tree = self._parse_safe_python_expression(normalized)
        root = tree.body
        if isinstance(root, ast.Lambda):
            root = root.body
        return self._normalize_action_sequence(
            self._evaluate_python_node(root, context),
            context,
        )

    def validate_program_alignment(
        self,
        plan: list[str],
        actions: list[PrimitiveAction],
    ) -> list[str]:
        violations: list[str] = []
        declared_action_types = [action.action_type for action in actions]
        plan_counts: dict[str, int] = {}
        declared_counts: dict[str, int] = {}
        for action_type in plan:
            plan_counts[action_type] = plan_counts.get(action_type, 0) + 1
        for action_type in declared_action_types:
            declared_counts[action_type] = declared_counts.get(action_type, 0) + 1
        for action_type, count in plan_counts.items():
            declared_count = declared_counts.get(action_type, 0)
            if declared_count == 0:
                violations.append(f"lambda program references undeclared action: {action_type}")
            elif count > declared_count:
                violations.append(
                    f"lambda program requests action {action_type} {count} time(s) but only {declared_count} declared"
                )

        undeclared = [action_type for action_type in declared_action_types if action_type not in plan_counts]
        if undeclared:
            violations.append(
                "lambda program does not reference declared action(s): " + ", ".join(sorted(set(undeclared)))
            )

        if not declared_action_types and plan:
            violations.append("lambda program references actions but action frontier is empty")
        return violations

    def actions_in_plan(
        self,
        plan: list[str],
        actions: list[PrimitiveAction],
    ) -> list[PrimitiveAction]:
        buckets: dict[str, list[PrimitiveAction]] = {}
        for action in actions:
            buckets.setdefault(action.action_type, []).append(action)
        ordered: list[PrimitiveAction] = []
        for action_type in plan:
            bucket = buckets.get(action_type, [])
            if not bucket:
                raise ValueError(f"lambda plan referenced missing action {action_type}")
            ordered.append(bucket.pop(0))
        return ordered

    def _parse_safe_python_expression(self, expression: str) -> ast.Expression:
        tree = ast.parse(expression, mode="eval")
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.Attribute, ast.Subscript)):
                raise ValueError("unsupported node in lambda runtime")
            if not isinstance(node, PYTHON_SAFE_NODES):
                raise ValueError(f"unsafe node {node.__class__.__name__} in lambda runtime")
        return tree

    def _evaluate_python_node(self, node: ast.AST, context: _RuntimeContext) -> Any:
        if isinstance(node, ast.Call):
            callable_name = self._resolve_callable_name(node.func)
            return self._evaluate_python_call(callable_name, node.args, context)
        if isinstance(node, ast.List | ast.Tuple):
            return [self._evaluate_python_node(item, context) for item in node.elts]
        if isinstance(node, ast.IfExp):
            condition = self._coerce_bool(self._evaluate_python_node(node.test, context))
            branch = node.body if condition else node.orelse
            return self._evaluate_python_node(branch, context)
        if isinstance(node, ast.BoolOp):
            values = [self._coerce_bool(self._evaluate_python_node(item, context)) for item in node.values]
            return all(values) if isinstance(node.op, ast.And) else any(values)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return not self._coerce_bool(self._evaluate_python_node(node.operand, context))
        if isinstance(node, ast.Compare):
            return self._evaluate_compare(node, context)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            lowered = node.id.lower()
            if lowered == "true":
                return True
            if lowered == "false":
                return False
            return node.id
        raise ValueError(f"unsupported python lambda construct: {node.__class__.__name__}")

    def _evaluate_python_call(
        self,
        callable_name: str,
        args: list[ast.expr],
        context: _RuntimeContext,
    ) -> Any:
        if callable_name in ACTION_ALIASES:
            return ACTION_ALIASES[callable_name]
        if callable_name == "sequence":
            return [self._evaluate_python_node(item, context) for item in args]
        if callable_name == "emit":
            if len(args) != 1:
                raise ValueError("emit() expects exactly one action reference")
            return self._action_reference_from_python(args[0])
        if callable_name == "all_actions":
            return list(context.declared_action_types)
        if callable_name == "noop":
            return []
        if callable_name == "has_action":
            if len(args) != 1:
                raise ValueError("has_action() expects exactly one action reference")
            action_type = self._action_reference_from_python(args[0])
            return context.has_action(action_type)
        if callable_name == "action_count":
            if len(args) != 1:
                raise ValueError("action_count() expects exactly one action reference")
            action_type = self._action_reference_from_python(args[0])
            return context.action_count(action_type)
        if callable_name == "when":
            if len(args) != 2:
                raise ValueError("when() expects a condition and a branch")
            condition = self._coerce_bool(self._evaluate_python_node(args[0], context))
            if not condition:
                return []
            return self._evaluate_python_node(args[1], context)
        if callable_name == "if_else":
            if len(args) != 3:
                raise ValueError("if_else() expects condition, then and else branches")
            condition = self._coerce_bool(self._evaluate_python_node(args[0], context))
            branch = args[1] if condition else args[2]
            return self._evaluate_python_node(branch, context)
        raise ValueError(f"unsupported runtime helper: {callable_name}")

    def _evaluate_compare(self, node: ast.Compare, context: _RuntimeContext) -> bool:
        current = self._evaluate_python_node(node.left, context)
        for operator, comparator in zip(node.ops, node.comparators):
            right = self._evaluate_python_node(comparator, context)
            if isinstance(operator, ast.Eq):
                ok = current == right
            elif isinstance(operator, ast.NotEq):
                ok = current != right
            elif isinstance(operator, ast.Lt):
                ok = current < right
            elif isinstance(operator, ast.LtE):
                ok = current <= right
            elif isinstance(operator, ast.Gt):
                ok = current > right
            elif isinstance(operator, ast.GtE):
                ok = current >= right
            else:
                raise ValueError(f"unsupported comparison operator: {operator.__class__.__name__}")
            if not ok:
                return False
            current = right
        return True

    def _resolve_callable_name(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        raise ValueError("only direct helper calls are supported in lambda runtime")

    def _parse_symbolic_expression(self, expression: str) -> Any:
        tokens = re.findall(r'\(|\)|"[^"]*"|[^\s()]+', expression)
        if not tokens:
            return []
        def _parse(index: int) -> tuple[Any, int]:
            token = tokens[index]
            if token != "(":
                return self._normalize_symbolic_atom(token), index + 1
            items: list[Any] = []
            index += 1
            while index < len(tokens) and tokens[index] != ")":
                item, index = _parse(index)
                items.append(item)
            if index >= len(tokens):
                raise ValueError("unbalanced symbolic lambda expression")
            return items, index + 1
        parsed, next_index = _parse(0)
        if next_index != len(tokens):
            raise ValueError("trailing tokens in symbolic lambda expression")
        return parsed

    def _normalize_symbolic_atom(self, token: str) -> Any:
        if token.startswith('"') and token.endswith('"'):
            return token[1:-1]
        lowered = token.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        if token.isdigit():
            return int(token)
        return token

    def _evaluate_symbolic_node(self, node: Any, context: _RuntimeContext) -> Any:
        if not isinstance(node, list):
            return node
        if not node:
            return []
        head = node[0]
        args = node[1:]
        if head == "lambda":
            return self._evaluate_symbolic_node(args[-1], context) if args else []
        if head in {"sequence", "synthesize"}:
            return [self._evaluate_symbolic_node(item, context) for item in args]
        if head == "emit":
            if not args:
                raise ValueError("emit requires at least one action reference")
            if args == ["typed_actions"]:
                return list(context.declared_action_types)
            return [ACTION_ALIASES.get(item, item) for item in args]
        if head == "when":
            if len(args) != 2:
                raise ValueError("when requires condition and branch")
            if not self._coerce_bool(self._evaluate_symbolic_node(args[0], context)):
                return []
            return self._evaluate_symbolic_node(args[1], context)
        if head == "if":
            if len(args) != 3:
                raise ValueError("if requires condition, then and else")
            condition = self._coerce_bool(self._evaluate_symbolic_node(args[0], context))
            return self._evaluate_symbolic_node(args[1] if condition else args[2], context)
        if head == "has_action":
            return context.has_action(ACTION_ALIASES.get(args[0], args[0]))
        if head == "action_count":
            return context.action_count(ACTION_ALIASES.get(args[0], args[0]))
        if head == "and":
            return all(self._coerce_bool(self._evaluate_symbolic_node(item, context)) for item in args)
        if head == "or":
            return any(self._coerce_bool(self._evaluate_symbolic_node(item, context)) for item in args)
        if head == "not":
            return not self._coerce_bool(self._evaluate_symbolic_node(args[0], context))
        if head in {"apply_soft_prompts", "retrieve", "walk"}:
            return []
        raise ValueError(f"unsupported symbolic runtime helper: {head}")

    def _normalize_action_sequence(self, raw_value: Any, context: _RuntimeContext) -> list[str]:
        flattened = self._flatten_value(raw_value)
        plan: list[str] = []
        for item in flattened:
            if item in (None, False, True):
                continue
            if not isinstance(item, str):
                raise ValueError(f"non-action value {item!r} produced by lambda runtime")
            plan.append(ACTION_ALIASES.get(item, item))
        if not plan and context.declared_action_types:
            return list(context.declared_action_types)
        return plan

    def _flatten_value(self, value: Any) -> list[Any]:
        if isinstance(value, list | tuple):
            items: list[Any] = []
            for item in value:
                items.extend(self._flatten_value(item))
            return items
        return [value]

    def _coerce_bool(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, list | tuple):
            return len(value) > 0
        if isinstance(value, str):
            lowered = value.lower()
            if lowered in {"true", "yes"}:
                return True
            if lowered in {"false", "no", ""}:
                return False
        return bool(value)

    def _action_reference_from_python(self, node: ast.AST) -> str:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return ACTION_ALIASES.get(node.value, node.value)
        if isinstance(node, ast.Name):
            return ACTION_ALIASES.get(node.id, node.id)
        raise ValueError("action references must be names or string literals")
