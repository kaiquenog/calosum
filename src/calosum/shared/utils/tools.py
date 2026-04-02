from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine


_EXPECTED_TYPE_MAP: dict[str, Any] = {
    "str": str,
    "string": str,
    "list": list,
    "dict": dict,
    "object": dict,
    "bool": bool,
    "int": int,
    "integer": int,
    "float": float,
    "number": (int, float),
}


@dataclass(slots=True)
class ToolSchema:
    name: str
    description: str
    parameters: dict[str, Any]
    required_permissions: list[str] = field(default_factory=list)
    needs_approval: bool = False


class ToolRegistry:
    def __init__(self):
        self._schemas: dict[str, ToolSchema] = {}
        self._handlers: dict[str, Callable[..., Coroutine[Any, Any, str]]] = {}

    def register(
        self,
        schema: ToolSchema,
        handler: Callable[..., Coroutine[Any, Any, str]],
    ) -> None:
        self._schemas[schema.name] = schema
        self._handlers[schema.name] = handler

    def get_schema(self, name: str) -> ToolSchema | None:
        return self._schemas.get(name)

    def validate_payload(self, name: str, payload: dict[str, Any]) -> list[str]:
        schema = self.get_schema(name)
        if schema is None:
            return [f"Tool '{name}' not found in registry"]

        violations: list[str] = []
        for parameter_name, expected_type in schema.parameters.items():
            if parameter_name not in payload:
                violations.append(f"missing required parameter: {parameter_name}")
                continue

            if not self._matches_type(payload[parameter_name], expected_type):
                violations.append(
                    f"parameter '{parameter_name}' must be of type {expected_type}"
                )
        return violations

    async def execute(self, name: str, payload: dict[str, Any], **kwargs: Any) -> str:
        handler = self._handlers.get(name)
        if not handler:
            raise ValueError(f"Tool '{name}' not found in registry")
        signature = inspect.signature(handler)
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if accepts_kwargs:
            return await handler(payload, **kwargs)
        filtered = {
            key: value
            for key, value in kwargs.items()
            if key in signature.parameters
        }
        return await handler(payload, **filtered)

    def list_schemas(self) -> list[ToolSchema]:
        return list(self._schemas.values())

    def get_descriptors(self) -> list["ToolDescriptor"]:
        from calosum.shared.models.types import ToolDescriptor, ComponentHealth
        return [
            ToolDescriptor(
                name=schema.name,
                description=schema.description,
                requires_approval=schema.needs_approval,
                required_permissions=schema.required_permissions,
                health=ComponentHealth.HEALTHY,
            )
            for schema in self._schemas.values()
        ]

    def supports_expected_type(self, expected_type: Any) -> bool:
        normalized = str(expected_type).strip().lower()
        if not normalized:
            return False
        return normalized in _EXPECTED_TYPE_MAP

    def supported_parameter_types(self) -> list[str]:
        return sorted(_EXPECTED_TYPE_MAP.keys())

    def _matches_type(self, value: Any, expected_type: Any) -> bool:
        normalized = str(expected_type).strip().lower()
        python_type = _EXPECTED_TYPE_MAP.get(normalized)
        if python_type is None:
            return True
        return isinstance(value, python_type)


def build_runtime_contract_audit_report(
    registry: ToolRegistry,
    failure_types: dict[str, int] | None = None,
) -> dict[str, object]:
    failure_types = failure_types or {}
    validation_failures = int(failure_types.get("validation_failed", 0))
    supported_types = registry.supported_parameter_types()

    tool_contracts: list[dict[str, object]] = []
    unsupported_type_violations: list[str] = []

    for schema in registry.list_schemas():
        parameters: list[dict[str, str]] = []
        for name, expected in schema.parameters.items():
            expected_label = str(expected).strip().lower() or str(expected)
            parameters.append({"name": name, "type": expected_label})
            if not registry.supports_expected_type(expected):
                unsupported_type_violations.append(
                    f"{schema.name}.{name} uses unsupported type contract '{expected_label}'"
                )

        tool_contracts.append(
            {
                "tool": schema.name,
                "parameter_count": len(schema.parameters),
                "parameters": parameters,
                "required_permissions": list(schema.required_permissions),
                "requires_approval": schema.needs_approval,
            }
        )

    recommendations: list[str] = []
    if validation_failures > 0:
        recommendations.append(
            "Inject explicit tool contract block in left-hemisphere prompt and repair feedback."
        )
        recommendations.append(
            "Track per-tool validation_failed counts in telemetry to identify dominant mismatch patterns."
        )
    if unsupported_type_violations:
        recommendations.append(
            "Normalize unsupported schema parameter types to canonical runtime types."
        )
    if not recommendations:
        recommendations.append("Runtime contracts look consistent with current registry type system.")

    return {
        "status": "ok",
        "registered_tools": len(tool_contracts),
        "supported_parameter_types": supported_types,
        "validation_failed_recent_count": validation_failures,
        "unsupported_type_violations": unsupported_type_violations,
        "tool_contracts": tool_contracts,
        "recommendations": recommendations,
    }
