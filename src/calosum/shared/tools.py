from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine


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
        self._handlers: dict[str, Callable[[dict[str, Any]], Coroutine[Any, Any, str]]] = {}

    def register(
        self,
        schema: ToolSchema,
        handler: Callable[[dict[str, Any]], Coroutine[Any, Any, str]],
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

    async def execute(self, name: str, payload: dict[str, Any]) -> str:
        handler = self._handlers.get(name)
        if not handler:
            raise ValueError(f"Tool '{name}' not found in registry")
        return await handler(payload)

    def list_schemas(self) -> list[ToolSchema]:
        return list(self._schemas.values())

    def get_descriptors(self) -> list["ToolDescriptor"]:
        from calosum.shared.types import ToolDescriptor, ComponentHealth
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

    def _matches_type(self, value: Any, expected_type: Any) -> bool:
        normalized = str(expected_type).strip().lower()
        expected_map = {
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
        python_type = expected_map.get(normalized)
        if python_type is None:
            return True
        return isinstance(value, python_type)
