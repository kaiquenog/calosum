from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class PrimitiveActionSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    action_type: str = Field(min_length=1)
    typed_signature: str = Field(min_length=1)
    payload: dict[str, Any]
    safety_invariants: list[str] = Field(default_factory=list)


class TypedLambdaProgramSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    signature: str = Field(min_length=1)
    expression: str = Field(min_length=1)
    expected_effect: str = Field(min_length=1)


class LeftHemisphereResultSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    response_text: str
    lambda_program: TypedLambdaProgramSchema
    actions: list[PrimitiveActionSchema] = Field(default_factory=list)
    reasoning_summary: list[str] = Field(default_factory=list)


def collect_left_result_schema_issues(result: object) -> list[str]:
    try:
        LeftHemisphereResultSchema.model_validate(result, from_attributes=True)
        return []
    except ValidationError as exc:
        return _format_validation_errors(exc)


def _format_validation_errors(exc: ValidationError) -> list[str]:
    issues: list[str] = []
    for error in exc.errors():
        location = ".".join(str(item) for item in error.get("loc", ()))
        message = error.get("msg", "invalid value")
        if location:
            issues.append(f"{location}: {message}")
        else:
            issues.append(message)
    return issues
