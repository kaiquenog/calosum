from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from calosum.shared.tools import ToolSchema


class HttpRequestPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"]
    url: str
    headers: dict[str, str] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    json_body: Any | None = Field(default=None, alias="json")
    body: str | None = None
    timeout_seconds: float = 8.0
    expected_status: list[int] = Field(default_factory=list)

    @field_validator("url")
    @classmethod
    def _validate_url(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("url must use http or https")
        if not parsed.netloc:
            raise ValueError("url must include a host")
        return value

    @field_validator("timeout_seconds")
    @classmethod
    def _validate_timeout(cls, value: float) -> float:
        return max(0.5, min(20.0, value))

    @model_validator(mode="after")
    def _validate_body_modes(self) -> "HttpRequestPayload":
        if self.json_body is not None and self.body is not None:
            raise ValueError("use either json or body, not both")
        return self


@dataclass(slots=True)
class HttpRequestTool:
    max_body_chars: int = 4000
    schema: ToolSchema = field(
        default_factory=lambda: ToolSchema(
            name="http_request",
            description="Make an outbound HTTP request with bounded timeout and structured response",
            parameters={"method": "string", "url": "string"},
            required_permissions=["network"],
        )
    )

    async def execute(self, payload: dict[str, object]) -> str:
        try:
            request = HttpRequestPayload.model_validate(payload)
        except ValidationError as exc:
            issues = ", ".join(
                f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
                for error in exc.errors()
            )
            return f"HTTP request rejected: {issues}"

        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=request.timeout_seconds,
        ) as client:
            response = await client.request(
                request.method,
                request.url,
                headers=request.headers or None,
                params=request.params or None,
                json=request.json_body,
                content=request.body,
            )

        body = _render_response_body(response, self.max_body_chars)
        payload_out = {
            "method": request.method,
            "url": str(response.url),
            "status_code": response.status_code,
            "reason_phrase": response.reason_phrase,
            "content_type": response.headers.get("content-type", ""),
            "body": body,
        }
        if request.expected_status and response.status_code not in request.expected_status:
            payload_out["warning"] = (
                f"unexpected status: expected one of {request.expected_status}, "
                f"got {response.status_code}"
            )
        return json.dumps(payload_out, ensure_ascii=False)


def _render_response_body(response: httpx.Response, max_chars: int) -> Any:
    content_type = response.headers.get("content-type", "").lower()
    if "application/json" in content_type:
        try:
            return response.json()
        except ValueError:
            pass

    text = response.text
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 16] + "\n...[truncated]"
