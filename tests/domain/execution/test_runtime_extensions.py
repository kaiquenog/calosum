from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from calosum.adapters.execution.tool_runtime import ConcreteActionRuntime
from calosum.adapters.tools.mcp_client import HttpMcpClientAdapter, McpServerEndpoint
from calosum.domain.infrastructure.interceptors import AuditLogInterceptor, InterceptorManager
from calosum.shared.models.types import (
    CognitiveWorkspace,
    ActionPlannerResult,
    PrimitiveAction,
    TypedLambdaProgram,
)


def _left_result(actions: list[PrimitiveAction]) -> ActionPlannerResult:
    return ActionPlannerResult(
        response_text="",
        lambda_program=TypedLambdaProgram("", "", ""),
        actions=actions,
        reasoning_summary=[],
    )


class _FakeHttpResponse:
    def __init__(self, body: str) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body.encode("utf-8")

    def __enter__(self) -> "_FakeHttpResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class RuntimeExtensionsTests(unittest.IsolatedAsyncioTestCase):
    async def test_call_mcp_tool_disabled_returns_message(self) -> None:
        runtime = ConcreteActionRuntime()
        result = await runtime.arun(
            _left_result(
                [
                    PrimitiveAction(
                        action_type="call_mcp_tool",
                        typed_signature="Call -> Result",
                        payload={
                            "server": "dummy",
                            "tool_name": "hello",
                            "arguments": {},
                            "approved": True,
                        },
                    )
                ]
            )
        )
        self.assertIn("not enabled", result[0].output["result"])

    async def test_call_mcp_tool_executes_when_client_available(self) -> None:
        client = HttpMcpClientAdapter(
            servers={"local": McpServerEndpoint(name="local", url="http://localhost:9999/mcp")}
        )
        runtime = ConcreteActionRuntime(mcp_client=client)
        fake_body = json.dumps({"jsonrpc": "2.0", "id": "1", "result": {"ok": True}})
        with patch(
            "calosum.adapters.tools.mcp_client.urllib_request.urlopen",
            return_value=_FakeHttpResponse(fake_body),
        ):
            result = await runtime.arun(
                _left_result(
                    [
                        PrimitiveAction(
                            action_type="call_mcp_tool",
                            typed_signature="Call -> Result",
                            payload={
                                "server": "local",
                                "tool_name": "ping",
                                "arguments": {"x": 1},
                                "approved": True,
                            },
                        )
                    ]
                )
            )
        self.assertIn('"ok": true', result[0].output["result"].lower())

    async def test_spawn_subordinate_returns_completed(self) -> None:
        runtime = ConcreteActionRuntime()
        result = await runtime.arun(
            _left_result(
                [
                    PrimitiveAction(
                        action_type="spawn_subordinate",
                        typed_signature="Task -> Verification",
                        payload={"task": "criar plano curto"},
                    )
                ]
            ),
            workspace=CognitiveWorkspace(task_frame={"session_id": "subordinate-test"}),
        )
        self.assertIn('"status": "completed"', result[0].output["result"])
        self.assertIn('"is_valid": true', result[0].output["result"].lower())

    async def test_runtime_interceptor_captures_tool_events(self) -> None:
        audit = AuditLogInterceptor()
        runtime = ConcreteActionRuntime(interceptor_manager=InterceptorManager([audit]))
        await runtime.arun(
            _left_result(
                [
                    PrimitiveAction(
                        action_type="respond_text",
                        typed_signature="Text -> Out",
                        payload={"text": "ok"},
                    )
                ]
            ),
            workspace=CognitiveWorkspace(task_frame={"session_id": "hook-test"}),
        )
        stages = [item["stage"] for item in audit.events]
        self.assertIn("before_tool_execution", stages)
        self.assertIn("after_tool_execution", stages)


if __name__ == "__main__":
    unittest.main()
