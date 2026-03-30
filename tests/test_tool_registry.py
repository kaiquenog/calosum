from __future__ import annotations

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from calosum.adapters.action_runtime import ConcreteActionRuntime
from calosum.shared.tools import ToolRegistry, ToolSchema
from calosum.shared.types import ActionExecutionResult, LeftHemisphereResult, PrimitiveAction, TypedLambdaProgram


class ToolRegistryTests(unittest.IsolatedAsyncioTestCase):
    async def test_tool_registry_execution(self):
        registry = ToolRegistry()
        
        async def dummy_tool(payload):
            return f"Hello {payload.get('name')}"
            
        registry.register(
            ToolSchema("dummy", "Dummy tool", {"name": "str"}),
            dummy_tool
        )
        
        runtime = ConcreteActionRuntime(registry=registry)
        left_result = LeftHemisphereResult(
            response_text="Doing dummy",
            lambda_program=TypedLambdaProgram("A->B", "lambda x: x", "effect"),
            actions=[
                PrimitiveAction("dummy", "A->B", {"name": "Alice"}, [])
            ],
            reasoning_summary=[],
        )
        
        results = await runtime.arun(left_result)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].status, "executed")
        self.assertIn("Hello Alice", results[0].output.get("result", ""))

    async def test_tool_not_found(self):
        runtime = ConcreteActionRuntime()
        left_result = LeftHemisphereResult(
            response_text="Doing unknown",
            lambda_program=TypedLambdaProgram("A->B", "lambda x: x", "effect"),
            actions=[
                PrimitiveAction("unknown_tool", "A->B", {}, [])
            ],
            reasoning_summary=[],
        )
        
        results = await runtime.arun(left_result)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].status, "rejected")
        self.assertIn("not found in registry", results[0].output.get("error", ""))

    async def test_needs_approval(self):
        registry = ToolRegistry()
        
        async def dummy_tool(payload):
            return "Executed"
            
        registry.register(
            ToolSchema("dangerous_tool", "Dangerous tool", {}, needs_approval=True),
            dummy_tool
        )
        
        runtime = ConcreteActionRuntime(registry=registry)
        left_result = LeftHemisphereResult(
            response_text="Doing dangerous",
            lambda_program=TypedLambdaProgram("A->B", "lambda x: x", "effect"),
            actions=[
                PrimitiveAction("dangerous_tool", "A->B", {}, [])
            ],
            reasoning_summary=[],
        )
        
        results = await runtime.arun(left_result)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].status, "needs_approval")

        # Now with approval
        left_result.actions[0].payload["approved"] = True
        results_approved = await runtime.arun(left_result)
        self.assertEqual(len(results_approved), 1)
        self.assertEqual(results_approved[0].status, "executed")

    async def test_schema_validation_rejects_invalid_payload(self):
        runtime = ConcreteActionRuntime()
        left_result = LeftHemisphereResult(
            response_text="Invalid payload",
            lambda_program=TypedLambdaProgram("A->B", "lambda x: x", "effect"),
            actions=[
                PrimitiveAction("respond_text", "A->B", {}, [])
            ],
            reasoning_summary=[],
        )

        results = await runtime.arun(left_result)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].status, "rejected")
        self.assertEqual(results[0].output.get("error_type"), "validation_failed")

    async def test_runtime_contract_audit_reports_tool_contracts(self):
        runtime = ConcreteActionRuntime()

        report = runtime.audit_runtime_contracts({"validation_failed": 1})

        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["validation_failed_recent_count"], 1)
        tool_names = {item["tool"] for item in report["tool_contracts"]}
        self.assertIn("respond_text", tool_names)
        self.assertIn("search_web", tool_names)

    async def test_code_execution_runs_constrained_python(self):
        runtime = ConcreteActionRuntime()
        left_result = LeftHemisphereResult(
            response_text="Execute code",
            lambda_program=TypedLambdaProgram("Code->Text", "lambda code: code_execution()", "effect"),
            actions=[
                PrimitiveAction(
                    "code_execution",
                    "Code->Text",
                    {"code": "print(sum(i * i for i in range(4)))", "approved": True},
                    [],
                )
            ],
            reasoning_summary=[],
        )

        results = await runtime.arun(left_result)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].status, "executed")
        self.assertIn("14", results[0].output.get("result", ""))

    async def test_code_execution_blocks_unsafe_imports(self):
        runtime = ConcreteActionRuntime()
        left_result = LeftHemisphereResult(
            response_text="Execute code",
            lambda_program=TypedLambdaProgram("Code->Text", "lambda code: code_execution()", "effect"),
            actions=[
                PrimitiveAction(
                    "code_execution",
                    "Code->Text",
                    {"code": "import os\nprint('nope')", "approved": True},
                    [],
                )
            ],
            reasoning_summary=[],
        )

        results = await runtime.arun(left_result)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].status, "executed")
        self.assertIn("rejected", results[0].output.get("result", "").lower())
        self.assertIn("imports are not allowed", results[0].output.get("result", ""))

    async def test_http_request_returns_structured_payload(self):
        runtime = ConcreteActionRuntime()
        fake_response = SimpleNamespace(
            status_code=200,
            reason_phrase="OK",
            url="https://example.com/api",
            headers={"content-type": "application/json"},
            text='{"status":"ok"}',
            json=lambda: {"status": "ok"},
        )
        left_result = LeftHemisphereResult(
            response_text="HTTP request",
            lambda_program=TypedLambdaProgram("Request->Text", "lambda req: http_request()", "effect"),
            actions=[
                PrimitiveAction(
                    "http_request",
                    "Request->Text",
                    {"method": "GET", "url": "https://example.com/api"},
                    [],
                )
            ],
            reasoning_summary=[],
        )

        with patch(
            "calosum.adapters.tools.http_request.httpx.AsyncClient.request",
            new=AsyncMock(return_value=fake_response),
        ):
            results = await runtime.arun(left_result)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].status, "executed")
        payload = json.loads(results[0].output.get("result", ""))
        self.assertEqual(payload["status_code"], 200)
        self.assertEqual(payload["body"]["status"], "ok")
