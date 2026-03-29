from __future__ import annotations

import asyncio
import unittest

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
