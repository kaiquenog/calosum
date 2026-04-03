from __future__ import annotations

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from calosum.adapters.execution.tool_runtime import ConcreteActionRuntime
from calosum.shared.utils.tools import ToolRegistry, ToolSchema
from calosum.shared.models.types import ActionExecutionResult, ActionPlannerResult, PrimitiveAction, TypedLambdaProgram


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
        left_result = ActionPlannerResult(
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
        left_result = ActionPlannerResult(
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
        left_result = ActionPlannerResult(
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
        left_result = ActionPlannerResult(
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
        left_result = ActionPlannerResult(
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

        with patch(
            "calosum.adapters.execution.docker_sandbox.DockerToolSandbox.execute_command",
            new=AsyncMock(return_value={"stdout": "14", "stderr": "", "exit_code": 0, "status": "success"}),
        ):
            results = await runtime.arun(left_result)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].status, "executed")
        self.assertIn("14", results[0].output.get("result", ""))

    async def test_code_execution_blocks_unsafe_imports(self):
        runtime = ConcreteActionRuntime()
        left_result = ActionPlannerResult(
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

        with patch(
            "calosum.adapters.execution.docker_sandbox.DockerToolSandbox.execute_command",
            new=AsyncMock(return_value={"stdout": "Imports are not allowed", "stderr": "", "exit_code": 0, "status": "success"}),
        ):
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
        left_result = ActionPlannerResult(
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

    async def test_query_session_stats_reports_aggregates(self):
        fake_agent = SimpleNamespace(
            cognitive_dashboard=lambda _session_id: {
                "decision": [
                    {"tool_success_rate": 0.5, "runtime_retry_count": 2},
                    {"tool_success_rate": 1.0, "runtime_retry_count": 0},
                ],
                "execution": [
                    {"results": [{"status": "rejected", "output": {"error_type": "validation_failed"}}]}
                ],
                "felt": [{"surprise_score": 0.2}, {"surprise_score": 0.6}],
            }
        )
        runtime = ConcreteActionRuntime(agent_accessor=lambda: (fake_agent, None))
        left_result = ActionPlannerResult(
            response_text="Stats",
            lambda_program=TypedLambdaProgram("A->B", "lambda x: x", "effect"),
            actions=[PrimitiveAction("query_session_stats", "A->B", {"session_id": "s1"}, [])],
            reasoning_summary=[],
        )

        results = await runtime.arun(left_result)
        self.assertEqual(results[0].status, "executed")
        payload = json.loads(results[0].output["result"])
        self.assertEqual(payload["dominant_failure"], "validation_failed (1x)")
        self.assertAlmostEqual(payload["tool_success_rate"], 0.75)

    async def test_read_architecture_returns_source_and_dependencies(self):
        runtime = ConcreteActionRuntime()
        left_result = ActionPlannerResult(
            response_text="Read architecture",
            lambda_program=TypedLambdaProgram("A->B", "lambda x: x", "effect"),
            actions=[PrimitiveAction("read_architecture", "A->B", {"component_name": "CalosumAgent"}, [])],
            reasoning_summary=[],
        )

        results = await runtime.arun(left_result)
        self.assertEqual(results[0].status, "executed")
        payload = json.loads(results[0].output["result"])
        self.assertIn("source_code", payload)
        self.assertIn("dependencies", payload)
        self.assertTrue(str(payload["path"]).endswith(".py"))

    async def test_propose_config_change_queues_pending_directive(self):
        class EvolutionManagerStub:
            def __init__(self) -> None:
                self.pending_directives = []

            def queue_directive(self, directive) -> None:
                self.pending_directives.append(directive)

        evo = EvolutionManagerStub()
        fake_agent = SimpleNamespace(evolution_manager=evo)
        runtime = ConcreteActionRuntime(agent_accessor=lambda: (fake_agent, None))
        left_result = ActionPlannerResult(
            response_text="Propose",
            lambda_program=TypedLambdaProgram("A->B", "lambda x: x", "effect"),
            actions=[
                PrimitiveAction(
                    "propose_config_change",
                    "A->B",
                    {
                        "parameter": "orchestrator.max_runtime_retries",
                        "reason": "Reducao de falhas transientes",
                        "new_value": "3",
                    },
                    [],
                )
            ],
            reasoning_summary=[],
        )

        results = await runtime.arun(left_result)
        self.assertEqual(results[0].status, "executed")
        self.assertEqual(len(evo.pending_directives), 1)
        self.assertEqual(evo.pending_directives[0].status, "pending")
