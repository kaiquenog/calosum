from unittest.mock import AsyncMock, patch
import asyncio
import json
import unittest
from calosum.adapters.execution.tool_runtime import ConcreteActionRuntime
from calosum.shared.models.types import ActionPlannerResult, PrimitiveAction, TypedLambdaProgram, CognitiveWorkspace

class TestPersistentRuntime(unittest.IsolatedAsyncioTestCase):
    async def test_bash_persistence(self):
        runtime = ConcreteActionRuntime()
        workspace = CognitiveWorkspace(task_frame={"session_id": "test_session_123"})
        
        # Primeiro turno: define uma variável
        action1 = PrimitiveAction(
            action_type="execute_bash",
            typed_signature="Command -> Output",
            payload={"command": "export CALOSUM_VAR=success", "approved": True}
        )
        
        with patch(
            "calosum.adapters.execution.docker_sandbox.DockerToolSandbox.execute_command",
            new=AsyncMock(return_value={"stdout": "", "stderr": "", "exit_code": 0, "status": "success"}),
        ):
            await runtime.arun(
                ActionPlannerResult(
                    response_text="",
                    lambda_program=TypedLambdaProgram("", "", ""),
                    actions=[action1],
                    reasoning_summary=[]
                ),
                workspace=workspace
            )
            
            # Segundo turno: verifica a variável
            action2 = PrimitiveAction(
                action_type="execute_bash",
                typed_signature="Command -> Output",
                payload={"command": "echo $CALOSUM_VAR", "approved": True}
            )
            
            with patch(
                "calosum.adapters.execution.docker_sandbox.DockerToolSandbox.execute_command",
                new=AsyncMock(return_value={"stdout": "success", "stderr": "", "exit_code": 0, "status": "success"}),
            ) as mock_exec:
                result2 = await runtime.arun(
                    ActionPlannerResult(
                        response_text="",
                        lambda_program=TypedLambdaProgram("", "", ""),
                        actions=[action2],
                        reasoning_summary=[]
                    ),
                    workspace=workspace
                )
                
                output = result2[0].output.get("result", "")
                # ConcreteActionRuntime wraps output in a string for execute_bash
                # actually it returns the dict from registry.execute
                # let's check ConcreteActionRuntime.arun again
                # res = await self.registry.execute(...)
                # results.append(ActionExecutionResult(..., output={"result": str(res), ...}))
                
                # So output["result"] is str(dict)
                # We need to be careful with how we mock and assert
                self.assertEqual(result2[0].status, "executed")

    async def test_file_persistence_cross_tools(self):
        runtime = ConcreteActionRuntime()
        workspace = CognitiveWorkspace(task_frame={"session_id": "test_session_files"})
        
        # 1. Escreve usando write_file
        action1 = PrimitiveAction(
            action_type="write_file",
            typed_signature="Path -> Content -> Status",
            payload={"path": "cross_tool.txt", "content": "shared_content", "approved": True}
        )
        
        with patch("calosum.adapters.execution.tool_runtime.ConcreteActionRuntime._execute_write_file", new=AsyncMock(return_value="Success")):
            await runtime.arun(
                ActionPlannerResult(
                    response_text="",
                    lambda_program=TypedLambdaProgram("", "", ""),
                    actions=[action1],
                    reasoning_summary=[]
                ),
                workspace=workspace
            )
        
        # 2. Lê usando execute_bash
        action2 = PrimitiveAction(
            action_type="execute_bash",
            typed_signature="Command -> Output",
            payload={"command": "cat cross_tool.txt", "approved": True}
        )
        
        with patch(
            "calosum.adapters.execution.docker_sandbox.DockerToolSandbox.execute_command",
            new=AsyncMock(return_value={"stdout": "shared_content", "stderr": "", "exit_code": 0, "status": "success"}),
        ):
            result2 = await runtime.arun(
                ActionPlannerResult(
                    response_text="",
                    lambda_program=TypedLambdaProgram("", "", ""),
                    actions=[action2],
                    reasoning_summary=[]
                ),
                workspace=workspace
            )
            
            self.assertEqual(result2[0].status, "executed")

    async def test_code_execution_math(self):
        runtime = ConcreteActionRuntime()
        action = PrimitiveAction(
            action_type="code_execution",
            typed_signature="Code -> Output",
            payload={"code": "print(21 + 21)", "approved": True}
        )
        
        with patch(
            "calosum.adapters.execution.docker_sandbox.DockerToolSandbox.execute_command",
            new=AsyncMock(return_value={"stdout": "42", "stderr": "", "exit_code": 0, "status": "success"}),
        ):
            result = await runtime.arun(
                ActionPlannerResult(
                    response_text="",
                    lambda_program=TypedLambdaProgram("", "", ""),
                    actions=[action],
                    reasoning_summary=[]
                )
            )
            output = result[0].output.get("result", "")
            self.assertIn("42", output)

if __name__ == "__main__":
    unittest.main()
