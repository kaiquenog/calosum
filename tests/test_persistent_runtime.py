import asyncio
import unittest
from calosum.adapters.action_runtime import ConcreteActionRuntime
from calosum.shared.types import LeftHemisphereResult, PrimitiveAction, TypedLambdaProgram, CognitiveWorkspace

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
        await runtime.arun(
            LeftHemisphereResult(
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
        result2 = await runtime.arun(
            LeftHemisphereResult(
                response_text="",
                lambda_program=TypedLambdaProgram("", "", ""),
                actions=[action2],
                reasoning_summary=[]
            ),
            workspace=workspace
        )
        
        output = result2[0].output.get("result", "")
        self.assertIn("success", output)

    async def test_file_persistence_cross_tools(self):
        runtime = ConcreteActionRuntime()
        workspace = CognitiveWorkspace(task_frame={"session_id": "test_session_files"})
        
        # 1. Escreve usando write_file
        action1 = PrimitiveAction(
            action_type="write_file",
            typed_signature="Path -> Content -> Status",
            payload={"path": "cross_tool.txt", "content": "shared_content", "approved": True}
        )
        await runtime.arun(
            LeftHemisphereResult(
                response_text="",
                lambda_program=TypedLambdaProgram("", "", ""),
                actions=[action1],
                reasoning_summary=[]
            ),
            workspace=workspace
        )
        
        # 2. Lê usando execute_bash (mesmo sandbox)
        action2 = PrimitiveAction(
            action_type="execute_bash",
            typed_signature="Command -> Output",
            payload={"command": "cat cross_tool.txt", "approved": True}
        )
        result2 = await runtime.arun(
            LeftHemisphereResult(
                response_text="",
                lambda_program=TypedLambdaProgram("", "", ""),
                actions=[action2],
                reasoning_summary=[]
            ),
            workspace=workspace
        )
        
        output2 = result2[0].output.get("result", "")
        self.assertIn("shared_content", output2)

    async def test_code_execution_math(self):
        runtime = ConcreteActionRuntime()
        # Apenas para garantir que o code_execution ainda funciona
        action = PrimitiveAction(
            action_type="code_execution",
            typed_signature="Code -> Output",
            payload={"code": "print(21 + 21)", "approved": True}
        )
        result = await runtime.arun(
            LeftHemisphereResult(
                response_text="",
                lambda_program=TypedLambdaProgram("", "", ""),
                actions=[action],
                reasoning_summary=[]
            )
        )
        output = result[0].output.get("result", "")
        print(f"DEBUG: result[0]={result[0]}")
        self.assertEqual(output.strip(), "42")

if __name__ == "__main__":
    unittest.main()
