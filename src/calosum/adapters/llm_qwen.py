from __future__ import annotations

import json
from dataclasses import dataclass

import httpx

from calosum.shared.async_utils import run_sync
from calosum.shared.types import (
    ActionExecutionResult,
    CognitiveBridgePacket,
    LeftHemisphereResult,
    MemoryContext,
    PrimitiveAction,
    TypedLambdaProgram,
    UserTurn,
)


@dataclass(slots=True)
class QwenAdapterConfig:
    api_url: str = "http://localhost:8000/v1/chat/completions"
    api_key: str = "empty"
    model_name: str = "Qwen/Qwen-3.5-9B-Instruct"
    max_tokens: int = 4096


class QwenLeftHemisphereAdapter:
    """
    Adapter real que se comunica com um modelo Qwen3.5 (via vLLM/Ollama ou endpoint compatível).
    Ele empacota o contexto em um prompt e obriga o modelo a responder estruturadamente em JSON.
    """

    def __init__(self, config: QwenAdapterConfig | None = None) -> None:
        self.config = config or QwenAdapterConfig()
        self.client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {self.config.api_key}"},
            timeout=300.0,
        )

    def reason(
        self,
        user_turn: UserTurn,
        bridge_packet: CognitiveBridgePacket,
        memory_context: MemoryContext,
        runtime_feedback: list[str] | None = None,
        attempt: int = 0,
    ) -> LeftHemisphereResult:
        return run_sync(
            self.areason(
                user_turn, bridge_packet, memory_context, runtime_feedback, attempt
            )
        )

    async def areason(
        self,
        user_turn: UserTurn,
        bridge_packet: CognitiveBridgePacket,
        memory_context: MemoryContext,
        runtime_feedback: list[str] | None = None,
        attempt: int = 0,
    ) -> LeftHemisphereResult:
        prompt = self._build_prompt(user_turn, bridge_packet, memory_context, runtime_feedback)

        payload = {
            "model": self.config.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a logical neuro-symbolic agent. Output valid JSON only, corresponding to LeftHemisphereResult format.",
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.config.max_tokens,
            "response_format": {"type": "json_object"},
        }

        try:
            response = await self.client.post(self.config.api_url, json=payload)
            response.raise_for_status()
            data = response.json()
            msg = data["choices"][0]["message"]
            content = msg.get("content", "")
            
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()
            elif content.startswith("```"):
                content = content.replace("```", "").strip()
            
            if not content.strip():
                reasoning = msg.get("reasoning", "")
                if reasoning:
                    return self._fallback_result(f"Modelo retornou apenas reasoning vazio de content: {reasoning}")
                return self._fallback_result("Conteúdo de resposta vazio do modelo.")

            try:
                parsed = json.loads(content)
                return self._parse_to_result(parsed)
            except json.JSONDecodeError as e:
                return LeftHemisphereResult(
                    response_text=content,
                    lambda_program=TypedLambdaProgram("Any", "()", "None"),
                    actions=[],
                    reasoning_summary=[f"Raw text extraction (JSON parse failed: {e})"],
                    telemetry={"adapter": "QwenLeftHemisphereAdapter"}
                )
        except Exception as e:
            import traceback
            # Fallback seguro com repr(e)
            return self._fallback_result(repr(e))

    def repair(
        self,
        user_turn: UserTurn,
        bridge_packet: CognitiveBridgePacket,
        memory_context: MemoryContext,
        previous_result: LeftHemisphereResult,
        rejected_results: list[ActionExecutionResult],
        attempt: int,
    ) -> LeftHemisphereResult:
        return run_sync(
            self.arepair(
                user_turn, bridge_packet, memory_context, previous_result, rejected_results, attempt
            )
        )

    async def arepair(
        self,
        user_turn: UserTurn,
        bridge_packet: CognitiveBridgePacket,
        memory_context: MemoryContext,
        previous_result: LeftHemisphereResult,
        rejected_results: list[ActionExecutionResult],
        attempt: int,
    ) -> LeftHemisphereResult:
        feedback = [f"Ação {r.action_type} rejeitada: {', '.join(r.violations)}" for r in rejected_results]
        return await self.areason(user_turn, bridge_packet, memory_context, feedback, attempt)

    def _build_prompt(
        self,
        user_turn: UserTurn,
        bridge_packet: CognitiveBridgePacket,
        memory_context: MemoryContext,
        feedback: list[str] | None,
    ) -> str:
        rules = [r.statement for r in memory_context.semantic_rules]
        return f"""
Analyze the input and generate a JSON.

Input: {user_turn.user_text}

Soft Prompts (Bridge):
{[t.token for t in bridge_packet.soft_prompts]}

Semantic Rules: {rules}
Runtime Feedback: {feedback or []}

Available Action Types (Use exactly these for action_type, and matching payload):
- "respond_text": {{ "text": "your response here" }}
- "propose_plan": {{ "steps": ["step1", "step2"] }}
- "search_web": {{ "query": "search keywords" }}
- "write_file": {{ "path": "file/path.txt", "content": "file content" }}

Expected JSON Schema:
{{
  "response_text": "string",
  "lambda_program": {{ "signature": "string", "expression": "string", "expected_effect": "string" }},
  "actions": [
    {{ "action_type": "string", "typed_signature": "string", "payload": {{}}, "safety_invariants": ["string"] }}
  ],
  "reasoning_summary": ["string"]
}}
"""

    def _parse_to_result(self, parsed: dict) -> LeftHemisphereResult:
        lambda_prog = parsed.get("lambda_program", {})
        program = TypedLambdaProgram(
            signature=lambda_prog.get("signature", "Any -> Any"),
            expression=lambda_prog.get("expression", ""),
            expected_effect=lambda_prog.get("expected_effect", ""),
        )

        actions = []
        for act in parsed.get("actions", []):
            actions.append(PrimitiveAction(
                action_type=act.get("action_type", "unknown"),
                typed_signature=act.get("typed_signature", "Any -> Any"),
                payload=act.get("payload", {}),
                safety_invariants=act.get("safety_invariants", []),
            ))

        return LeftHemisphereResult(
            response_text=parsed.get("response_text", parsed.get("result", parsed.get("message", ""))),
            lambda_program=program,
            actions=actions,
            reasoning_summary=parsed.get("reasoning_summary", []),
            telemetry={"adapter": "QwenLeftHemisphereAdapter"}
        )

    def _fallback_result(self, error: str) -> LeftHemisphereResult:
        return LeftHemisphereResult(
            response_text="Desculpe, meu subsistema de raciocínio falhou temporariamente.",
            lambda_program=TypedLambdaProgram("Fallback", "()", "None"),
            actions=[],
            reasoning_summary=[f"Erro LLM: {error}"],
            telemetry={"error": error}
        )
