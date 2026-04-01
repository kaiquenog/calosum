from __future__ import annotations

import asyncio
import json
import os
import subprocess
from dataclasses import dataclass
from typing import Any

from calosum.shared.ports import LeftHemispherePort
from calosum.shared.types import (
    ActionExecutionResult,
    CognitiveBridgePacket,
    CognitiveWorkspace,
    LeftHemisphereResult,
    MemoryContext,
    PrimitiveAction,
    TypedLambdaProgram,
    UserTurn,
)

@dataclass(slots=True)
class RlmAdapterConfig:
    runtime_command: str | None = None
    model_path: str | None = None
    max_depth: int = 3
    timeout_seconds: float = 35.0
    endpoint: str | None = None
    model: str | None = None

class RlmLeftHemisphereAdapter(LeftHemispherePort):
    """Recursive Language Model adapter seguindo o paradigma RLM."""

    def __init__(self, config: RlmAdapterConfig | None = None) -> None:
        self.config = config or RlmAdapterConfig()
        self.MAX_DEPTH = self.config.max_depth
        self.CHUNK_SIZE = int(os.getenv("CALOSUM_RLM_CHUNK_SIZE", "2000"))
        self._depth = 0

    def reason(
        self,
        user_turn: UserTurn,
        bridge_packet: CognitiveBridgePacket,
        memory_context: MemoryContext,
        runtime_feedback: list[str] | None = None,
        attempt: int = 0,
        workspace: CognitiveWorkspace | None = None,
    ) -> LeftHemisphereResult:
        self._depth = 0
        result = self._recursive_reason(user_turn.user_text, bridge_packet, memory_context)
        
        if workspace is not None:
            workspace.left_notes.update(
                {
                    "backend": result.telemetry.get("backend", "rlm"),
                    "reasoning_summary": result.reasoning_summary,
                    "actions": [a.action_type for a in result.actions],
                }
            )
        return result

    def _recursive_reason(
        self,
        text: str,
        bridge_packet: CognitiveBridgePacket,
        memory_context: MemoryContext,
        depth: int = 0,
    ) -> LeftHemisphereResult:
        if depth >= self.MAX_DEPTH:
            return self._base_reason(text, bridge_packet, memory_context)

        if len(text) <= self.CHUNK_SIZE:
            return self._base_reason(text, bridge_packet, memory_context)

        chunks = self._decompose(text)

        partial_results = []
        for chunk in chunks:
            result = self._recursive_reason(chunk, bridge_packet, memory_context, depth + 1)
            partial_results.append(result)

        return self._compose_results(partial_results, text)

    def _decompose(self, text: str) -> list[str]:
        paragraphs = text.split("\n\n")

        chunks: list[str] = []
        current = ""
        for para in paragraphs:
            if len(current) + len(para) > self.CHUNK_SIZE:
                if current:
                    chunks.append(current.strip())
                current = para
            else:
                current += "\n\n" + para if current else para

        if current.strip():
            chunks.append(current.strip())

        return chunks if chunks else [text]

    def _base_reason(
        self,
        text: str,
        bridge_packet: CognitiveBridgePacket,
        memory_context: MemoryContext,
    ) -> LeftHemisphereResult:
        if self.config.runtime_command:
            return self._call_rlm_binary(text, bridge_packet)
        return self._fallback_reason(text, bridge_packet)

    def _call_rlm_binary(self, text: str, bridge_packet: CognitiveBridgePacket) -> LeftHemisphereResult:
        assert self.config.runtime_command is not None
        cmd = self.config.runtime_command.split() + [
            "--model", self.config.model_path or "rlm-qwen3-8b",
            "--prompt", text,
            "--json",
        ]
        
        payload = json.dumps({"text": text, "latent": bridge_packet.latent_vector})
        result = subprocess.run(cmd, input=payload, capture_output=True, text=True, timeout=self.config.timeout_seconds)
        data = json.loads(result.stdout)
        
        actions = []
        for raw in data.get("actions", []):
            actions.append(PrimitiveAction(
                action_type=raw.get("action_type", "respond_text"),
                typed_signature=raw.get("typed_signature", "Context -> Text"),
                payload=raw.get("payload", {}),
                safety_invariants=raw.get("safety_invariants", []),
            ))
            
        return LeftHemisphereResult(
            response_text=data.get("response_text", data.get("response", "")),
            lambda_program=TypedLambdaProgram(
                signature="Context -> Response",
                expression=data.get("lambda_expression", f"lambda ctx: respond()"),
                expected_effect="output safely",
            ),
            actions=actions or [PrimitiveAction(
                action_type="respond_text",
                typed_signature="Context -> Text",
                payload={"text": data.get("response_text", "")},
                safety_invariants=["no_injection"],
            )],
            reasoning_summary=data.get("reasoning_summary", data.get("reasoning_steps", [])),
            telemetry={"adapter": "RlmLeftHemisphereAdapter", "backend": "rlm_runtime"},
        )

    def _fallback_reason(self, text: str, bridge_packet: CognitiveBridgePacket) -> LeftHemisphereResult:
        response_text = "Vou resolver de forma recursiva e segura. " + text[:100]
        return LeftHemisphereResult(
            response_text=response_text,
            lambda_program=TypedLambdaProgram(
                signature="Context -> Memory -> Decision",
                expression="(lambda context memory (sequence (emit typed_actions)))",
                expected_effect="Compose recursive safe answer",
            ),
            actions=[
                PrimitiveAction(
                    action_type="respond_text",
                    typed_signature="ResponsePlan -> SafeTextMessage",
                    payload={"text": response_text},
                    safety_invariants=["safe output only", "typed runtime constraints"],
                )
            ],
            reasoning_summary=["recursive_fallback"],
            telemetry={"adapter": "RlmLeftHemisphereAdapter", "backend": "rlm_local_recursive"},
        )

    def _compose_results(self, results: list[LeftHemisphereResult], original_text: str) -> LeftHemisphereResult:
        combined_text = "\n\n".join(r.response_text for r in results)
        combined_actions = []
        for r in results:
            combined_actions.extend(r.actions)
        combined_summary = []
        for r in results:
            combined_summary.extend(r.reasoning_summary)

        return LeftHemisphereResult(
            response_text=combined_text,
            lambda_program=TypedLambdaProgram(
                signature="Context -> ComposedResponse",
                expression=f"lambda ctx: compose({len(results)} partials)",
                expected_effect="compose output",
            ),
            actions=combined_actions,
            reasoning_summary=combined_summary,
            telemetry={"adapter": "RlmLeftHemisphereAdapter", "backend": "composed_rlm"},
        )

    async def areason(self, *args: Any, **kwargs: Any) -> LeftHemisphereResult:
        return await asyncio.to_thread(self.reason, *args, **kwargs)

    def repair(
        self,
        user_turn: UserTurn,
        bridge_packet: CognitiveBridgePacket,
        memory_context: MemoryContext,
        previous_result: LeftHemisphereResult,
        rejected_results: list[ActionExecutionResult],
        attempt: int,
        critique_feedback: list[str] | None = None,
        workspace: CognitiveWorkspace | None = None,
    ) -> LeftHemisphereResult:
        return self.reason(
            user_turn=user_turn,
            bridge_packet=bridge_packet,
            memory_context=memory_context,
            runtime_feedback=critique_feedback,
            attempt=attempt,
            workspace=workspace,
        )

    async def arepair(
        self,
        user_turn: UserTurn,
        bridge_packet: CognitiveBridgePacket,
        memory_context: MemoryContext,
        previous_result: LeftHemisphereResult,
        rejected_results: list[ActionExecutionResult],
        attempt: int,
        critique_feedback: list[str] | None = None,
        workspace: CognitiveWorkspace | None = None,
    ) -> LeftHemisphereResult:
        return await asyncio.to_thread(
            self.repair,
            user_turn,
            bridge_packet,
            memory_context,
            previous_result,
            rejected_results,
            attempt,
            critique_feedback,
            workspace,
        )
