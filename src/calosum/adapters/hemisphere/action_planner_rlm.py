from __future__ import annotations

import asyncio
import json
import os
import subprocess
from dataclasses import dataclass
from typing import Any

from calosum.shared.models.ports import ActionPlannerPort
from calosum.shared.models.types import (
    ActionExecutionResult,
    PerceptionSummary,
    CognitiveWorkspace,
    ActionPlannerResult,
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

class RlmLeftHemisphereAdapter(ActionPlannerPort):
    """Recursive Language Model adapter seguindo o paradigma RLM."""

    def __init__(self, config: RlmAdapterConfig | None = None) -> None:
        self.config = config or RlmAdapterConfig()
        self.MAX_DEPTH = self.config.max_depth
        self.CHUNK_SIZE = int(os.getenv("CALOSUM_RLM_CHUNK_SIZE", "2000"))
        self._depth = 0

    def reason(
        self,
        user_turn: UserTurn,
        bridge_packet: PerceptionSummary,
        memory_context: MemoryContext,
        runtime_feedback: list[str] | None = None,
        attempt: int = 0,
        workspace: CognitiveWorkspace | None = None,
    ) -> ActionPlannerResult:
        self._depth = 0
        
        # Backtracking real: se houver falha, reduz o tamanho do chunk para focar em precisao
        effective_chunk_size = self.CHUNK_SIZE
        if attempt > 0:
            effective_chunk_size = max(500, self.CHUNK_SIZE // (attempt + 1))
            
        result = self._recursive_reason(
            user_turn.user_text, 
            bridge_packet, 
            memory_context, 
            chunk_size=effective_chunk_size,
            feedback=runtime_feedback
        )
        
        if workspace is not None:
            workspace.left_notes.update(
                {
                    "backend": result.telemetry.get("backend", "rlm"),
                    "attempt": attempt,
                    "effective_chunk_size": effective_chunk_size,
                    "reasoning_summary": result.reasoning_summary,
                }
            )
        return result

    def _recursive_reason(
        self,
        text: str,
        bridge_packet: PerceptionSummary,
        memory_context: MemoryContext,
        depth: int = 0,
        chunk_size: int | None = None,
        feedback: list[str] | None = None,
    ) -> ActionPlannerResult:
        actual_chunk_size = chunk_size or self.CHUNK_SIZE
        
        if depth >= self.MAX_DEPTH:
            return self._base_reason(text, bridge_packet, memory_context, feedback)

        if len(text) <= actual_chunk_size:
            return self._base_reason(text, bridge_packet, memory_context, feedback)

        chunks = self._decompose(text, actual_chunk_size)

        partial_results = []
        for chunk in chunks:
            result = self._recursive_reason(
                chunk, 
                bridge_packet, 
                memory_context, 
                depth + 1, 
                chunk_size=actual_chunk_size,
                feedback=feedback
            )
            partial_results.append(result)

        return self._compose_results(partial_results, text)

    def _decompose(self, text: str, chunk_size: int | None = None) -> list[str]:
        actual_chunk_size = chunk_size or self.CHUNK_SIZE
        paragraphs = text.split("\n\n")

        chunks: list[str] = []
        current = ""
        for para in paragraphs:
            if len(current) + len(para) > actual_chunk_size:
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
        bridge_packet: PerceptionSummary,
        memory_context: MemoryContext,
        feedback: list[str] | None = None,
    ) -> ActionPlannerResult:
        if self.config.runtime_command:
            return self._call_rlm_binary(text, bridge_packet, feedback)
        return self._fallback_reason(text, bridge_packet, feedback)

    def _call_rlm_binary(
        self, 
        text: str, 
        bridge_packet: PerceptionSummary,
        feedback: list[str] | None = None,
    ) -> ActionPlannerResult:
        assert self.config.runtime_command is not None
        cmd = self.config.runtime_command.split() + [
            "--model", self.config.model_path or "rlm-qwen3-8b",
            "--prompt", text,
            "--json",
        ]
        
        payload_dict = {
            "text": text, 
            "latent": bridge_packet.latent_vector,
            "feedback": feedback or []
        }
        payload = json.dumps(payload_dict)
        result = subprocess.run(cmd, input=payload, capture_output=True, text=True, timeout=self.config.timeout_seconds)
        # ... rest of method unchanged ...
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
            
        return ActionPlannerResult(
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

    def _fallback_reason(
        self, 
        text: str, 
        bridge_packet: PerceptionSummary,
        feedback: list[str] | None = None,
    ) -> ActionPlannerResult:
        response_text = "Vou resolver de forma recursiva e segura. " + text[:100]
        if feedback:
            response_text += f"\nRevisando devido ao feedback: {feedback[0][:50]}..."
        return ActionPlannerResult(
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

    def _compose_results(self, results: list[ActionPlannerResult], original_text: str) -> ActionPlannerResult:
        combined_text = "\n\n".join(r.response_text for r in results)
        combined_actions = []
        for r in results:
            combined_actions.extend(r.actions)
        combined_summary = []
        for r in results:
            combined_summary.extend(r.reasoning_summary)

        return ActionPlannerResult(
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

    async def areason(self, *args: Any, **kwargs: Any) -> ActionPlannerResult:
        return await asyncio.to_thread(self.reason, *args, **kwargs)

    def repair(
        self,
        user_turn: UserTurn,
        bridge_packet: PerceptionSummary,
        memory_context: MemoryContext,
        previous_result: ActionPlannerResult,
        rejected_results: list[ActionExecutionResult],
        attempt: int,
        critique_feedback: list[str] | None = None,
        workspace: CognitiveWorkspace | None = None,
    ) -> ActionPlannerResult:
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
        bridge_packet: PerceptionSummary,
        memory_context: MemoryContext,
        previous_result: ActionPlannerResult,
        rejected_results: list[ActionExecutionResult],
        attempt: int,
        critique_feedback: list[str] | None = None,
        workspace: CognitiveWorkspace | None = None,
    ) -> ActionPlannerResult:
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
