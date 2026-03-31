from __future__ import annotations

import asyncio
import json
import subprocess
from dataclasses import dataclass
from typing import Any

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


class RlmLeftHemisphereAdapter:
    """Recursive Language Model adapter with local subprocess + native fallback."""

    def __init__(self, config: RlmAdapterConfig | None = None) -> None:
        self.config = config or RlmAdapterConfig()

    def reason(
        self,
        user_turn: UserTurn,
        bridge_packet: CognitiveBridgePacket,
        memory_context: MemoryContext,
        runtime_feedback: list[str] | None = None,
        attempt: int = 0,
        workspace: CognitiveWorkspace | None = None,
    ) -> LeftHemisphereResult:
        payload = {
            "query": user_turn.user_text,
            "salience": bridge_packet.salience,
            "soft_prompts": [token.token for token in bridge_packet.soft_prompts],
            "directives": bridge_packet.control.system_directives,
            "runtime_feedback": runtime_feedback or [],
            "max_depth": self.config.max_depth,
            "model_path": self.config.model_path,
        }

        if self.config.runtime_command:
            parsed = self._invoke_runtime(payload)
            result = self._parsed_to_result(parsed, bridge_packet)
        else:
            result = self._local_recursive_reason(payload, bridge_packet)

        if workspace is not None:
            workspace.left_notes.update(
                {
                    "backend": result.telemetry.get("backend", "rlm"),
                    "reasoning_summary": result.reasoning_summary,
                    "actions": [a.action_type for a in result.actions],
                }
            )
        return result

    async def areason(
        self,
        user_turn: UserTurn,
        bridge_packet: CognitiveBridgePacket,
        memory_context: MemoryContext,
        runtime_feedback: list[str] | None = None,
        attempt: int = 0,
        workspace: CognitiveWorkspace | None = None,
    ) -> LeftHemisphereResult:
        return await asyncio.to_thread(
            self.reason,
            user_turn,
            bridge_packet,
            memory_context,
            runtime_feedback,
            attempt,
            workspace,
        )

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
        feedback = list(critique_feedback or [])
        if not feedback:
            feedback = [f"{item.action_type}:{'; '.join(item.violations)}" for item in rejected_results]
        return self.reason(
            user_turn,
            bridge_packet,
            memory_context,
            runtime_feedback=feedback,
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

    def _invoke_runtime(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert self.config.runtime_command is not None
        completed = subprocess.run(
            self.config.runtime_command.split(),
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            timeout=self.config.timeout_seconds,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"RLM runtime failed: rc={completed.returncode} stderr={completed.stderr.strip()}"
            )
        try:
            parsed = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("RLM runtime returned invalid JSON") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("RLM runtime payload must be an object")
        return parsed

    def _parsed_to_result(
        self,
        parsed: dict[str, Any],
        bridge_packet: CognitiveBridgePacket,
    ) -> LeftHemisphereResult:
        response_text = str(parsed.get("response_text") or parsed.get("answer") or "")
        reasoning = [str(item) for item in parsed.get("reasoning_summary", [])]
        if not reasoning:
            reasoning = ["rlm_runtime_response"]

        lambda_expression = str(
            parsed.get("lambda_expression")
            or "(lambda context memory (sequence (emit respond_text)))"
        )

        actions_raw = parsed.get("actions", [])
        actions: list[PrimitiveAction] = []
        for item in actions_raw:
            if not isinstance(item, dict):
                continue
            actions.append(
                PrimitiveAction(
                    action_type=str(item.get("action_type", "respond_text")),
                    typed_signature=str(item.get("typed_signature", "ResponsePlan -> SafeTextMessage")),
                    payload=dict(item.get("payload", {"text": response_text})),
                    safety_invariants=list(item.get("safety_invariants", ["safe output only"])),
                )
            )

        if not actions:
            actions = [
                PrimitiveAction(
                    action_type="respond_text",
                    typed_signature="ResponsePlan -> SafeTextMessage",
                    payload={
                        "text": response_text,
                        "temperature": bridge_packet.control.target_temperature,
                    },
                    safety_invariants=["safe output only"],
                )
            ]

        return LeftHemisphereResult(
            response_text=response_text,
            lambda_program=TypedLambdaProgram(
                signature="Context -> Response",
                expression=lambda_expression,
                expected_effect="Deliver safe response",
            ),
            actions=actions,
            reasoning_summary=reasoning,
            telemetry={
                "adapter": "RlmLeftHemisphereAdapter",
                "backend": "rlm_runtime",
                "max_depth": self.config.max_depth,
                "system_directives": bridge_packet.control.system_directives,
            },
        )

    def _local_recursive_reason(
        self,
        payload: dict[str, Any],
        bridge_packet: CognitiveBridgePacket,
    ) -> LeftHemisphereResult:
        query = str(payload.get("query", "")).strip()
        feedback = [str(x) for x in payload.get("runtime_feedback", [])]
        decomposition = self._decompose(query, depth=self.config.max_depth)

        plan_keywords = ("plano", "passo", "roteiro", "organizar", "reorganizar")
        wants_plan = any(k in query.lower() for k in plan_keywords)

        opening = "Vou resolver de forma recursiva e segura."
        if bridge_packet.control.empathy_priority:
            opening = "Entendi o contexto e vou estruturar uma resposta segura e objetiva."

        if feedback:
            opening += " Ajustei a resposta com base no feedback de runtime."

        if wants_plan:
            steps = [f"{i + 1}. {item}" for i, item in enumerate(decomposition[:3])]
            response_text = opening + " Plano: " + " ".join(steps)
        else:
            response_text = opening + " " + " ".join(decomposition[:2])

        actions = [
            PrimitiveAction(
                action_type="respond_text",
                typed_signature="ResponsePlan -> SafeTextMessage",
                payload={
                    "text": response_text,
                    "temperature": bridge_packet.control.target_temperature,
                },
                safety_invariants=["safe output only", "typed runtime constraints"],
            )
        ]
        if wants_plan:
            actions.append(
                PrimitiveAction(
                    action_type="propose_plan",
                    typed_signature="DecisionContext -> TypedPlan",
                    payload={"steps": decomposition[:3], "style": "short"},
                    safety_invariants=["advisory only"],
                )
            )

        reasoning_summary = [
            f"recursive_depth={self.config.max_depth}",
            f"subproblems={len(decomposition)}",
            f"feedback_items={len(feedback)}",
        ]

        return LeftHemisphereResult(
            response_text=response_text,
            lambda_program=TypedLambdaProgram(
                signature="Context -> Memory -> Decision",
                expression="(lambda context memory (sequence (emit typed_actions)))",
                expected_effect="Compose recursive safe answer",
            ),
            actions=actions,
            reasoning_summary=reasoning_summary,
            telemetry={
                "adapter": "RlmLeftHemisphereAdapter",
                "backend": "rlm_local_recursive",
                "max_depth": self.config.max_depth,
                "system_directives": bridge_packet.control.system_directives,
            },
        )

    def _decompose(self, text: str, depth: int) -> list[str]:
        cleaned = text.strip()
        if depth <= 0 or len(cleaned) < 40:
            return [cleaned or "Responder com clareza e seguranca."]

        separators = [". ", "; ", " e ", " mas ", " porque "]
        for sep in separators:
            if sep in cleaned:
                parts = [p.strip() for p in cleaned.split(sep) if p.strip()]
                if len(parts) >= 2:
                    out: list[str] = []
                    for part in parts[:3]:
                        out.extend(self._decompose(part, depth - 1))
                    return out

        mid = len(cleaned) // 2
        return self._decompose(cleaned[:mid], depth - 1) + self._decompose(cleaned[mid:], depth - 1)
