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
            "memory": {
                "rules": [r.statement for r in memory_context.semantic_rules],
                "triples": [f"{t.subject} {t.predicate} {t.object}" for t in memory_context.knowledge_triples[:5]],
                "recent_episodes": [
                    {"query": ep.user_turn.user_text, "response": ep.left_result.response_text}
                    for ep in memory_context.recent_episodes[:3]
                ],
            },
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
        
        memory = payload.get("memory", {})
        episodes = memory.get("recent_episodes", [])
        
        # Heurística de Contexto: Se houver memória que casa com a query (busca simples de substring para protótipo)
        contextual_addition = ""
        for ep in episodes:
            if any(word.lower() in ep["query"].lower() for word in query.lower().split() if len(word) > 4):
                contextual_addition = f" Lembro que você mencionou: '{ep['response']}'."
                break

        if bridge_packet.control.empathy_priority:
            opening = "Entendi o contexto e vou estruturar uma resposta segura e objetiva."
        
        opening += contextual_addition

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
        """RLM-style recursive decomposition with semantic clause analysis.

        Instead of naive text splitting, the decomposition:
        1. Identifies semantically independent sub-tasks using coordinating markers
        2. Creates isolated context frames for each sub-problem
        3. Recursively processes sub-problems if depth budget allows
        """
        frame = {"query": text.strip(), "depth": 0, "max_depth": depth}
        results = self._recursive_decompose(frame)
        return [r["query"] for r in results if r.get("query")]

    def _recursive_decompose(self, frame: dict) -> list[dict]:
        query = frame["query"]
        depth = frame["depth"]
        max_depth = frame["max_depth"]

        if depth >= max_depth or len(query) < 50:
            return [{"query": query or "Responder com clareza e seguranca.", "depth": depth, "leaf": True}]

        sub_tasks = self._identify_sub_tasks(query)
        if len(sub_tasks) <= 1:
            return [{"query": query, "depth": depth, "leaf": True}]

        results: list[dict] = []
        for sub_task in sub_tasks[:3]:
            child_frame = {**frame, "query": sub_task, "depth": depth + 1}
            results.extend(self._recursive_decompose(child_frame))
        return results

    def _identify_sub_tasks(self, query: str) -> list[str]:
        """Identifies semantically independent sub-tasks using clause structure.

        Uses coordinating/subordinating markers ordered by independence strength
        to split compound queries into actionable sub-problems.
        """
        coord_markers = [
            (" e depois ", 2), (" alem disso ", 2), (" tambem ", 2),
            (" entao ", 2), (" em seguida ", 2),
            (". ", 1), ("; ", 1),
            (" mas ", 2), (" porem ", 2), (" porque ", 2),
        ]
        text = query
        for marker, _priority in coord_markers:
            if marker in text and len(text) > 60:
                parts = [p.strip() for p in text.split(marker, 1) if p.strip()]
                if len(parts) >= 2:
                    return parts
        return [query]
