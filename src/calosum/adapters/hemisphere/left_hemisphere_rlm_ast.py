from __future__ import annotations

import asyncio
import json
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any, Literal

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


@dataclass
class ReasoningNode:
    id: str
    goal: str
    operation: Literal["decompose", "inspect", "synthesize", "verify"] = "inspect"
    status: Literal["pending", "success", "failed"] = "pending"
    children: list[ReasoningNode] = field(default_factory=list)
    result: ActionPlannerResult | None = None
    feedback: list[str] = field(default_factory=list)

    def is_complete(self) -> bool:
        if not self.children:
            return self.status == "success"
        return all(c.is_complete() for c in self.children) and self.status != "failed"


@dataclass(slots=True)
class RlmAstAdapterConfig:
    runtime_command: str | None = None
    model_path: str | None = None
    max_depth: int = 3
    timeout_seconds: float = 35.0


class RlmAstLeftHemisphereAdapter(ActionPlannerPort):
    """Recursive Language Model adapter usando AST."""

    def __init__(self, config: RlmAstAdapterConfig | None = None) -> None:
        self.config = config or RlmAstAdapterConfig()
        self.MAX_DEPTH = self.config.max_depth

    def reason(
        self,
        user_turn: UserTurn,
        bridge_packet: PerceptionSummary,
        memory_context: MemoryContext,
        runtime_feedback: list[str] | None = None,
        attempt: int = 0,
        workspace: CognitiveWorkspace | None = None,
    ) -> ActionPlannerResult:
        
        # Build initial AST root
        root = ReasoningNode(
            id="root",
            goal=f"Solve: {user_turn.user_text}",
            operation="decompose",
            feedback=runtime_feedback or []
        )
        
        self._evaluate_node(root, bridge_packet, memory_context, depth=0)
        
        result = self._compose_ast(root)

        if workspace is not None:
            workspace.left_notes.update(
                {
                    "backend": result.telemetry.get("backend", "rlm_ast"),
                    "attempt": attempt,
                    "ast_nodes": self._count_nodes(root),
                    "reasoning_summary": result.reasoning_summary,
                }
            )
        return result

    def _evaluate_node(self, node: ReasoningNode, bridge_packet: PerceptionSummary, memory_context: MemoryContext, depth: int) -> None:
        if node.status == "success":
            return
            
        # Try to resolve directly
        res = self._base_reason(node.goal, bridge_packet, memory_context, node.feedback)
        if depth < self.MAX_DEPTH and not node.children:
            node.children = self._build_children(node, bridge_packet, memory_context, depth)
                
        if node.children:
            for child in node.children:
                self._evaluate_node(child, bridge_packet, memory_context, depth + 1)
            
            if all(c.status == "success" for c in node.children):
                node.status = "success"
                node.result = self._compose_ast(node)
            else:
                node.status = "failed"
        else:
            node.status = "success"
            node.result = res

    def _count_nodes(self, node: ReasoningNode) -> int:
        return 1 + sum(self._count_nodes(c) for c in node.children)

    def _compose_ast(self, node: ReasoningNode) -> ActionPlannerResult:
        if node.result and not node.children:
            return node.result
            
        combined_text = ""
        combined_actions = []
        combined_summary = [f"Composed {len(node.children)} AST nodes"]
        
        for c in node.children:
            if c.result:
                combined_text += c.result.response_text + "\n"
                combined_actions.extend(c.result.actions)
                combined_summary.extend(c.result.reasoning_summary)
                
        return ActionPlannerResult(
            response_text=combined_text.strip() or "Empty AST Result",
            lambda_program=TypedLambdaProgram(
                signature="Context -> ComposedResponse",
                expression=json.dumps(
                    {
                        "plan": ["respond_text"],
                        "ast_operation": node.operation,
                        "child_count": len(node.children),
                    },
                    ensure_ascii=False,
                ),
                expected_effect="compose output",
            ),
            actions=combined_actions,
            reasoning_summary=combined_summary,
            telemetry={"adapter": "RlmAstLeftHemisphereAdapter", "backend": "rlm_local_recursive"},
        )

    def _build_children(
        self,
        node: ReasoningNode,
        bridge_packet: PerceptionSummary,
        memory_context: MemoryContext,
        depth: int,
    ) -> list[ReasoningNode]:
        if self.config.runtime_command:
            return []
        goal = node.goal.strip()
        lowered = goal.lower()
        if depth >= self.MAX_DEPTH:
            return []
        if node.operation == "decompose":
            subtasks = self._decompose_goal(goal)
            if len(subtasks) <= 1:
                return [
                    ReasoningNode(id=f"{node.id}_inspect", goal=goal, operation="inspect", feedback=node.feedback),
                    ReasoningNode(id=f"{node.id}_verify", goal=goal, operation="verify", feedback=node.feedback),
                ]
            return [
                ReasoningNode(id=f"{node.id}_{index}", goal=subtask, operation="inspect", feedback=node.feedback)
                for index, subtask in enumerate(subtasks)
            ] + [
                ReasoningNode(id=f"{node.id}_synthesize", goal=goal, operation="synthesize", feedback=node.feedback),
                ReasoningNode(id=f"{node.id}_verify", goal=goal, operation="verify", feedback=node.feedback),
            ]
        if node.operation == "inspect" and self._needs_extra_verification(lowered, bridge_packet, node.feedback):
            return [
                ReasoningNode(id=f"{node.id}_verify", goal=goal, operation="verify", feedback=node.feedback),
            ]
        return []

    def _decompose_goal(self, goal: str) -> list[str]:
        text = goal.replace("Solve:", "").strip()
        chunks = [segment.strip(" .") for segment in text.split(",") if segment.strip()]
        if len(chunks) > 1:
            return chunks[:3]
        if any(marker in text.lower() for marker in ("plano", "reorganizar", "arquitetura", "benchmark", "tradeoff")):
            return [
                "inspect current constraints and risks",
                "synthesize an ordered response plan",
                "verify the plan against safety and clarity requirements",
            ]
        return [text]

    def _needs_extra_verification(
        self,
        lowered_goal: str,
        bridge_packet: PerceptionSummary,
        feedback: list[str],
    ) -> bool:
        if feedback:
            return True
        if bridge_packet.control.empathy_priority:
            return True
        return any(marker in lowered_goal for marker in ("risco", "verify", "seguro", "arquitetura"))

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
            "--model", self.config.model_path or "rlm-ast-8b",
            "--prompt", text,
            "--json",
        ]
        
        payload_dict = {
            "text": text, 
            "latent": bridge_packet.latent_vector,
            "feedback": feedback or []
        }
        payload = json.dumps(payload_dict)
        try:
            result = subprocess.run(cmd, input=payload, capture_output=True, text=True, timeout=self.config.timeout_seconds)
            data = json.loads(result.stdout)
        except Exception:
            return self._fallback_reason(text, bridge_packet, feedback)
        
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
            telemetry={"adapter": "RlmAstLeftHemisphereAdapter", "backend": "rlm_runtime"},
        )

    def _fallback_reason(
        self, 
        text: str, 
        bridge_packet: PerceptionSummary,
        feedback: list[str] | None = None,
    ) -> ActionPlannerResult:
        response_text = "AST Leaf Resolution: " + text[:100]
        if feedback:
            response_text += f"\nRevisando devido ao feedback: {feedback[0][:50]}..."
        return ActionPlannerResult(
            response_text=response_text,
            lambda_program=TypedLambdaProgram(
                signature="Context -> Memory -> Decision",
                expression=json.dumps({"plan": ["respond_text"], "mode": "recursive_leaf"}, ensure_ascii=False),
                expected_effect="Resolve leaf node",
            ),
            actions=[
                PrimitiveAction(
                    action_type="respond_text",
                    typed_signature="ResponsePlan -> SafeTextMessage",
                    payload={"text": response_text},
                    safety_invariants=["safe output only", "typed runtime constraints"],
                )
            ],
            reasoning_summary=["ast_leaf_fallback", f"operation_trace={text[:60]}"],
            telemetry={"adapter": "RlmAstLeftHemisphereAdapter", "backend": "rlm_local_recursive"},
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
