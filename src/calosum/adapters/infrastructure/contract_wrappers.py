from __future__ import annotations

import math
from typing import Any

from calosum.shared.models.types import (
    ActionExecutionResult,
    PerceptionSummary,
    CognitiveWorkspace,
    ActionPlannerResult,
    MemoryContext,
    PrimitiveAction,
    InputPerceptionState,
    TypedLambdaProgram,
    UserTurn,
)


class ContractEnforcedLeftHemisphereAdapter:
    """Enforce minimal ActionPlannerResult contract regardless of provider backend."""

    def __init__(self, provider: Any) -> None:
        self.provider = provider

    def __getattr__(self, name: str) -> Any:
        return getattr(self.provider, name)

    def reason(
        self,
        user_turn: UserTurn,
        bridge_packet: PerceptionSummary,
        memory_context: MemoryContext,
        runtime_feedback: list[str] | None = None,
        attempt: int = 0,
        workspace: CognitiveWorkspace | None = None,
    ) -> ActionPlannerResult:
        try:
            result = self.provider.reason(
                user_turn,
                bridge_packet,
                memory_context,
                runtime_feedback,
                attempt,
                workspace,
            )
        except TypeError:
            result = self.provider.reason(
                user_turn,
                bridge_packet,
                memory_context,
                runtime_feedback,
                attempt,
            )
        return self._normalize(result)

    async def areason(
        self,
        user_turn: UserTurn,
        bridge_packet: PerceptionSummary,
        memory_context: MemoryContext,
        runtime_feedback: list[str] | None = None,
        attempt: int = 0,
        workspace: CognitiveWorkspace | None = None,
    ) -> ActionPlannerResult:
        result = await self.provider.areason(
            user_turn,
            bridge_packet,
            memory_context,
            runtime_feedback,
            attempt,
            workspace,
        )
        return self._normalize(result)

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
        result = self.provider.repair(
            user_turn,
            bridge_packet,
            memory_context,
            previous_result,
            rejected_results,
            attempt,
            critique_feedback,
            workspace,
        )
        return self._normalize(result)

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
        result = await self.provider.arepair(
            user_turn,
            bridge_packet,
            memory_context,
            previous_result,
            rejected_results,
            attempt,
            critique_feedback,
            workspace,
        )
        return self._normalize(result)

    def _normalize(self, result: Any) -> ActionPlannerResult:
        if not isinstance(result, ActionPlannerResult):
            return self._fallback_result("invalid_result_type")

        adjustments: list[str] = []
        response_text = str(result.response_text or "").strip()
        reasoning_summary = [str(item) for item in result.reasoning_summary if str(item).strip()]
        telemetry = dict(result.telemetry)

        lambda_program = self._normalize_lambda(result.lambda_program, adjustments)
        actions = self._normalize_actions(result.actions, adjustments)

        if not response_text:
            response_text = _response_from_actions(actions)
            if response_text:
                adjustments.append("response_text_recovered_from_action_payload")

        if response_text and not actions:
            actions = [_respond_action(response_text)]
            adjustments.append("respond_text_action_injected")

        if not response_text and not actions:
            response_text = "Desculpe, tive uma falha de contrato no raciocínio. Vou tentar novamente."
            actions = [_respond_action(response_text)]
            adjustments.append("fallback_response_injected")

        if not reasoning_summary:
            reasoning_summary = ["contract_wrapper_normalized_output"]
            adjustments.append("reasoning_summary_defaulted")

        telemetry["contract_wrapper"] = "left_v1"
        telemetry["contract_provider"] = self.provider.__class__.__name__
        if adjustments:
            telemetry["contract_adjustments"] = adjustments

        return ActionPlannerResult(
            response_text=response_text,
            lambda_program=lambda_program,
            actions=actions,
            reasoning_summary=reasoning_summary,
            telemetry=telemetry,
        )

    def _normalize_lambda(
        self,
        program: TypedLambdaProgram,
        adjustments: list[str],
    ) -> TypedLambdaProgram:
        signature = str(getattr(program, "signature", "") or "").strip()
        expression = str(getattr(program, "expression", "") or "").strip()
        effect = str(getattr(program, "expected_effect", "") or "").strip()

        if not signature:
            signature = "Context -> Response"
            adjustments.append("lambda_signature_defaulted")
        if not expression:
            expression = "(lambda context memory (sequence (emit respond_text)))"
            adjustments.append("lambda_expression_defaulted")
        if not effect:
            effect = "Deliver safe response"
            adjustments.append("lambda_expected_effect_defaulted")
        return TypedLambdaProgram(signature=signature, expression=expression, expected_effect=effect)

    def _normalize_actions(
        self,
        raw_actions: list[PrimitiveAction],
        adjustments: list[str],
    ) -> list[PrimitiveAction]:
        out: list[PrimitiveAction] = []
        for item in raw_actions:
            if not isinstance(item, PrimitiveAction):
                continue
            action_type = str(item.action_type or "").strip() or "respond_text"
            typed_signature = str(item.typed_signature or "").strip()
            payload = dict(item.payload or {})
            invariants = [str(x) for x in item.safety_invariants if isinstance(x, str) and x.strip()]

            if not typed_signature:
                typed_signature = "ResponsePlan -> SafeTextMessage"
                adjustments.append(f"typed_signature_defaulted:{action_type}")
            if not invariants:
                invariants = ["safe output only"]
                adjustments.append(f"safety_invariants_defaulted:{action_type}")
            out.append(
                PrimitiveAction(
                    action_type=action_type,
                    typed_signature=typed_signature,
                    payload=payload,
                    safety_invariants=invariants,
                )
            )
        return out

    def _fallback_result(self, error: str) -> ActionPlannerResult:
        response = "Desculpe, tive uma falha de contrato no raciocínio. Vou tentar novamente."
        return ActionPlannerResult(
            response_text=response,
            lambda_program=TypedLambdaProgram(
                signature="Context -> Response",
                expression="(lambda context memory (sequence (emit respond_text)))",
                expected_effect="Deliver safe response",
            ),
            actions=[_respond_action(response)],
            reasoning_summary=[f"contract_wrapper_fallback:{error}"],
            telemetry={
                "contract_wrapper": "left_v1",
                "contract_provider": self.provider.__class__.__name__,
                "contract_error": error,
            },
        )


class ContractEnforcedRightHemisphereAdapter:
    """Normalize InputPerceptionState output from multiple perception adapters."""

    def __init__(self, provider: Any) -> None:
        self.provider = provider

    def __getattr__(self, name: str) -> Any:
        return getattr(self.provider, name)

    def perceive(
        self,
        user_turn: UserTurn,
        memory_context: MemoryContext | None = None,
        workspace: CognitiveWorkspace | None = None,
    ) -> InputPerceptionState:
        try:
            result = self.provider.perceive(user_turn, memory_context, workspace)
        except TypeError:
            result = self.provider.perceive(user_turn, memory_context)
        return self._normalize(result, fallback_context_id=user_turn.turn_id)

    async def aperceive(
        self,
        user_turn: UserTurn,
        memory_context: MemoryContext | None = None,
        workspace: CognitiveWorkspace | None = None,
    ) -> InputPerceptionState:
        if hasattr(self.provider, "aperceive"):
            try:
                result = await self.provider.aperceive(user_turn, memory_context, workspace)
            except TypeError:
                result = await self.provider.aperceive(user_turn, memory_context)
        else:
            result = self.provider.perceive(user_turn, memory_context, workspace)
        return self._normalize(result, fallback_context_id=user_turn.turn_id)

    def _normalize(self, result: Any, *, fallback_context_id: str) -> InputPerceptionState:
        if not isinstance(result, InputPerceptionState):
            return _fallback_right_state(fallback_context_id, self.provider.__class__.__name__, "invalid_result_type")

        adjustments: list[str] = []
        context_id = str(result.context_id or fallback_context_id)
        latent = _normalize_latent(result.latent_vector)
        if not latent:
            latent = [0.0]
            adjustments.append("latent_defaulted")

        latent_mu = _normalize_latent(result.latent_mu) if result.latent_mu is not None else None
        latent_logvar = _normalize_latent(result.latent_logvar) if result.latent_logvar is not None else None

        salience = _clamp01(result.salience)
        confidence = _clamp01(result.confidence)
        surprise = _clamp01(result.surprise_score)
        if salience != result.salience:
            adjustments.append("salience_clamped")
        if confidence != result.confidence:
            adjustments.append("confidence_clamped")
        if surprise != result.surprise_score:
            adjustments.append("surprise_clamped")

        labels = [str(label).strip() for label in result.emotional_labels if str(label).strip()]
        if not labels:
            labels = ["neutral"]
            adjustments.append("emotional_labels_defaulted")

        world = _normalize_world_hypotheses(result.world_hypotheses)
        telemetry = dict(result.telemetry)
        telemetry["contract_wrapper"] = "right_v1"
        telemetry["contract_provider"] = self.provider.__class__.__name__
        if adjustments:
            telemetry["contract_adjustments"] = adjustments

        try:
            return InputPerceptionState(
                context_id=context_id,
                latent_vector=latent,
                latent_mu=latent_mu,
                latent_logvar=latent_logvar,
                salience=salience,
                emotional_labels=labels,
                world_hypotheses=world,
                confidence=confidence,
                surprise_score=surprise,
                telemetry=telemetry,
            )
        except Exception as exc:
            return _fallback_right_state(context_id, self.provider.__class__.__name__, repr(exc))


def _respond_action(text: str) -> PrimitiveAction:
    return PrimitiveAction(
        action_type="respond_text",
        typed_signature="ResponsePlan -> SafeTextMessage",
        payload={"text": text},
        safety_invariants=["safe output only"],
    )


def _response_from_actions(actions: list[PrimitiveAction]) -> str:
    for action in actions:
        if action.action_type != "respond_text":
            continue
        candidate = action.payload.get("text")
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return ""


def _normalize_latent(raw: list[Any]) -> list[float]:
    out: list[float] = []
    for value in raw:
        try:
            number = float(value)
        except Exception:
            continue
        if not math.isfinite(number):
            continue
        out.append(number)
    return out


def _normalize_world_hypotheses(raw: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in dict(raw or {}).items():
        try:
            number = float(value)
        except Exception:
            continue
        if not math.isfinite(number):
            continue
        out[str(key)] = _clamp01(number)
    return out


def _clamp01(value: Any) -> float:
    try:
        number = float(value)
    except Exception:
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return max(0.0, min(1.0, number))


def _fallback_right_state(context_id: str, provider_name: str, error: str) -> InputPerceptionState:
    return InputPerceptionState(
        context_id=context_id,
        latent_vector=[0.0],
        latent_mu=None,
        latent_logvar=None,
        salience=0.2,
        emotional_labels=["neutral"],
        world_hypotheses={"interaction_complexity": 0.0},
        confidence=0.5,
        surprise_score=0.5,
        telemetry={
            "contract_wrapper": "right_v1",
            "contract_provider": provider_name,
            "contract_error": error,
            "degraded_reason": "contract_wrapper_fallback",
        },
    )
