from __future__ import annotations

import unittest

from calosum.adapters.llm.llm_failover import ResilientLeftHemisphereAdapter
from calosum.shared.models.types import (
    BridgeControlSignal,
    PerceptionSummary,
    ActionPlannerResult,
    MemoryContext,
    PrimitiveAction,
    TypedLambdaProgram,
    UserTurn,
)


def _bridge_packet() -> PerceptionSummary:
    return PerceptionSummary(
        context_id="ctx-1",
        soft_prompts=[],
        control=BridgeControlSignal(
            target_temperature=0.2,
            empathy_priority=False,
            system_directives=["be precise"],
            annotations={},
        ),
        salience=0.2,
        bridge_metadata={},
    )


def _bridge_packet_emergency() -> PerceptionSummary:
    return PerceptionSummary(
        context_id="ctx-1",
        soft_prompts=[],
        control=BridgeControlSignal(
            target_temperature=0.2,
            empathy_priority=False,
            system_directives=["high uncertainty: prioritize epistemic foraging with tools before final response"],
            annotations={"jepa_uncertainty": 0.9, "perception_status": "degraded"},
        ),
        salience=0.2,
        bridge_metadata={},
    )


class UnusableProvider:
    async def areason(self, *args, **kwargs) -> ActionPlannerResult:
        return ActionPlannerResult(
            response_text="",
            lambda_program=TypedLambdaProgram("Fallback", "()", "None"),
            actions=[],
            reasoning_summary=["Structured output parse failed: invalid json"],
            telemetry={"error": "timeout"},
        )

    async def arepair(self, *args, **kwargs) -> ActionPlannerResult:
        return await self.areason(*args, **kwargs)


class HealthyProvider:
    def __init__(self, response_text: str = "Resposta final") -> None:
        self.response_text = response_text

    async def areason(self, *args, **kwargs) -> ActionPlannerResult:
        return ActionPlannerResult(
            response_text=self.response_text,
            lambda_program=TypedLambdaProgram("Context -> Response", "lambda _: respond_text()", "respond"),
            actions=[
                PrimitiveAction(
                    "respond_text",
                    "ResponsePlan -> SafeTextMessage",
                    {"text": self.response_text},
                    [],
                )
            ],
            reasoning_summary=["ok"],
            telemetry={},
        )

    async def arepair(self, *args, **kwargs) -> ActionPlannerResult:
        return await self.areason(*args, **kwargs)


class LlmFailoverTests(unittest.IsolatedAsyncioTestCase):
    async def test_failover_uses_secondary_provider_when_primary_returns_unusable_result(self) -> None:
        adapter = ResilientLeftHemisphereAdapter([UnusableProvider(), HealthyProvider()])

        result = await adapter.areason(
            UserTurn(session_id="failover-session", user_text="Oi"),
            _bridge_packet(),
            MemoryContext(),
        )

        self.assertEqual(result.response_text, "Resposta final")
        self.assertEqual(result.telemetry["failover_attempt_count"], 2)
        self.assertFalse(result.telemetry["failover_exhausted"])
        self.assertEqual(len(result.telemetry["failover_attempts"]), 2)

    async def test_failover_returns_fallback_when_all_providers_fail(self) -> None:
        adapter = ResilientLeftHemisphereAdapter([UnusableProvider(), UnusableProvider()])

        result = await adapter.areason(
            UserTurn(session_id="failover-session", user_text="Oi"),
            _bridge_packet(),
            MemoryContext(),
        )

        self.assertIn("falharam temporariamente", result.response_text)
        self.assertTrue(result.telemetry["failover_exhausted"])
        self.assertEqual(result.telemetry["failover_attempt_count"], 2)

    async def test_failover_uses_context_routing_for_emergency_state(self) -> None:
        adapter = ResilientLeftHemisphereAdapter([HealthyProvider("primario"), HealthyProvider("emergencia")])

        result = await adapter.areason(
            UserTurn(session_id="failover-session", user_text="Oi"),
            _bridge_packet_emergency(),
            MemoryContext(),
        )

        self.assertEqual(result.response_text, "emergencia")
        self.assertEqual(result.telemetry["routing_reason"], "context_route_emergency")
        self.assertEqual(result.telemetry["failover_attempt_count"], 1)


if __name__ == "__main__":
    unittest.main()
