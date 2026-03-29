from __future__ import annotations

import unittest

from calosum.adapters.llm_failover import ResilientLeftHemisphereAdapter
from calosum.shared.types import (
    BridgeControlSignal,
    CognitiveBridgePacket,
    LeftHemisphereResult,
    MemoryContext,
    PrimitiveAction,
    TypedLambdaProgram,
    UserTurn,
)


def _bridge_packet() -> CognitiveBridgePacket:
    return CognitiveBridgePacket(
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


class UnusableProvider:
    async def areason(self, *args, **kwargs) -> LeftHemisphereResult:
        return LeftHemisphereResult(
            response_text="",
            lambda_program=TypedLambdaProgram("Fallback", "()", "None"),
            actions=[],
            reasoning_summary=["Structured output parse failed: invalid json"],
            telemetry={"error": "timeout"},
        )

    async def arepair(self, *args, **kwargs) -> LeftHemisphereResult:
        return await self.areason(*args, **kwargs)


class HealthyProvider:
    async def areason(self, *args, **kwargs) -> LeftHemisphereResult:
        return LeftHemisphereResult(
            response_text="Resposta final",
            lambda_program=TypedLambdaProgram("Context -> Response", "lambda _: respond_text()", "respond"),
            actions=[
                PrimitiveAction(
                    "respond_text",
                    "ResponsePlan -> SafeTextMessage",
                    {"text": "Resposta final"},
                    [],
                )
            ],
            reasoning_summary=["ok"],
            telemetry={},
        )

    async def arepair(self, *args, **kwargs) -> LeftHemisphereResult:
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


if __name__ == "__main__":
    unittest.main()
