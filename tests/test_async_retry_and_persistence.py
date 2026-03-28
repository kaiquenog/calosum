from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from calosum import (
    CalosumAgent,
    CalosumAgentConfig,
    CognitiveTelemetryBus,
    LeftHemisphereResult,
    OTLPJsonlTelemetrySink,
    PersistentDualMemorySystem,
    PrimitiveAction,
    TypedLambdaProgram,
    UserTurn,
)


class FaultyLeftHemisphere:
    def reason(self, user_turn, bridge_packet, memory_context, runtime_feedback=None, attempt=0):
        return LeftHemisphereResult(
            response_text="unsafe attempt",
            lambda_program=TypedLambdaProgram(
                signature="Context -> UnsafeDecision",
                expression="lambda ctx: call_external_api()",
                expected_effect="unsafe",
            ),
            actions=[
                PrimitiveAction(
                    action_type="call_external_api",
                    typed_signature="Request -> Response",
                    payload={"endpoint": "https://example.com"},
                    safety_invariants=["requires approval"],
                )
            ],
            reasoning_summary=[f"attempt={attempt}", f"feedback={len(runtime_feedback or [])}"],
        )

    async def areason(
        self, user_turn, bridge_packet, memory_context, runtime_feedback=None, attempt=0
    ):
        return self.reason(user_turn, bridge_packet, memory_context, runtime_feedback, attempt)

    def repair(
        self,
        user_turn,
        bridge_packet,
        memory_context,
        previous_result,
        rejected_results,
        attempt,
    ):
        return LeftHemisphereResult(
            response_text="repaired response",
            lambda_program=TypedLambdaProgram(
                signature="Context -> SafeDecision",
                expression="lambda ctx: respond_text()",
                expected_effect="safe",
            ),
            actions=[
                PrimitiveAction(
                    action_type="respond_text",
                    typed_signature="ResponsePlan -> SafeTextMessage",
                    payload={"text": "repaired response", "temperature": 0.2},
                    safety_invariants=["text only"],
                )
            ],
            reasoning_summary=[f"attempt={attempt}", f"rejected={len(rejected_results)}"],
        )

    async def arepair(
        self,
        user_turn,
        bridge_packet,
        memory_context,
        previous_result,
        rejected_results,
        attempt,
    ):
        return self.repair(
            user_turn,
            bridge_packet,
            memory_context,
            previous_result,
            rejected_results,
            attempt,
        )


class MockLeftHemisphere:
    def reason(self, user_turn, bridge_packet, memory_context, runtime_feedback=None, attempt=0):
        return LeftHemisphereResult(
            response_text="Mocked response",
            lambda_program=TypedLambdaProgram("Context -> Response", "()", "None"),
            actions=[],
            reasoning_summary=[],
        )

    async def areason(self, user_turn, bridge_packet, memory_context, runtime_feedback=None, attempt=0):
        return self.reason(user_turn, bridge_packet, memory_context, runtime_feedback, attempt)

    def repair(self, *args, **kwargs):
        return self.reason(*args[:3])

    async def arepair(self, *args, **kwargs):
        return self.reason(*args[:3])

class AsyncRetryAndPersistenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_pipeline_retries_after_runtime_rejection(self) -> None:
        agent = CalosumAgent(
            left_hemisphere=FaultyLeftHemisphere(),
            config=CalosumAgentConfig(max_runtime_retries=1),
        )

        result = await agent.aprocess_turn(
            UserTurn(session_id="async-session", user_text="Preciso de ajuda urgente.")
        )

        self.assertEqual(result.runtime_retry_count, 1)
        self.assertTrue(all(item.status == "executed" for item in result.execution_results))
        self.assertEqual(result.left_result.response_text, "repaired response")
        dashboard = agent.cognitive_dashboard("async-session")
        self.assertEqual(dashboard["decision"][0]["runtime_retry_count"], 1)

    async def test_persistent_memory_and_otlp_sink_survive_reloads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            memory_system = PersistentDualMemorySystem.from_directory(base / "memory")
            telemetry_bus = CognitiveTelemetryBus(OTLPJsonlTelemetrySink(base / "telemetry.jsonl"))
            agent = CalosumAgent(left_hemisphere=MockLeftHemisphere(), memory_system=memory_system, telemetry_bus=telemetry_bus)

            repeated_preference = (
                "Prefiro respostas curtas com passos claros quando a situacao estiver urgente."
            )
            await agent.aprocess_turn(
                UserTurn(session_id="persist-session", user_text=repeated_preference)
            )
            await agent.aprocess_turn(
                UserTurn(session_id="persist-session", user_text=repeated_preference)
            )
            await agent.asleep_mode()

            reloaded = PersistentDualMemorySystem.from_directory(base / "memory")
            context = reloaded.build_context(
                UserTurn(
                    session_id="persist-session",
                    user_text="Preciso de um plano urgente para o projeto.",
                )
            )

            self.assertGreaterEqual(len(reloaded.episodic_store.all()), 2)
            self.assertTrue(any(rule.rule_id == "emotion::urgente" for rule in reloaded.semantic_store.all()))
            self.assertTrue(
                any(
                    triple.predicate == "prefers_response_style" and triple.object == "short"
                    for triple in context.knowledge_triples
                )
            )

            lines = (base / "telemetry.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertGreaterEqual(len(lines), 4)
            envelope = json.loads(lines[0])
            self.assertIn("resourceSpans", envelope)


if __name__ == "__main__":
    unittest.main()
