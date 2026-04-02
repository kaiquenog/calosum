from __future__ import annotations

import unittest

from calosum.adapters.contract_wrappers import (
    ContractEnforcedLeftHemisphereAdapter,
    ContractEnforcedRightHemisphereAdapter,
)
from calosum.shared.models.types import (
    BridgeControlSignal,
    CognitiveBridgePacket,
    LeftHemisphereResult,
    MemoryContext,
    PrimitiveAction,
    RightHemisphereState,
    TypedLambdaProgram,
    UserTurn,
)


def _user_turn() -> UserTurn:
    return UserTurn(session_id="wrapper-test", user_text="teste")


def _bridge() -> CognitiveBridgePacket:
    return CognitiveBridgePacket(
        context_id="ctx",
        soft_prompts=[],
        control=BridgeControlSignal(
            target_temperature=0.2,
            empathy_priority=False,
            system_directives=[],
        ),
        salience=0.2,
        bridge_metadata={},
    )


class _LeftProviderInvalid:
    def reason(self, *_args, **_kwargs) -> LeftHemisphereResult:
        return LeftHemisphereResult(
            response_text="",
            lambda_program=TypedLambdaProgram(signature="", expression="", expected_effect=""),
            actions=[],
            reasoning_summary=[],
            telemetry={},
        )

    async def areason(self, *_args, **_kwargs) -> LeftHemisphereResult:
        return self.reason()

    def repair(self, *_args, **_kwargs) -> LeftHemisphereResult:
        return self.reason()

    async def arepair(self, *_args, **_kwargs) -> LeftHemisphereResult:
        return self.reason()


class _LeftProviderPayloadResponse:
    def reason(self, *_args, **_kwargs) -> LeftHemisphereResult:
        return LeftHemisphereResult(
            response_text="",
            lambda_program=TypedLambdaProgram(
                signature="Context -> Response",
                expression="(lambda context (emit respond_text))",
                expected_effect="Deliver response",
            ),
            actions=[
                PrimitiveAction(
                    action_type="respond_text",
                    typed_signature="ResponsePlan -> SafeTextMessage",
                    payload={"text": "ok via action payload"},
                    safety_invariants=["safe output only"],
                )
            ],
            reasoning_summary=["raw"],
            telemetry={},
        )

    async def areason(self, *_args, **_kwargs) -> LeftHemisphereResult:
        return self.reason()

    def repair(self, *_args, **_kwargs) -> LeftHemisphereResult:
        return self.reason()

    async def arepair(self, *_args, **_kwargs) -> LeftHemisphereResult:
        return self.reason()


class _RightProviderSparse:
    def perceive(self, *_args, **_kwargs) -> RightHemisphereState:
        return RightHemisphereState(
            context_id="ctx-r",
            latent_vector=[],
            salience=0.3,
            emotional_labels=[],
            world_hypotheses={"interaction_complexity": "bad", "urgency": 2.0},
            confidence=0.7,
            surprise_score=0.4,
            telemetry={},
        )

    async def aperceive(self, *_args, **_kwargs) -> RightHemisphereState:
        return self.perceive()


class ContractWrappersTests(unittest.IsolatedAsyncioTestCase):
    async def test_left_wrapper_injects_fallback_for_empty_output(self) -> None:
        wrapper = ContractEnforcedLeftHemisphereAdapter(_LeftProviderInvalid())
        result = await wrapper.areason(_user_turn(), _bridge(), MemoryContext())

        self.assertTrue(result.response_text.strip())
        self.assertTrue(result.actions)
        self.assertEqual(result.actions[0].action_type, "respond_text")
        self.assertEqual(result.lambda_program.signature, "Context -> Response")
        self.assertTrue(result.reasoning_summary)
        self.assertEqual(result.telemetry["contract_wrapper"], "left_v1")

    async def test_left_wrapper_recovers_response_from_action_payload(self) -> None:
        wrapper = ContractEnforcedLeftHemisphereAdapter(_LeftProviderPayloadResponse())
        result = await wrapper.areason(_user_turn(), _bridge(), MemoryContext())

        self.assertEqual(result.response_text, "ok via action payload")
        self.assertEqual(result.actions[0].action_type, "respond_text")

    async def test_right_wrapper_defaults_sparse_state(self) -> None:
        wrapper = ContractEnforcedRightHemisphereAdapter(_RightProviderSparse())
        state = await wrapper.aperceive(_user_turn(), MemoryContext())

        self.assertEqual(state.latent_vector, [0.0])
        self.assertEqual(state.emotional_labels, ["neutral"])
        self.assertIn("urgency", state.world_hypotheses)
        self.assertEqual(state.world_hypotheses["urgency"], 1.0)
        self.assertEqual(state.telemetry["contract_wrapper"], "right_v1")


if __name__ == "__main__":
    unittest.main()
