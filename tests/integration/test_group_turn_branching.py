from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from calosum.domain.agent.orchestrator import CalosumAgent
from calosum.domain.metacognition.metacognition import GroupTurnResult
from calosum.shared.models.types import (
    ActionPlannerResult,
    InputPerceptionState,
    PrimitiveAction,
    TypedLambdaProgram,
    UserTurn,
)


class _BranchingRightHemisphere:
    def perceive(self, user_turn, memory_context=None, workspace=None):
        return InputPerceptionState(
            context_id=user_turn.turn_id,
            latent_vector=[0.1, 0.2, 0.3],
            latent_mu=[0.1, 0.2, 0.3],
            latent_logvar=[-1.0, -1.0, -1.0],
            salience=0.85,
            emotional_labels=["ansioso"],
            world_hypotheses={"interaction_complexity": 0.9, "semantic_density": 0.8},
            confidence=0.7,
            surprise_score=0.88,
            telemetry={"jepa_uncertainty": 0.8},
        )

    async def aperceive(self, user_turn, memory_context=None, workspace=None):
        return self.perceive(user_turn, memory_context, workspace)


class _VariantAwareLeftHemisphere:
    def reason(self, user_turn, bridge_packet, memory_context, runtime_feedback=None, attempt=0, workspace=None):
        variant = bridge_packet.control.annotations.get("variant_label", "base")
        return ActionPlannerResult(
            response_text=f"response:{variant}",
            lambda_program=TypedLambdaProgram(
                "Context -> ResponsePlan",
                '{"plan":["respond_text"]}',
                "respond",
            ),
            actions=[
                PrimitiveAction(
                    action_type="respond_text",
                    typed_signature="ResponsePlan -> SafeTextMessage",
                    payload={"text": f"response:{variant}"},
                    safety_invariants=["safe_output"],
                )
            ],
            reasoning_summary=[f"variant={variant}"],
        )

    async def areason(self, *args, **kwargs):
        return self.reason(*args, **kwargs)

    def repair(self, *args, **kwargs):
        return self.reason(args[0], args[1], args[2])

    async def arepair(self, *args, **kwargs):
        return self.repair(*args, **kwargs)


class GroupTurnBranchingTests(unittest.TestCase):
    def test_agent_generates_group_turn_when_surprise_and_complexity_are_high(self) -> None:
        with patch.dict(os.environ, {"CALOSUM_GEA_MAX_CANDIDATES": "3"}):
            agent = CalosumAgent(
                right_hemisphere=_BranchingRightHemisphere(),
                left_hemisphere=_VariantAwareLeftHemisphere(),
            )

            result = agent.process_turn(
                UserTurn(session_id="branching", user_text="Preciso de um plano complexo com tradeoffs e comparacao.")
            )

        self.assertIsInstance(result, GroupTurnResult)
        assert isinstance(result, GroupTurnResult)
        self.assertGreaterEqual(len(result.candidates), 2)
        self.assertTrue(result.selected_result.left_result.response_text.startswith("response:"))


if __name__ == "__main__":
    unittest.main()
