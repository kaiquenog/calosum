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


class _PeerAwareLatentExchange:
    async def broadcast_latent(self, session_id: str, latent_vector: list[float]) -> None:
        return None

    async def get_peer_latents(self, session_id: str) -> list[list[float]]:
        return [[0.1, 0.2, 0.3], [0.1, 0.19, 0.29], [0.11, 0.18, 0.31]]


class _EmotionRightHemisphere:
    def perceive(self, user_turn, memory_context=None, workspace=None):
        return InputPerceptionState(
            context_id=user_turn.turn_id,
            latent_vector=[0.1, 0.2, 0.3],
            latent_mu=[0.1, 0.2, 0.3],
            latent_logvar=[-1.0, -1.0, -1.0],
            salience=0.9,
            emotional_labels=["ansioso", "frustrado"],
            world_hypotheses={"interaction_complexity": 0.8, "semantic_density": 0.7},
            confidence=0.82,
            surprise_score=0.81,
            telemetry={"jepa_uncertainty": 0.5},
        )

    async def aperceive(self, user_turn, memory_context=None, workspace=None):
        return self.perceive(user_turn, memory_context, workspace)


class _FlatLeftHemisphere:
    def reason(self, user_turn, bridge_packet, memory_context, runtime_feedback=None, attempt=0, workspace=None):
        variant = bridge_packet.control.annotations.get("variant_label", "base")
        return ActionPlannerResult(
            response_text=f"selected:{variant}",
            lambda_program=TypedLambdaProgram("Context -> ResponsePlan", '{"plan":["respond_text"]}', "respond"),
            actions=[
                PrimitiveAction(
                    action_type="respond_text",
                    typed_signature="ResponsePlan -> SafeTextMessage",
                    payload={"text": f"selected:{variant}"},
                    safety_invariants=["safe_output"],
                )
            ],
            reasoning_summary=[variant],
        )

    async def areason(self, *args, **kwargs):
        return self.reason(*args, **kwargs)

    def repair(self, *args, **kwargs):
        return self.reason(args[0], args[1], args[2])

    async def arepair(self, *args, **kwargs):
        return self.repair(*args, **kwargs)


class LatentExchangeSelectionTests(unittest.TestCase):
    def test_peer_latents_bias_multi_candidate_selection(self) -> None:
        with patch.dict(os.environ, {"CALOSUM_GEA_MAX_CANDIDATES": "3"}):
            agent = CalosumAgent(
                right_hemisphere=_EmotionRightHemisphere(),
                left_hemisphere=_FlatLeftHemisphere(),
                latent_exchange=_PeerAwareLatentExchange(),
            )

            result = agent.process_turn(UserTurn(session_id="latent", user_text="Preciso de ajuda complexa e sensivel."))

        self.assertIsInstance(result, GroupTurnResult)
        assert isinstance(result, GroupTurnResult)
        self.assertEqual(result.reflection.selected_variant_id, "empatico")


if __name__ == "__main__":
    unittest.main()
