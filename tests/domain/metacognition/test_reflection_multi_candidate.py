from __future__ import annotations

import unittest

from calosum.domain.metacognition.metacognition import (
    CognitiveCandidate,
    CognitiveVariantSpec,
    GEAReflectionController,
)
from calosum.shared.models.types import (
    ActionPlannerResult,
    AgentTurnResult,
    InputPerceptionState,
    MemoryContext,
    PerceptionSummary,
    BridgeControlSignal,
    TypedLambdaProgram,
    UserTurn,
)


def _candidate(variant_id: str, *, tool_success_rate: float, peer_count: int = 0, emotional: bool = False) -> CognitiveCandidate:
    turn = UserTurn(session_id="s", user_text="Preciso comparar abordagens com empatia.")
    state = InputPerceptionState(
        context_id=turn.turn_id,
        latent_vector=[0.0, 0.1, 0.0],
        latent_mu=[0.0, 0.1, 0.0],
        latent_logvar=[-2.0, -2.0, -2.0],
        salience=0.8,
        emotional_labels=["ansioso"] if emotional else ["neutral"],
        world_hypotheses={"interaction_complexity": 0.8},
        confidence=0.85,
        surprise_score=0.72,
        telemetry={"peer_latents_count": peer_count, "peer_latent_alignment": 0.9},
    )
    result = AgentTurnResult(
        user_turn=turn,
        memory_context=MemoryContext(),
        right_state=state,
        bridge_packet=PerceptionSummary(
            context_id=turn.turn_id,
            soft_prompts=[],
            control=BridgeControlSignal(target_temperature=0.2, empathy_priority=emotional),
            salience=0.8,
        ),
        left_result=ActionPlannerResult(
            response_text=f"variant={variant_id}",
            lambda_program=TypedLambdaProgram("Context -> Response", '{"plan":["respond_text"]}', "respond"),
            actions=[],
            reasoning_summary=[],
        ),
        telemetry={
            "felt": {},
            "thought": {},
            "decision": {"tool_success_rate": tool_success_rate},
            "capabilities": {},
            "bridge_config": {},
            "active_variant": variant_id,
        },
    )
    return CognitiveCandidate(
        variant=CognitiveVariantSpec(variant_id=variant_id),
        turn_result=result,
    )


class ReflectionMultiCandidateTests(unittest.TestCase):
    def test_reflection_scores_multiple_candidates_and_keeps_candidate_count(self) -> None:
        controller = GEAReflectionController()
        candidates = [
            _candidate("base", tool_success_rate=0.7),
            _candidate("analitico", tool_success_rate=0.95),
            _candidate("empatico", tool_success_rate=0.8, peer_count=4, emotional=True),
        ]

        outcome = controller.evaluate(candidates, None)

        self.assertEqual(outcome.cost_metrics["candidate_count"], 3)
        self.assertEqual(len(outcome.scoreboard), 3)
        self.assertIn(outcome.selected_variant_id, {"analitico", "empatico"})

    def test_peer_latents_can_shift_selection_towards_empatico(self) -> None:
        controller = GEAReflectionController()
        candidates = [
            _candidate("base", tool_success_rate=0.9),
            _candidate("empatico", tool_success_rate=0.9, peer_count=6, emotional=True),
        ]

        outcome = controller.evaluate(candidates, None)

        self.assertEqual(outcome.selected_variant_id, "empatico")


if __name__ == "__main__":
    unittest.main()
