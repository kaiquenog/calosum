from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from calosum import (
    ActionPlannerResult,
    AgentTurnResult,
    CognitiveCandidate,
    CognitiveTokenizer,
    CognitiveTokenizerConfig,
    CognitiveVariantSpec,
    GEAReflectionController,
    ReflectionOutcome,
    TypedLambdaProgram,
    UserTurn,
)
from calosum.shared.models.types import InputPerceptionState


class ReflectionTests(unittest.TestCase):
    def test_efe_reflection_selects_single_candidate(self) -> None:
        controller = GEAReflectionController()
        turn = UserTurn(session_id="s", user_text="text")
        variant = CognitiveVariantSpec(variant_id="winner")
        result = AgentTurnResult(
            user_turn=turn,
            memory_context=None,  # type: ignore[arg-type]
            right_state=InputPerceptionState(
                context_id=turn.turn_id,
                latent_vector=[0.0, 0.0, 0.0],
                latent_mu=[0.0, 0.0, 0.0],
                latent_logvar=[-2.0, -2.0, -2.0],
                salience=0.2,
                emotional_labels=["neutral"],
                world_hypotheses={},
                confidence=0.9,
                surprise_score=0.1,
            ),
            bridge_packet=None,   # type: ignore[arg-type]
            left_result=ActionPlannerResult(
                response_text="ok",
                lambda_program=TypedLambdaProgram("", "", ""),
                actions=[],
                reasoning_summary=[],
            ),
            telemetry={},
        )
        candidates = [CognitiveCandidate(variant=variant, turn_result=result)]
        outcome = controller.evaluate(candidates, None)
        self.assertEqual(outcome.selected_variant_id, "winner")
        self.assertEqual(outcome.selected_by, "efe_minimization_loop")

    def test_neuroplasticity_interfaces_are_noop_but_safe(self) -> None:
        controller = GEAReflectionController()
        tokenizer = CognitiveTokenizer()
        outcome = ReflectionOutcome(selected_variant_id="any")
        controller.apply_config_adaptation(tokenizer, outcome)
        controller.apply_neuroplasticity(tokenizer, outcome)

    def test_bridge_store_persists_config_independently(self) -> None:
        """
        Garante que a persistencia do estado da Bridge funciona mesmo sem GEA.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            from calosum.adapters.bridge.bridge_store import LocalBridgeStateStore
            
            store = LocalBridgeStateStore(
                adaptation_path=base / "bridge_config.json"
            )
            
            tokenizer = CognitiveTokenizer(
                CognitiveTokenizerConfig(salience_threshold=0.42),
                store=store
            )
            
            # Persistencia manual (simulando fim de turno ou evolução)
            tokenizer.persist_adaptation_state()
            
            reloaded = CognitiveTokenizer(
                CognitiveTokenizerConfig(),
                store=store
            )
            
            self.assertEqual(reloaded.config.salience_threshold, 0.42)


if __name__ == "__main__":
    unittest.main()
