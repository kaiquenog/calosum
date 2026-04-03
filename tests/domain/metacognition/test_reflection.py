from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from calosum import (
    CognitiveTokenizer,
    CognitiveTokenizerConfig,
    LinearReflectionController,
    ReflectionOutcome,
    ReflectionScore,
    CognitiveCandidate,
    CognitiveVariantSpec,
    AgentTurnResult,
    ActionPlannerResult,
    TypedLambdaProgram,
    UserTurn,
)
from calosum.shared.models.types import utc_now


class ReflectionTests(unittest.TestCase):
    def test_linear_reflection_always_selects_first_candidate(self) -> None:
        controller = LinearReflectionController()
        
        turn = UserTurn(session_id="s", user_text="text")
        variant = CognitiveVariantSpec(variant_id="winner")
        result = AgentTurnResult(
            user_turn=turn,
            memory_context=None,  # type: ignore
            right_state=None,     # type: ignore
            bridge_packet=None,   # type: ignore
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
        self.assertEqual(outcome.selected_by, "linear_no_branch")

    def test_neuroplasticity_interfaces_are_noop_but_safe(self) -> None:
        controller = LinearReflectionController()
        tokenizer = CognitiveTokenizer()
        outcome = ReflectionOutcome(selected_variant_id="any")
        
        # Should not raise
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
