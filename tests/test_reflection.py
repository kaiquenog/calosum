from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from calosum import (
    CalosumAgent,
    CognitiveTokenizer,
    CognitiveTokenizerConfig,
    CognitiveVariantSpec,
    GEAReflectionController,
    LeftHemisphereResult,
    PrimitiveAction,
    ReflectionOutcome,
    ReflectionScore,
    RightHemisphereState,
    TypedLambdaProgram,
    UserTurn,
)


class ReflectionTests(unittest.TestCase):
    def test_group_turn_selects_variant_and_updates_tokenizer(self) -> None:
        agent = CalosumAgent()
        turn = UserTurn(
            session_id="reflection-session",
            user_text="Estou ansioso e preciso de ajuda urgente para reorganizar o projeto.",
        )
        variants = [
            CognitiveVariantSpec(
                variant_id="empathetic_low_threshold",
                tokenizer_overrides={"salience_threshold": 0.45},
            ),
            CognitiveVariantSpec(
                variant_id="strict_high_threshold",
                tokenizer_overrides={"salience_threshold": 0.9},
            ),
        ]

        result = agent.process_group_turn(turn, variants)

        selected_variant = next(
            item for item in variants if item.variant_id == result.reflection.selected_variant_id
        )
        self.assertEqual(
            agent.tokenizer.config.salience_threshold,
            selected_variant.tokenizer_overrides["salience_threshold"],
        )
        self.assertEqual(result.reflection.cost_metrics["branch_count"], len(result.candidates))
        self.assertGreaterEqual(result.reflection.cost_metrics["total_latency_ms"], 0.0)
        dashboard = agent.cognitive_dashboard(turn.session_id)
        self.assertEqual(len(dashboard["reflection"]), 1)

    def test_neuroplasticity_persists_bridge_adjustments(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)

            from calosum.adapters.bridge_store import LocalBridgeStateStore
            store = LocalBridgeStateStore(
                weights_path=base / "bridge_weights.pt",
                adaptation_path=base / "bridge_config.json",
                reflection_history_path=base / "bridge_reflections.jsonl",
            )

            tokenizer = CognitiveTokenizer(
                CognitiveTokenizerConfig(),
                store=store,
            )
            controller = GEAReflectionController()
            outcome = ReflectionOutcome(
                selected_variant_id="winner",
                scoreboard=[ReflectionScore(variant_id="winner", score=1.0)],
                bridge_adjustments={
                    "salience_threshold": 0.42,
                    "base_temperature": 0.19,
                    "max_directives": 5,
                    "bottleneck_tokens": 7,
                    "salience_gain": 1.12,
                    "salience_bias": 0.03,
                    "temperature_bias": -0.02,
                },
                selected_metrics={"score": 1.0, "runtime_retry_count": 0},
            )

            controller.apply_neuroplasticity(tokenizer, outcome)

            reloaded = CognitiveTokenizer(
                CognitiveTokenizerConfig(),
                store=store,
            )
            history = (base / "bridge_reflections.jsonl").read_text(encoding="utf-8").splitlines()

            self.assertEqual(reloaded.config.salience_threshold, 0.42)
            self.assertEqual(reloaded.config.base_temperature, 0.19)
            self.assertEqual(reloaded.config.max_directives, 5)
            self.assertEqual(reloaded.config.bottleneck_tokens, 7)
            self.assertEqual(reloaded.config.salience_gain, 1.12)
            self.assertEqual(reloaded.config.salience_bias, 0.03)
            self.assertEqual(reloaded.config.temperature_bias, -0.02)
            self.assertEqual(len(history), 1)

    def test_process_turn_uses_default_personas_when_surprise_triggers_branching(self) -> None:
        class HighSurpriseRightHemisphere:
            def perceive(self, user_turn, memory_context=None):
                return RightHemisphereState(
                    context_id=user_turn.turn_id,
                    latent_vector=[0.1] * 16,
                    salience=0.88,
                    emotional_labels=["ansioso"],
                    world_hypotheses={"interaction_complexity": 0.92, "urgency": 0.88},
                    confidence=0.9,
                    surprise_score=0.91,
                    telemetry={},
                )

            async def aperceive(self, user_turn, memory_context=None):
                return self.perceive(user_turn, memory_context)

        class DirectiveEchoLeftHemisphere:
            def reason(self, user_turn, bridge_packet, memory_context, runtime_feedback=None, attempt=0):
                variant = bridge_packet.bridge_metadata.get("variant_label", "none")
                return LeftHemisphereResult(
                    response_text=f"variant={variant}",
                    lambda_program=TypedLambdaProgram(
                        "Context -> Response",
                        "lambda _: respond_text()",
                        "respond",
                    ),
                    actions=[
                        PrimitiveAction(
                            "respond_text",
                            "ResponsePlan -> SafeTextMessage",
                            {"text": f"variant={variant}"},
                            [],
                        )
                    ],
                    reasoning_summary=[variant],
                    telemetry={"system_directives": bridge_packet.control.system_directives},
                )

            async def areason(self, *args, **kwargs):
                return self.reason(*args[:3], runtime_feedback=kwargs.get("runtime_feedback"), attempt=kwargs.get("attempt", 0))

            def repair(self, *args, **kwargs):
                return self.reason(*args[:3])

            async def arepair(self, *args, **kwargs):
                return self.repair(*args, **kwargs)

        agent = CalosumAgent(
            right_hemisphere=HighSurpriseRightHemisphere(),
            left_hemisphere=DirectiveEchoLeftHemisphere(),
        )

        result = agent.process_turn(
            UserTurn(
                session_id="persona-session",
                user_text="Estou muito ansioso e preciso reorganizar o projeto.",
            )
        )

        variant_ids = [candidate.variant.variant_id for candidate in result.candidates]
        self.assertEqual(variant_ids, ["analitico", "empatico", "pragmatico"])

        directives_by_variant = {
            candidate.variant.variant_id: candidate.turn_result.left_result.telemetry["system_directives"]
            for candidate in result.candidates
        }
        self.assertTrue(
            any("consistencia logica" in directive for directive in directives_by_variant["analitico"])
        )
        self.assertTrue(
            any("impacto emocional" in directive for directive in directives_by_variant["empatico"])
        )
        self.assertTrue(
            any("fronteira de acoes" in directive for directive in directives_by_variant["pragmatico"])
        )


if __name__ == "__main__":
    unittest.main()
