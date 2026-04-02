from __future__ import annotations

import unittest

from calosum.adapters.llm.llm_fusion import (
    FusionResult,
    MultiSampleFusionConfig,
    MultiSampleFusionLeftHemisphereAdapter,
    SemanticFusionSelector,
)
from calosum.shared.models.types import (
    BridgeControlSignal,
    CognitiveBridgePacket,
    LeftHemisphereResult,
    MemoryContext,
    TypedLambdaProgram,
    UserTurn,
)


def _result(text: str) -> LeftHemisphereResult:
    return LeftHemisphereResult(
        response_text=text,
        lambda_program=TypedLambdaProgram(
            "Context -> Response",
            "lambda _: respond_text()",
            "respond",
        ),
        actions=[],
        reasoning_summary=[text],
        telemetry={},
    )


def _packet(uncertainty: float, temperature: float = 0.3) -> CognitiveBridgePacket:
    return CognitiveBridgePacket(
        context_id="ctx-1",
        soft_prompts=[],
        control=BridgeControlSignal(
            target_temperature=temperature,
            empathy_priority=False,
            system_directives=[],
            annotations={"jepa_uncertainty": uncertainty},
        ),
        salience=0.2,
        latent_vector=[0.01] * 384,
        bridge_metadata={},
    )


class _StubSelector:
    def select(self, candidates, jepa_pred, uncertainty):
        return FusionResult(
            result=candidates[2],
            selected_index=2,
            method="jepa_guided",
            score=0.88,
            scores=[0.2, 0.3, 0.88],
        )


class _DummyProvider:
    def __init__(self) -> None:
        self.temperatures: list[float] = []

    async def areason(
        self,
        user_turn,
        bridge_packet,
        memory_context,
        runtime_feedback=None,
        attempt=0,
        workspace=None,
    ):
        self.temperatures.append(round(float(bridge_packet.control.target_temperature), 3))
        return _result(f"temp={bridge_packet.control.target_temperature:.2f}")

    async def arepair(
        self,
        user_turn,
        bridge_packet,
        memory_context,
        previous_result,
        rejected_results,
        attempt,
        critique_feedback=None,
        workspace=None,
    ):
        return _result("repair")


class SemanticFusionSelectorTests(unittest.TestCase):
    def test_selector_passthrough_when_uncertainty_is_high(self) -> None:
        selector = SemanticFusionSelector(uncertainty_threshold=0.5)
        candidates = [_result("a"), _result("b")]
        chosen = selector.select(candidates, [0.0] * 384, uncertainty=0.9)
        self.assertEqual(chosen.method, "passthrough")
        self.assertEqual(chosen.selected_index, 0)

    def test_selector_prefers_semantically_aligned_candidate(self) -> None:
        selector = SemanticFusionSelector(uncertainty_threshold=0.5)
        candidates = [_result("planejar rollout incremental"), _result("poesia abstrata")]
        prediction = selector._encode_text("planejar rollout incremental")
        chosen = selector.select(candidates, prediction, uncertainty=0.1)
        self.assertEqual(chosen.method, "jepa_guided")
        self.assertEqual(chosen.selected_index, 0)


class MultiSampleFusionAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_multi_sample_triggers_when_uncertainty_is_low(self) -> None:
        provider = _DummyProvider()
        adapter = MultiSampleFusionLeftHemisphereAdapter(
            provider,
            config=MultiSampleFusionConfig(enabled=True, n_candidates=3, uncertainty_threshold=0.5),
            selector=_StubSelector(),
        )
        result = await adapter.areason(
            UserTurn(session_id="s1", user_text="oi"),
            _packet(uncertainty=0.2, temperature=0.3),
            MemoryContext(recent_episodes=[], semantic_rules=[], knowledge_triples=[]),
        )

        self.assertEqual(provider.temperatures, [0.3, 0.4, 0.2])
        self.assertEqual(result.telemetry["fusion_method"], "jepa_guided")
        self.assertEqual(result.telemetry["fusion_selected_index"], 2)
        self.assertEqual(result.telemetry["fusion_candidate_count"], 3)

    async def test_multi_sample_falls_back_to_passthrough_when_uncertainty_is_high(self) -> None:
        provider = _DummyProvider()
        adapter = MultiSampleFusionLeftHemisphereAdapter(
            provider,
            config=MultiSampleFusionConfig(enabled=True, n_candidates=3, uncertainty_threshold=0.5),
        )
        result = await adapter.areason(
            UserTurn(session_id="s2", user_text="oi"),
            _packet(uncertainty=0.9, temperature=0.3),
            MemoryContext(recent_episodes=[], semantic_rules=[], knowledge_triples=[]),
        )
        self.assertEqual(provider.temperatures, [0.3])
        self.assertEqual(result.telemetry["fusion_method"], "passthrough")
