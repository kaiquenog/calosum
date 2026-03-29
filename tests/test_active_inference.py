from __future__ import annotations

import sys
import types
import unittest
from datetime import datetime, timezone

import numpy as np

from calosum.adapters.active_inference import ActiveInferenceRightHemisphereAdapter
from calosum.shared.types import MemoryContext, MemoryEpisode, RightHemisphereState, UserTurn


class _StaticRightHemisphere:
    def __init__(self, latent_vector: list[float], surprise_score: float = 0.5) -> None:
        self._latent_vector = latent_vector
        self._surprise_score = surprise_score

    def perceive(self, user_turn: UserTurn, memory_context: MemoryContext | None = None) -> RightHemisphereState:
        return RightHemisphereState(
            context_id=user_turn.turn_id,
            latent_vector=list(self._latent_vector),
            salience=0.4,
            emotional_labels=["neutral"],
            world_hypotheses={},
            confidence=0.8,
            surprise_score=self._surprise_score,
            telemetry={"source": "static_test"},
        )


def _episode(text: str, latent_vector: list[float]) -> MemoryEpisode:
    turn = UserTurn(session_id="memory", user_text=text, observed_at=datetime.now(timezone.utc))
    right_state = RightHemisphereState(
        context_id=turn.turn_id,
        latent_vector=list(latent_vector),
        salience=0.2,
        emotional_labels=["neutral"],
        world_hypotheses={},
        confidence=0.7,
        surprise_score=0.2,
    )
    return MemoryEpisode(
        episode_id=f"episode-{turn.turn_id}",
        recorded_at=datetime.now(timezone.utc),
        user_turn=turn,
        right_state=right_state,
        bridge_packet=None,  # type: ignore[arg-type]
        left_result=None,  # type: ignore[arg-type]
    )


class ActiveInferenceAdapterTests(unittest.TestCase):
    def test_active_inference_increases_surprise_for_novel_vectors(self) -> None:
        familiar_memory = MemoryContext(
            recent_episodes=[
                _episode("familiar", [1.0, 0.0, 0.0]),
                _episode("familiar", [0.98, 0.02, 0.0]),
                _episode("familiar", [0.97, 0.03, 0.0]),
            ]
        )

        familiar = ActiveInferenceRightHemisphereAdapter(
            _StaticRightHemisphere([1.0, 0.0, 0.0])
        ).perceive(UserTurn(session_id="s", user_text="familiar"), familiar_memory)
        novel = ActiveInferenceRightHemisphereAdapter(
            _StaticRightHemisphere([-1.0, 0.0, 0.0])
        ).perceive(UserTurn(session_id="s", user_text="novel"), familiar_memory)

        self.assertLess(familiar.surprise_score, novel.surprise_score)
        self.assertIn("active_inference", novel.telemetry["surprise_backend"])

    def test_active_inference_uses_pymdp_math_when_available(self) -> None:
        fake_pymdp = types.ModuleType("pymdp")
        fake_maths = types.ModuleType("pymdp.maths")

        def softmax(values):
            arr = np.asarray(values, dtype=float)
            shifted = arr - np.max(arr)
            exps = np.exp(shifted)
            return exps / np.sum(exps)

        def log_stable(values):
            arr = np.asarray(values, dtype=float)
            return np.log(np.clip(arr, 1e-9, 1.0))

        fake_maths.softmax = softmax
        fake_maths.spm_log_single = log_stable
        sys.modules["pymdp"] = fake_pymdp
        sys.modules["pymdp.maths"] = fake_maths

        try:
            state = ActiveInferenceRightHemisphereAdapter(
                _StaticRightHemisphere([1.0, 0.0, 0.0])
            ).perceive(
                UserTurn(session_id="s", user_text="familiar"),
                MemoryContext(recent_episodes=[_episode("familiar", [1.0, 0.0, 0.0])]),
            )
        finally:
            sys.modules.pop("pymdp.maths", None)
            sys.modules.pop("pymdp", None)

        self.assertEqual(state.telemetry["surprise_engine"], "pymdp_vfe")


if __name__ == "__main__":
    unittest.main()
