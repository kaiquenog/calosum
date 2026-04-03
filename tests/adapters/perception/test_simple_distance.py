from __future__ import annotations

import unittest
from datetime import datetime, timezone
import numpy as np

from calosum.adapters.perception.simple_distance import SimpleDistanceSurpriseAdapter
from calosum.shared.models.types import MemoryContext, MemoryEpisode, InputPerceptionState, UserTurn


class _StaticRightHemisphere:
    def __init__(self, latent_vector: list[float], surprise_score: float = 0.5) -> None:
        self._latent_vector = latent_vector
        self._surprise_score = surprise_score

    def perceive(self, user_turn: UserTurn, memory_context: MemoryContext | None = None) -> InputPerceptionState:
        return InputPerceptionState(
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
    right_state = InputPerceptionState(
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


class SimpleDistanceAdapterTests(unittest.TestCase):
    def test_distance_increases_surprise_for_novel_vectors(self) -> None:
        familiar_memory = MemoryContext(
            recent_episodes=[
                _episode("familiar", [1.0, 0.0, 0.0]),
                _episode("familiar", [1.0, 0.0, 0.0]),
            ]
        )

        adapter = SimpleDistanceSurpriseAdapter(_StaticRightHemisphere([1.0, 0.0, 0.0]))
        
        # Primeiro turno inicializa EMA
        adapter.perceive(UserTurn(session_id="s", user_text="familiar"), familiar_memory)
        
        # Segundo turno familiar
        familiar = adapter.perceive(UserTurn(session_id="s", user_text="familiar"), familiar_memory)
        
        # Terceiro turno novel - mudamos o vetor base do adapter mockado
        adapter.base_adapter = _StaticRightHemisphere([0.0, 1.0, 0.0])
        novel = adapter.perceive(UserTurn(session_id="s", user_text="novel"), familiar_memory)
        
        # Distância de cosseno entre [1,0,0] e [1,0,0] é 0
        # Distância de cosseno entre [1,0,0] e [-1,0,0] (se usarmos um vetor bem diferente)
        
        # Vamos usar um vetor ortogonal para garantir surpresa
        novel_state = adapter.perceive(
            UserTurn(session_id="s", user_text="novel"),
            familiar_memory
        )
        
        # Como o EMA é atualizado a cada turno, a comparação direta depende da ordem.
        # Mas novel deve ser > familiar em termos de surpresa se o EMA estiver estabilizado no familiar.
        
        self.assertIn("simple_distance_ema", novel.telemetry["surprise_backend"])
        self.assertEqual(familiar.surprise_score, 0.0)
        self.assertGreater(novel.surprise_score, 0.0)

    def test_ema_update_logic(self) -> None:
        adapter = SimpleDistanceSurpriseAdapter(_StaticRightHemisphere([1.0, 0.0, 0.0]))
        
        # Turno 1: EMA = [1, 0, 0]
        adapter.perceive(UserTurn(session_id="s", user_text="1"))
        self.assertTrue(np.allclose(adapter._ema_latent, [1.0, 0.0, 0.0]))
        
        # Turno 2: Percepção = [0, 1, 0], Alpha = 0.2
        # Novo EMA = 0.8 * [1, 0, 0] + 0.2 * [0, 1, 0] = [0.8, 0.2, 0]
        adapter.base_adapter = _StaticRightHemisphere([0.0, 1.0, 0.0])
        adapter.perceive(UserTurn(session_id="s", user_text="2"))
        self.assertTrue(np.allclose(adapter._ema_latent, [0.8, 0.2, 0.0]))


if __name__ == "__main__":
    unittest.main()
