from __future__ import annotations

import unittest
import math
from calosum import UserTurn
from calosum.adapters.hemisphere.input_perception_heuristic_jepa import HeuristicJEPAAdapter

class HonestJEPATests(unittest.TestCase):
    def test_identical_phrases_have_zero_distance(self) -> None:
        adapter = HeuristicJEPAAdapter()
        turn1 = UserTurn(session_id="test", user_text="Olá, mundo!")
        turn2 = UserTurn(session_id="test", user_text="Olá, mundo!")
        
        state1 = adapter.perceive(turn1)
        state2 = adapter.perceive(turn2)
        
        v1 = state1.latent_vector
        v2 = state2.latent_vector
        
        # Distância cosseno = 1 - similaridade
        dot = sum(a * b for a, b in zip(v1, v2))
        mag1 = math.sqrt(sum(a * a for a in v1))
        mag2 = math.sqrt(sum(a * a for a in v2))
        similarity = dot / (mag1 * mag2)
        
        self.assertAlmostEqual(similarity, 1.0, places=5)
        self.assertAlmostEqual(state1.surprise_score, state2.surprise_score, places=5)

    def test_different_phrases_have_non_zero_distance(self) -> None:
        adapter = HeuristicJEPAAdapter()
        turn1 = UserTurn(session_id="test", user_text="Estou muito feliz hoje!")
        turn2 = UserTurn(session_id="test", user_text="O sistema está apresentando erros críticos.")
        
        state1 = adapter.perceive(turn1)
        state2 = adapter.perceive(turn2)
        
        v1 = state1.latent_vector
        v2 = state2.latent_vector
        
        dot = sum(a * b for a, b in zip(v1, v2))
        mag1 = math.sqrt(sum(a * a for a in v1))
        mag2 = math.sqrt(sum(a * a for a in v2))
        similarity = dot / (mag1 * mag2)
        
        self.assertLess(similarity, 0.9)
        self.assertNotEqual(state1.salience, state2.salience)

if __name__ == "__main__":
    unittest.main()
