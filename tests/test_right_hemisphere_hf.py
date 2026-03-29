import unittest
from calosum.adapters.right_hemisphere_hf import HuggingFaceRightHemisphereAdapter, HuggingFaceRightHemisphereConfig
from calosum.shared.types import UserTurn

class RightHemisphereTests(unittest.TestCase):
    def setUp(self):
        # Utiliza um modelo menor ou mock se possível para o teste ser rápido,
        # mas como é test-driven local, vamos instanciar com o config default.
        self.config = HuggingFaceRightHemisphereConfig()
        self.adapter = HuggingFaceRightHemisphereAdapter(self.config)

    def test_semantic_density_calculation(self):
        turn = UserTurn(turn_id="test-1", session_id="sess-1", user_text="Uma frase bem complexa com várias emoções como desespero e ansiedade misturadas!", signals=[])
        state = self.adapter.perceive(turn)
        
        self.assertIsNotNone(state)
        self.assertIn("semantic_density", state.world_hypotheses)
        self.assertTrue(0.0 <= state.world_hypotheses["semantic_density"] <= 1.0)
        self.assertTrue(len(state.emotional_labels) > 0)
        self.assertTrue(state.salience > 0.0)

if __name__ == "__main__":
    unittest.main()
