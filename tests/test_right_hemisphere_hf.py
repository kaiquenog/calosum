from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch
from calosum.adapters.right_hemisphere_hf import HuggingFaceRightHemisphereAdapter, HuggingFaceRightHemisphereConfig
from calosum.shared.types import UserTurn, Modality, MultimodalSignal

class TestHuggingFaceRightHemisphere(unittest.TestCase):
    
    @patch("sentence_transformers.SentenceTransformer")
    def test_perceive_returns_state_with_embeddings(self, mock_sentence_transformer) -> None:
        # Mock the embedder so we don't download models during tests
        mock_embedder_instance = MagicMock()
        mock_embedder_instance.encode.side_effect = lambda x: [[0.1] * 384] if isinstance(x, list) else [0.1] * 384
        mock_sentence_transformer.return_value = mock_embedder_instance
        
        adapter = HuggingFaceRightHemisphereAdapter(HuggingFaceRightHemisphereConfig(latent_size=384))
        
        turn = UserTurn(
            session_id="test",
            user_text="Preciso de ajuda urgente",
            signals=[
                MultimodalSignal(modality=Modality.TEXT, source="user", payload={"text": "urgente"})
            ]
        )
        
        state = adapter.perceive(turn)
        
        self.assertEqual(len(state.latent_vector), 384)
        self.assertIn("urgente", state.emotional_labels)
        self.assertGreaterEqual(state.salience, 1.0)
        self.assertEqual(state.telemetry["vector_dimension"], 384)
        self.assertEqual(adapter.status, "healthy")

    def test_adapter_initialization_fails_gracefully_when_import_fails(self) -> None:
        import sys
        
        # Hide sentence_transformers from sys.modules
        real_module = sys.modules.get('sentence_transformers')
        sys.modules['sentence_transformers'] = None  # type: ignore
        
        try:
            with self.assertRaises(RuntimeError) as ctx:
                HuggingFaceRightHemisphereAdapter()
                
            self.assertIn("missing optional model stack", str(ctx.exception))
        finally:
            # Restore module
            if real_module is not None:
                sys.modules['sentence_transformers'] = real_module
            else:
                del sys.modules['sentence_transformers']

if __name__ == "__main__":
    unittest.main()
