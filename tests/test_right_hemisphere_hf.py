from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch
import types
from calosum.adapters.right_hemisphere_hf import HuggingFaceRightHemisphereAdapter, HuggingFaceRightHemisphereConfig
from calosum.shared.types import UserTurn, Modality, MultimodalSignal

class TestHuggingFaceRightHemisphere(unittest.TestCase):
    def test_perceive_returns_state_with_embeddings(self) -> None:
        # Mock dynamic imports from __init__ so we don't require optional deps.
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.logging = types.SimpleNamespace(set_verbosity_error=lambda: None)
        fake_transformers.utils = types.SimpleNamespace(
            logging=types.SimpleNamespace(disable_progress_bar=lambda: None)
        )
        fake_sentence_transformers = types.ModuleType("sentence_transformers")
        mock_embedder_instance = MagicMock()
        mock_embedder_instance.encode.side_effect = lambda x: [[0.1] * 384] if isinstance(x, list) else [0.1] * 384
        fake_sentence_transformers.SentenceTransformer = MagicMock(return_value=mock_embedder_instance)

        with patch.dict(
            "sys.modules",
            {"transformers": fake_transformers, "sentence_transformers": fake_sentence_transformers},
        ):
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
        self.assertEqual(state.telemetry["right_backend"], "huggingface_sentence_transformers")
        self.assertEqual(state.telemetry["right_model_name"], adapter.config.embedding_model_name)
        self.assertEqual(state.telemetry["right_mode"], "embedding")
        self.assertIsNone(state.telemetry["degraded_reason"])
        self.assertIn("emotion_keyword_hits", state.telemetry)
        self.assertIn("emotion_vector_hits", state.telemetry)
        self.assertIn("emotion_peak_similarity", state.telemetry)
        self.assertIn("raw_salience", state.telemetry)
        self.assertGreaterEqual(state.confidence, 0.55)
        self.assertEqual(adapter.status, "healthy")

    def test_perceive_reduces_false_positive_on_neutral_text(self) -> None:
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.logging = types.SimpleNamespace(set_verbosity_error=lambda: None)
        fake_transformers.utils = types.SimpleNamespace(
            logging=types.SimpleNamespace(disable_progress_bar=lambda: None)
        )
        fake_sentence_transformers = types.ModuleType("sentence_transformers")

        def fake_encode(values):
            if isinstance(values, list) and values and values[0] in {
                "urgente", "emergencia", "triste", "ansioso", "feliz",
                "frustrado", "raiva", "medo", "preocupado", "dor", "desespero"
            }:
                size = len(values)
                return [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]
            # Texto neutro vira vetor aproximadamente uniforme, reduzindo similaridade por label.
            return [[1.0] * 11]

        mock_embedder_instance = MagicMock()
        mock_embedder_instance.encode.side_effect = fake_encode
        fake_sentence_transformers.SentenceTransformer = MagicMock(return_value=mock_embedder_instance)

        with patch.dict(
            "sys.modules",
            {"transformers": fake_transformers, "sentence_transformers": fake_sentence_transformers},
        ):
            adapter = HuggingFaceRightHemisphereAdapter(HuggingFaceRightHemisphereConfig(latent_size=11))

        turn = UserTurn(session_id="test", user_text="Hoje foi um dia comum, sem urgencia.")
        state = adapter.perceive(turn)

        # Deve evitar falso positivo vetorial em texto neutro.
        self.assertEqual(state.emotional_labels, ["neutral"])
        self.assertLess(state.salience, 0.5)
        self.assertEqual(state.telemetry["emotion_vector_hits"], 0)
        self.assertEqual(state.telemetry["emotion_keyword_hits"], 0)

    def test_salience_is_temporally_calibrated_per_session(self) -> None:
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.logging = types.SimpleNamespace(set_verbosity_error=lambda: None)
        fake_transformers.utils = types.SimpleNamespace(
            logging=types.SimpleNamespace(disable_progress_bar=lambda: None)
        )
        fake_sentence_transformers = types.ModuleType("sentence_transformers")

        def fake_encode(values):
            # Distinct vectors for neutral vs urgent turns.
            if isinstance(values, list) and values and values[0] in {
                "urgente", "emergencia", "triste", "ansioso", "feliz",
                "frustrado", "raiva", "medo", "preocupado", "dor", "desespero"
            }:
                size = len(values)
                return [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]
            text = values[0]
            return [[1.0] * 11] if "urgente" in text else [[0.01] * 11]

        mock_embedder_instance = MagicMock()
        mock_embedder_instance.encode.side_effect = fake_encode
        fake_sentence_transformers.SentenceTransformer = MagicMock(return_value=mock_embedder_instance)

        with patch.dict(
            "sys.modules",
            {"transformers": fake_transformers, "sentence_transformers": fake_sentence_transformers},
        ):
            adapter = HuggingFaceRightHemisphereAdapter(HuggingFaceRightHemisphereConfig(latent_size=11))

        session_id = "salience-session"
        high_state = adapter.perceive(UserTurn(session_id=session_id, user_text="Estou urgente!"))
        low_state = adapter.perceive(UserTurn(session_id=session_id, user_text="Mensagem neutra sem sinal afetivo"))

        # Raw salience da segunda mensagem deve ser menor que a calibrada por janela.
        self.assertLess(low_state.telemetry["raw_salience"], high_state.salience)
        self.assertGreater(low_state.salience, low_state.telemetry["raw_salience"])
        self.assertLessEqual(abs(low_state.salience - high_state.salience), adapter.config.salience_max_step)

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

    def test_surprise_with_codec(self) -> None:
        """When codec is set, _calculate_surprise uses inner_product_approx."""
        import types
        from unittest.mock import MagicMock
        from calosum.adapters.quantized_embeddings import TurboQuantVectorCodec

        fake_transformers = types.ModuleType("transformers")
        fake_transformers.logging = types.SimpleNamespace(set_verbosity_error=lambda: None)
        fake_transformers.utils = types.SimpleNamespace(
            logging=types.SimpleNamespace(disable_progress_bar=lambda: None)
        )
        fake_sentence_transformers = types.ModuleType("sentence_transformers")
        mock_emb = MagicMock()
        mock_emb.encode.side_effect = lambda x: [[0.1] * 8] * (len(x) if isinstance(x, list) else 1)
        fake_sentence_transformers.SentenceTransformer = MagicMock(return_value=mock_emb)

        codec = TurboQuantVectorCodec(bits=3)

        with patch.dict(
            "sys.modules",
            {"transformers": fake_transformers, "sentence_transformers": fake_sentence_transformers},
        ):
            adapter = HuggingFaceRightHemisphereAdapter(
                HuggingFaceRightHemisphereConfig(latent_size=8), codec=codec
            )

        fake_vec = [0.5] * 8

        class _FakeRight:
            latent_vector = fake_vec

        class _FakeEp:
            right_state = _FakeRight()

        class _FakeCtx:
            recent_episodes = [_FakeEp()]

        surprise = adapter._calculate_surprise([0.5] * 8, _FakeCtx())
        self.assertIsInstance(surprise, float)
        self.assertGreaterEqual(surprise, 0.0)
        self.assertLessEqual(surprise, 1.0)

        turn = UserTurn(session_id="codec-test", user_text="test")
        state = adapter.perceive(turn)
        self.assertTrue(state.telemetry.get("codec_used"))


if __name__ == "__main__":
    unittest.main()
