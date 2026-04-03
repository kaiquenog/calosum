from __future__ import annotations

import base64
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from calosum.adapters.memory.memory_qdrant import QdrantAdapterConfig, QdrantDualMemoryAdapter
from calosum.adapters.memory.memory_qdrant_serializers import (
    episode_from_point,
    episode_payload,
)
from calosum.shared.models.types import (
    BridgeControlSignal,
    PerceptionSummary,
    ActionPlannerResult,
    MemoryEpisode,
    InputPerceptionState,
    TypedLambdaProgram,
    UserTurn,
    utc_now,
)


class _FakeQdrantState:
    collections: dict[str, list[SimpleNamespace]] = {}
    created_configs: dict[str, object] = {}


class FakeQdrantClient:
    def __init__(self, url: str) -> None:
        self.url = url

    def collection_exists(self, name: str) -> bool:
        return name in _FakeQdrantState.collections

    def create_collection(self, collection_name: str, vectors_config, **kwargs) -> None:
        _FakeQdrantState.collections.setdefault(collection_name, [])
        _FakeQdrantState.created_configs[collection_name] = kwargs.get("quantization_config")


class FakeAsyncQdrantClient:
    def __init__(self, url: str) -> None:
        self.url = url

    async def upsert(self, collection_name: str, points) -> None:
        items = _FakeQdrantState.collections.setdefault(collection_name, [])
        for point in points:
            items.append(
                SimpleNamespace(
                    id=point.id,
                    payload=point.payload,
                    vector=list(point.vector),
                )
            )

    async def search(self, collection_name: str, query_vector, limit: int, with_payload: bool):
        items = _FakeQdrantState.collections.get(collection_name, [])
        ranked = sorted(
            items,
            key=lambda item: sum(a * b for a, b in zip(query_vector, item.vector)),
            reverse=True,
        )
        return ranked[:limit]

    async def scroll(self, collection_name: str, limit: int, with_payload: bool):
        return _FakeQdrantState.collections.get(collection_name, [])[:limit], None


class FakeEmbedder:
    async def aembed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    1.0 if "projeto" in lowered else 0.0,
                    1.0 if "urgente" in lowered else 0.0,
                    1.0 if "bolo" in lowered else 0.0,
                    float(len(lowered.split())) / 10.0,
                ]
            )
        return vectors


def _episode(session_id: str, text: str, labels: list[str]) -> MemoryEpisode:
    turn = UserTurn(session_id=session_id, user_text=text)
    right_state = InputPerceptionState(
        context_id=turn.turn_id,
        latent_vector=[
            1.0 if "projeto" in text.lower() else 0.0,
            1.0 if "urgente" in text.lower() else 0.0,
            1.0 if "bolo" in text.lower() else 0.0,
            float(len(text.split())) / 10.0,
        ],
        salience=0.8 if "urgente" in text.lower() else 0.2,
        emotional_labels=labels,
        world_hypotheses={},
        confidence=0.7,
    )
    return MemoryEpisode(
        episode_id=f"episode-{turn.turn_id}",
        recorded_at=utc_now(),
        user_turn=turn,
        right_state=right_state,
        bridge_packet=PerceptionSummary(
            context_id=turn.turn_id,
            soft_prompts=[],
            control=BridgeControlSignal(
                target_temperature=0.25,
                empathy_priority=False,
                system_directives=[],
            ),
            salience=right_state.salience,
        ),
        left_result=ActionPlannerResult(
            response_text="ok",
            lambda_program=TypedLambdaProgram("Context -> Response", "lambda ctx: respond_text()", "safe"),
            actions=[],
            reasoning_summary=[],
        ),
    )


class QdrantAdapterTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        _FakeQdrantState.collections = {}
        _FakeQdrantState.created_configs = {}

    async def test_build_context_prefers_semantically_similar_episode(self) -> None:
        with patch("calosum.adapters.memory.memory_qdrant.QdrantClient", FakeQdrantClient), patch(
            "calosum.adapters.memory.memory_qdrant.AsyncQdrantClient",
            FakeAsyncQdrantClient,
        ):
            adapter = QdrantDualMemoryAdapter(
                QdrantAdapterConfig(url="http://fake-qdrant"),
                embedder=FakeEmbedder(),
            )
            await adapter.astore_episode(
                _episode("qdrant-session", "Preciso de um plano urgente para reorganizar o projeto.", ["urgente"])
            )
            await adapter.astore_episode(
                _episode("qdrant-session", "Quero uma receita simples de bolo.", ["neutral"])
            )

            context = await adapter.abuild_context(
                UserTurn(session_id="qdrant-session", user_text="Me ajude com um projeto urgente.")
            )

        self.assertGreaterEqual(len(context.recent_episodes), 1)
        self.assertIn("projeto", context.recent_episodes[0].user_turn.user_text.lower())
        self.assertTrue(context.recent_episodes[0].right_state.latent_vector)

    async def test_sleep_mode_promotes_rules_that_are_retrieved_by_vector_search(self) -> None:
        with patch("calosum.adapters.memory.memory_qdrant.QdrantClient", FakeQdrantClient), patch(
            "calosum.adapters.memory.memory_qdrant.AsyncQdrantClient",
            FakeAsyncQdrantClient,
        ):
            adapter = QdrantDualMemoryAdapter(
                QdrantAdapterConfig(url="http://fake-qdrant"),
                embedder=FakeEmbedder(),
            )
            await adapter.astore_episode(
                _episode("qdrant-session", "Estou urgente e preciso de ajuda.", ["urgente"])
            )
            await adapter.astore_episode(
                _episode("qdrant-session", "Ainda estou urgente e ansioso.", ["urgente", "ansioso"])
            )

            await adapter.asleep_mode()
            context = await adapter.abuild_context(
                UserTurn(session_id="qdrant-session", user_text="Minha situacao esta urgente.")
            )

        self.assertTrue(any(rule.rule_id == "emotion::urgente" for rule in context.semantic_rules))
        self.assertTrue(any(triple.predicate == "biases_response_toward" for triple in context.knowledge_triples))

    async def test_build_context_scopes_episodes_to_same_session_when_available(self) -> None:
        with patch("calosum.adapters.memory.memory_qdrant.QdrantClient", FakeQdrantClient), patch(
            "calosum.adapters.memory.memory_qdrant.AsyncQdrantClient",
            FakeAsyncQdrantClient,
        ):
            adapter = QdrantDualMemoryAdapter(
                QdrantAdapterConfig(url="http://fake-qdrant"),
                embedder=FakeEmbedder(),
            )
            await adapter.astore_episode(
                _episode("session-a", "Projeto urgente com escopo grande.", ["urgente"])
            )
            await adapter.astore_episode(
                _episode("session-b", "Projeto urgente com riscos similares.", ["urgente"])
            )

            context = await adapter.abuild_context(
                UserTurn(session_id="session-a", user_text="Preciso de ajuda com esse projeto urgente.")
            )

        self.assertGreaterEqual(len(context.recent_episodes), 1)
        self.assertTrue(
            all(episode.user_turn.session_id == "session-a" for episode in context.recent_episodes)
        )

    # -------------------------------------------------------------------
    # Sprint 1.5 — serializers roundtrip (no regression after extraction)
    # -------------------------------------------------------------------

    def test_serializers_roundtrip(self) -> None:
        """episode_from_point(point_from_episode(ep)) preserves critical fields."""
        ep = _episode("roundtrip-session", "Texto de teste para serializer.", ["neutral"])
        payload = episode_payload(ep)
        point = SimpleNamespace(id="fake-id-123", payload=payload)
        recovered = episode_from_point(point)
        self.assertEqual(recovered.user_turn.user_text, ep.user_turn.user_text)
        self.assertEqual(recovered.user_turn.session_id, ep.user_turn.session_id)
        self.assertEqual(recovered.right_state.emotional_labels, ep.right_state.emotional_labels)

    def test_no_regression_after_extraction(self) -> None:
        """episode_payload returns non-empty dict for a valid episode."""
        ep = _episode("reg-session", "Regression test.", ["neutral"])
        payload = episode_payload(ep)
        self.assertIn("text", payload)
        self.assertIn("session", payload)
        self.assertIn("emotional_labels", payload)
        self.assertIn("latent_vector", payload)

    # -------------------------------------------------------------------
    # Sprint 2 — store_with_codec_payload
    # -------------------------------------------------------------------

    async def test_store_with_codec_payload(self) -> None:
        """When codec is set, astore_episode adds latent_vector_compressed to payload."""
        from calosum.adapters.perception.quantized_embeddings import TurboQuantVectorCodec

        codec = TurboQuantVectorCodec(bits=3)

        with patch("calosum.adapters.memory.memory_qdrant.QdrantClient", FakeQdrantClient), patch(
            "calosum.adapters.memory.memory_qdrant.AsyncQdrantClient", FakeAsyncQdrantClient,
        ):
            adapter = QdrantDualMemoryAdapter(
                QdrantAdapterConfig(url="http://fake-qdrant"),
                embedder=FakeEmbedder(),
                codec=codec,
            )
            ep = _episode("codec-session", "Projeto urgente com codec.", ["urgente"])
            await adapter.astore_episode(ep)

        stored = _FakeQdrantState.collections["episodes"][0]
        self.assertIn("latent_vector_compressed", stored.payload)
        compressed_b64 = stored.payload["latent_vector_compressed"]
        compressed = base64.b64decode(compressed_b64)
        self.assertIsInstance(compressed, bytes)
        self.assertGreater(len(compressed), 0)

    # -------------------------------------------------------------------
    # Sprint 0 — scalar quantization flag
    # -------------------------------------------------------------------

    def test_qdrant_scalar_quantization_flag(self) -> None:
        """When scalar_quantization=True, create_collection is called with a quantization_config."""
        with patch("calosum.adapters.memory.memory_qdrant.QdrantClient", FakeQdrantClient), patch(
            "calosum.adapters.memory.memory_qdrant.AsyncQdrantClient", FakeAsyncQdrantClient,
        ):
            QdrantDualMemoryAdapter(
                QdrantAdapterConfig(url="http://fake-qdrant", scalar_quantization=True),
                embedder=FakeEmbedder(),
            )

        for col in ["episodes", "semantic_rules"]:
            self.assertIsNotNone(
                _FakeQdrantState.created_configs.get(col),
                f"quantization_config not set for collection {col}",
            )


if __name__ == "__main__":
    unittest.main()
