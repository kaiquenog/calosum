from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from calosum.adapters.memory_qdrant import QdrantAdapterConfig, QdrantDualMemoryAdapter
from calosum.shared.types import (
    BridgeControlSignal,
    CognitiveBridgePacket,
    LeftHemisphereResult,
    MemoryEpisode,
    RightHemisphereState,
    TypedLambdaProgram,
    UserTurn,
    utc_now,
)


class _FakeQdrantState:
    collections: dict[str, list[SimpleNamespace]] = {}


class FakeQdrantClient:
    def __init__(self, url: str) -> None:
        self.url = url

    def collection_exists(self, name: str) -> bool:
        return name in _FakeQdrantState.collections

    def create_collection(self, collection_name: str, vectors_config) -> None:
        _FakeQdrantState.collections.setdefault(collection_name, [])


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


def _episode(session_id: str, text: str, labels: list[str]) -> MemoryEpisode:
    turn = UserTurn(session_id=session_id, user_text=text)
    right_state = RightHemisphereState(
        context_id=turn.turn_id,
        latent_vector=[],
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
        bridge_packet=CognitiveBridgePacket(
            context_id=turn.turn_id,
            soft_prompts=[],
            control=BridgeControlSignal(
                target_temperature=0.25,
                empathy_priority=False,
                system_directives=[],
            ),
            salience=right_state.salience,
        ),
        left_result=LeftHemisphereResult(
            response_text="ok",
            lambda_program=TypedLambdaProgram("Context -> Response", "lambda ctx: respond_text()", "safe"),
            actions=[],
            reasoning_summary=[],
        ),
    )


class QdrantAdapterTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        _FakeQdrantState.collections = {}

    async def test_build_context_prefers_semantically_similar_episode(self) -> None:
        with patch("calosum.adapters.memory_qdrant.QdrantClient", FakeQdrantClient), patch(
            "calosum.adapters.memory_qdrant.AsyncQdrantClient",
            FakeAsyncQdrantClient,
        ):
            adapter = QdrantDualMemoryAdapter(QdrantAdapterConfig(url="http://fake-qdrant"))
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

    async def test_sleep_mode_promotes_rules_that_are_retrieved_by_vector_search(self) -> None:
        with patch("calosum.adapters.memory_qdrant.QdrantClient", FakeQdrantClient), patch(
            "calosum.adapters.memory_qdrant.AsyncQdrantClient",
            FakeAsyncQdrantClient,
        ):
            adapter = QdrantDualMemoryAdapter(QdrantAdapterConfig(url="http://fake-qdrant"))
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


if __name__ == "__main__":
    unittest.main()
