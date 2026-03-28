from __future__ import annotations

import hashlib
import math
import re
import uuid
from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from calosum.shared.async_utils import run_sync
from calosum.shared.types import (
    BridgeControlSignal,
    ConsolidationReport,
    CognitiveBridgePacket,
    KnowledgeTriple,
    LeftHemisphereResult,
    MemoryContext,
    MemoryEpisode,
    RightHemisphereState,
    SemanticRule,
    TypedLambdaProgram,
    UserTurn,
    utc_now,
)


@dataclass(slots=True)
class QdrantAdapterConfig:
    url: str = "http://localhost:6333"
    episodes_collection: str = "episodes"
    rules_collection: str = "semantic_rules"
    vector_size: int = 384


class QdrantDualMemoryAdapter:
    """
    Implementação concreta do MemorySystemPort via Qdrant para vetores.
    Como é um substituto de PersistentDualMemory, ele abstrai a persistência
    diretamente no banco de dados vetorial.
    """

    def __init__(self, config: QdrantAdapterConfig | None = None) -> None:
        self.config = config or QdrantAdapterConfig()
        self.client = QdrantClient(url=self.config.url)
        self.aclient = AsyncQdrantClient(url=self.config.url)
        self._ensure_collections()

    def _ensure_collections(self):
        # Setup blocks synchronous during init.
        for col_name in [self.config.episodes_collection, self.config.rules_collection]:
            if not self.client.collection_exists(col_name):
                self.client.create_collection(
                    collection_name=col_name,
                    vectors_config=VectorParams(
                        size=self.config.vector_size,
                        distance=Distance.COSINE,
                    ),
                )

    def build_context(self, user_turn: UserTurn, episodic_limit: int = 5) -> MemoryContext:
        return run_sync(self.abuild_context(user_turn, episodic_limit))

    async def abuild_context(self, user_turn: UserTurn, episodic_limit: int = 5) -> MemoryContext:
        query_vector = self._vector_for_text(user_turn.user_text)
        res = await self._asearch_points(
            self.config.episodes_collection,
            query_vector,
            episodic_limit,
        )
        
        episodes = []
        for point in res:
            p = point.payload or {}
            if "text" in p:
                episode_turn = UserTurn(
                    session_id=p.get("session", "*"),
                    user_text=p["text"],
                    signals=[],
                )
                right_state = _placeholder_right_state(
                    episode_turn,
                    emotional_labels=list(p.get("emotional_labels", [])),
                )
                # Reconstruindo mock do episodio p/ retornar contexto
                episodes.append(
                    MemoryEpisode(
                        episode_id=str(point.id),
                        recorded_at=utc_now(),
                        user_turn=episode_turn,
                        right_state=right_state,
                        bridge_packet=_placeholder_bridge_packet(right_state),
                        left_result=_placeholder_left_result(),
                    )
                )

        rules_res = await self._asearch_points(
            self.config.rules_collection,
            query_vector,
            10,
        )
        
        rules = [
            SemanticRule(
                rule_id=r.payload.get("rule_id", str(r.id)),
                statement=r.payload.get("statement", ""),
                strength=float(r.payload.get("strength", 1.0)),
                supporting_episodes=list(r.payload.get("supporting_episodes", [])),
                tags=list(r.payload.get("tags", [])),
            )
            for r in rules_res
        ]
        if not rules:
            rules = [SemanticRule(rule_id="default", statement="Mantenha respostas gentis.", strength=1.0, supporting_episodes=[], tags=[])]

        return MemoryContext(
            recent_episodes=episodes,
            semantic_rules=rules,
            knowledge_triples=[KnowledgeTriple(subject="user", predicate="is", object="human")],
        )

    def store_episode(self, episode: MemoryEpisode) -> None:
        run_sync(self.astore_episode(episode))

    async def astore_episode(self, episode: MemoryEpisode) -> None:
        point_id = str(uuid.uuid4())
        vector = self._vector_for_episode(episode)
        
        await self.aclient.upsert(
            collection_name=self.config.episodes_collection,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "text": episode.user_turn.user_text, 
                        "session": episode.user_turn.session_id,
                        "emotional_labels": episode.right_state.emotional_labels if episode.right_state else [],
                    },
                )
            ],
        )

    def sleep_mode(self) -> ConsolidationReport:
        return run_sync(self.asleep_mode())

    async def asleep_mode(self) -> ConsolidationReport:
        from calosum.domain.memory import SleepModeConsolidator
        
        # 1. Fetch episodes
        res, _ = await self.aclient.scroll(
            collection_name=self.config.episodes_collection,
            limit=100,
            with_payload=True,
        )
        
        episodes = []
        for point in res:
            p = point.payload or {}
            if "text" in p:
                episode_turn = UserTurn(
                    session_id=p.get("session", "*"),
                    user_text=p["text"],
                    signals=[],
                )
                right_state = _placeholder_right_state(
                    episode_turn,
                    emotional_labels=list(p.get("emotional_labels", [])),
                    salience=0.5,
                )
                episodes.append(
                    MemoryEpisode(
                        episode_id=str(point.id),
                        recorded_at=utc_now(),
                        user_turn=episode_turn,
                        right_state=right_state,
                        bridge_packet=_placeholder_bridge_packet(right_state),
                        left_result=_placeholder_left_result(),
                    )
                )

        # 2. Consolidate
        consolidator = SleepModeConsolidator(minimum_frequency=1) # Set to 1 for easier demo/testing
        report = consolidator.consolidate(episodes)

        # 3. Store Promoted Rules into Qdrant semantic_rules collection
        if report.promoted_rules:
            points = []
            for rule in report.promoted_rules:
                points.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=self._vector_for_text(rule.statement),
                        payload={
                            "rule_id": rule.rule_id,
                            "statement": rule.statement,
                            "strength": rule.strength,
                            "tags": rule.tags,
                            "supporting_episodes": rule.supporting_episodes
                        }
                    )
                )
            await self.aclient.upsert(
                collection_name=self.config.rules_collection,
                points=points
            )

        return report

    async def _asearch_points(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int,
    ) -> list:
        if hasattr(self.aclient, "search"):
            result = await self.aclient.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                with_payload=True,
            )
            return list(result)

        if hasattr(self.aclient, "query_points"):
            result = await self.aclient.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=limit,
                with_payload=True,
            )
            points = getattr(result, "points", None)
            if points is not None:
                return list(points)

        result, _ = await self.aclient.scroll(
            collection_name=collection_name,
            limit=limit,
            with_payload=True,
        )
        return list(result)

    def _vector_for_episode(self, episode: MemoryEpisode) -> list[float]:
        text = episode.user_turn.user_text
        emotional_labels = " ".join(episode.right_state.emotional_labels)
        return self._vector_for_text(f"{text} {emotional_labels}".strip())

    def _vector_for_text(self, text: str) -> list[float]:
        normalized = text.strip().lower() or "silence"
        tokens = re.findall(r"\w+", normalized) or ["silence"]
        vector = [0.0] * self.config.vector_size

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for index in range(0, len(digest), 4):
                chunk = digest[index:index + 4]
                position = int.from_bytes(chunk[:2], "big") % self.config.vector_size
                sign = 1.0 if chunk[2] % 2 == 0 else -1.0
                magnitude = 1.0 + (chunk[3] / 255.0)
                vector[position] += sign * magnitude

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [round(value / norm, 6) for value in vector]


def _placeholder_right_state(
    user_turn: UserTurn,
    *,
    emotional_labels: list[str] | None = None,
    salience: float = 0.15,
) -> RightHemisphereState:
    return RightHemisphereState(
        context_id=user_turn.turn_id,
        latent_vector=[],
        salience=salience,
        emotional_labels=emotional_labels or ["neutral"],
        world_hypotheses={},
        confidence=0.0,
        telemetry={"source": "qdrant_placeholder"},
    )


def _placeholder_bridge_packet(right_state: RightHemisphereState) -> CognitiveBridgePacket:
    return CognitiveBridgePacket(
        context_id=right_state.context_id,
        soft_prompts=[],
        control=BridgeControlSignal(
            target_temperature=0.25,
            empathy_priority=right_state.salience >= 0.7,
            system_directives=[],
            annotations={"source": "qdrant_placeholder"},
        ),
        salience=right_state.salience,
        bridge_metadata={"source": "qdrant_placeholder"},
    )


def _placeholder_left_result() -> LeftHemisphereResult:
    return LeftHemisphereResult(
        response_text="",
        lambda_program=TypedLambdaProgram(
            signature="Placeholder",
            expression="",
            expected_effect="hydrate memory episode without executable actions",
        ),
        actions=[],
        reasoning_summary=["placeholder_memory_episode"],
        telemetry={"source": "qdrant_placeholder"},
    )
