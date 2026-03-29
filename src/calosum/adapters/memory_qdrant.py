from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
import logging

from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from tenacity import retry, stop_after_attempt, wait_exponential

from calosum.adapters.text_embeddings import TextEmbeddingAdapter, TextEmbeddingAdapterConfig
from calosum.shared.async_utils import run_sync
from calosum.shared.types import (
    BridgeControlSignal,
    ConsolidationReport,
    CognitiveBridgePacket,
    KnowledgeTriple,
    LeftHemisphereResult,
    MemoryContext,
    MemoryEpisode,
    PrimitiveAction,
    RightHemisphereState,
    SemanticRule,
    TypedLambdaProgram,
    UserTurn,
    utc_now,
)


from calosum.shared.ports import DatasetExporterPort

logger = logging.getLogger(__name__)

@dataclass(slots=True)
class QdrantAdapterConfig:
    url: str = "http://localhost:6333"
    episodes_collection: str = "episodes"
    rules_collection: str = "semantic_rules"
    vector_size: int = 384


class QdrantDualMemoryAdapter:
    """
    Implementacao concreta do MemorySystemPort via Qdrant.

    O adapter faz indexacao semantica para episodios e regras, mas mantem
    fallback explicito quando o backend de embeddings nao esta disponivel.
    """

    def __init__(
        self,
        config: QdrantAdapterConfig | None = None,
        *,
        embedder: TextEmbeddingAdapter | None = None,
        exporter: DatasetExporterPort | None = None,
    ) -> None:
        self.config = config or QdrantAdapterConfig()
        self.client = QdrantClient(url=self.config.url)
        self.aclient = AsyncQdrantClient(url=self.config.url)
        self.embedder = embedder or TextEmbeddingAdapter(
            TextEmbeddingAdapterConfig(vector_size=self.config.vector_size)
        )
        self.exporter = exporter
        self._ensure_collections()

    def _ensure_collections(self) -> None:
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
        query_vector = await self._embed_text(self._query_text(user_turn))
        episode_points = await self._asearch_points(
            self.config.episodes_collection,
            query_vector,
            episodic_limit,
        )
        rule_points = await self._asearch_points(
            self.config.rules_collection,
            query_vector,
            10,
        )

        episodes = [self._episode_from_point(point) for point in episode_points if getattr(point, "payload", None)]
        rules = [self._rule_from_point(point) for point in rule_points if getattr(point, "payload", None)]
        if not rules:
            rules = [
                SemanticRule(
                    rule_id="default",
                    statement="Mantenha respostas gentis.",
                    strength=1.0,
                    supporting_episodes=[],
                    tags=[],
                )
            ]

        return MemoryContext(
            recent_episodes=episodes,
            semantic_rules=rules,
            knowledge_triples=[KnowledgeTriple(subject="user", predicate="is", object="human")],
        )

    def store_episode(self, episode: MemoryEpisode) -> None:
        run_sync(self.astore_episode(episode))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
    async def astore_episode(self, episode: MemoryEpisode) -> None:
        point_id = str(uuid.uuid4())
        payload = self._episode_payload(episode)
        vector = await self._embed_text(self._episode_document(episode))

        await self.aclient.upsert(
            collection_name=self.config.episodes_collection,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            ],
        )

    def sleep_mode(self) -> ConsolidationReport:
        return run_sync(self.asleep_mode())

    async def asleep_mode(self) -> ConsolidationReport:
        from calosum.domain.memory import SleepModeConsolidator

        points, _ = await self.aclient.scroll(
            collection_name=self.config.episodes_collection,
            limit=100,
            with_payload=True,
        )
        episodes = [self._episode_from_point(point) for point in points if getattr(point, "payload", None)]

        consolidator = SleepModeConsolidator(exporter=self.exporter, minimum_frequency=1)
        report = consolidator.consolidate(episodes)

        if report.promoted_rules:
            vectors = await self.embedder.aembed_texts(
                [self._rule_document(rule) for rule in report.promoted_rules]
            )
            points_to_upsert = []
            for rule, vector in zip(report.promoted_rules, vectors):
                points_to_upsert.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload={
                            "rule_id": rule.rule_id,
                            "statement": rule.statement,
                            "strength": rule.strength,
                            "tags": rule.tags,
                            "supporting_episodes": rule.supporting_episodes,
                        },
                    )
                )
            await self.aclient.upsert(
                collection_name=self.config.rules_collection,
                points=points_to_upsert,
            )

        return report

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
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

        points, _ = await self.aclient.scroll(
            collection_name=collection_name,
            limit=limit,
            with_payload=True,
        )
        return list(points)

    async def _embed_text(self, text: str) -> list[float]:
        return (await self.embedder.aembed_texts([text]))[0]

    def _query_text(self, user_turn: UserTurn) -> str:
        signal_terms = " ".join(signal.modality.value for signal in user_turn.signals)
        return " ".join(part for part in [user_turn.user_text, signal_terms] if part).strip()

    def _episode_document(self, episode: MemoryEpisode) -> str:
        parts = [
            episode.user_turn.user_text,
            " ".join(episode.right_state.emotional_labels),
            episode.left_result.response_text,
            " ".join(action.action_type for action in episode.left_result.actions),
            " ".join(episode.left_result.reasoning_summary),
        ]
        return " ".join(part for part in parts if part).strip()

    def _rule_document(self, rule: SemanticRule) -> str:
        return " ".join(
            part
            for part in [
                rule.statement,
                " ".join(rule.tags),
                " ".join(rule.supporting_episodes),
            ]
            if part
        ).strip()

    def _episode_payload(self, episode: MemoryEpisode) -> dict:
        return {
            "text": episode.user_turn.user_text,
            "session": episode.user_turn.session_id,
            "observed_at": episode.user_turn.observed_at.isoformat(),
            "recorded_at": episode.recorded_at.isoformat(),
            "emotional_labels": episode.right_state.emotional_labels if episode.right_state else [],
            "response_text": episode.left_result.response_text if episode.left_result else "",
            "action_types": [action.action_type for action in episode.left_result.actions] if episode.left_result else [],
            "reasoning_summary": episode.left_result.reasoning_summary if episode.left_result else [],
        }

    def _episode_from_point(self, point) -> MemoryEpisode:
        payload = point.payload or {}
        user_turn = UserTurn(
            session_id=payload.get("session", "*"),
            user_text=payload.get("text", ""),
            signals=[],
            observed_at=_parse_datetime(payload.get("observed_at")),
        )
        right_state = _placeholder_right_state(
            user_turn,
            emotional_labels=list(payload.get("emotional_labels", [])),
            salience=0.5 if payload.get("emotional_labels") else 0.15,
        )
        return MemoryEpisode(
            episode_id=str(point.id),
            recorded_at=_parse_datetime(payload.get("recorded_at")),
            user_turn=user_turn,
            right_state=right_state,
            bridge_packet=_placeholder_bridge_packet(right_state),
            left_result=_placeholder_left_result(
                response_text=payload.get("response_text", ""),
                action_types=list(payload.get("action_types", [])),
                reasoning_summary=list(payload.get("reasoning_summary", [])),
            ),
        )

    def _rule_from_point(self, point) -> SemanticRule:
        payload = point.payload or {}
        return SemanticRule(
            rule_id=payload.get("rule_id", str(point.id)),
            statement=payload.get("statement", ""),
            strength=float(payload.get("strength", 1.0)),
            supporting_episodes=list(payload.get("supporting_episodes", [])),
            tags=list(payload.get("tags", [])),
        )


def _parse_datetime(value: str | None) -> datetime:
    if not value:
        return utc_now()
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return utc_now()


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


def _placeholder_left_result(
    *,
    response_text: str = "",
    action_types: list[str] | None = None,
    reasoning_summary: list[str] | None = None,
) -> LeftHemisphereResult:
    actions = []
    for action_type in action_types or []:
        actions.append(
            PrimitiveAction(
                action_type=action_type,
                typed_signature="Placeholder -> Placeholder",
                payload={},
                safety_invariants=["placeholder reconstructed from qdrant payload"],
            )
        )

    return LeftHemisphereResult(
        response_text=response_text,
        lambda_program=TypedLambdaProgram(
            signature="Placeholder",
            expression="",
            expected_effect="hydrate memory episode without executable actions",
        ),
        actions=actions,
        reasoning_summary=reasoning_summary or ["placeholder_memory_episode"],
        telemetry={"source": "qdrant_placeholder"},
    )
