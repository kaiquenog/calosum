from __future__ import annotations

import base64
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

import logging

from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    VectorParams,
)
from tenacity import retry, stop_after_attempt, wait_exponential

from calosum.adapters.memory.memory_qdrant_serializers import (
    episode_document,
    episode_from_point,
    episode_payload,
    rule_document,
    rule_from_point,
)
from calosum.adapters.memory.text_embeddings import TextEmbeddingAdapter, TextEmbeddingAdapterConfig
from calosum.domain.memory import InMemorySemanticGraphStore
from calosum.shared.async_utils import run_sync
from calosum.shared.types import (
    ConsolidationReport,
    KnowledgeTriple,
    MemoryContext,
    MemoryEpisode,
    SemanticRule,
    UserTurn,
)

if TYPE_CHECKING:
    from calosum.shared.ports import VectorCodecPort

from calosum.shared.ports import DatasetExporterPort

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class QdrantAdapterConfig:
    url: str = "http://localhost:6333"
    episodes_collection: str = "episodes"
    rules_collection: str = "semantic_rules"
    vector_size: int = 384
    scalar_quantization: bool = False


class QdrantDualMemoryAdapter:
    """
    Implementacao concreta do MemorySystemPort via Qdrant.

    O adapter faz indexacao semantica para episodios e regras, mas mantem
    fallback explicito quando o backend de embeddings nao esta disponivel.
    Suporta opcionalmente scalar quantization no Qdrant (4× compressão de disco)
    e um VectorCodecPort (TurboQuant) para compressão adicional no payload.
    """

    def __init__(
        self,
        config: QdrantAdapterConfig | None = None,
        *,
        embedder: TextEmbeddingAdapter | None = None,
        exporter: DatasetExporterPort | None = None,
        graph_store=None,
        codec: VectorCodecPort | None = None,
    ) -> None:
        self.config = config or QdrantAdapterConfig()
        self.client = QdrantClient(url=self.config.url)
        self.aclient = AsyncQdrantClient(url=self.config.url)
        self.embedder = embedder or TextEmbeddingAdapter(
            TextEmbeddingAdapterConfig(vector_size=self.config.vector_size)
        )
        self.exporter = exporter
        self.graph_store = graph_store or InMemorySemanticGraphStore()
        self.codec: VectorCodecPort | None = codec
        self._ensure_collections()

    def _ensure_collections(self) -> None:
        quant_cfg = None
        if self.config.scalar_quantization:
            quant_cfg = ScalarQuantization(
                scalar=ScalarQuantizationConfig(type=ScalarType.INT8)
            )
        for col_name in [self.config.episodes_collection, self.config.rules_collection]:
            if not self.client.collection_exists(col_name):
                self.client.create_collection(
                    collection_name=col_name,
                    vectors_config=VectorParams(
                        size=self.config.vector_size,
                        distance=Distance.COSINE,
                    ),
                    quantization_config=quant_cfg,
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

        episodes = [
            episode_from_point(point, codec=self.codec)
            for point in episode_points
            if getattr(point, "payload", None)
        ]
        session_episodes = [
            ep for ep in episodes if ep.user_turn.session_id == user_turn.session_id
        ]
        episodes = session_episodes or episodes
        rules = [rule_from_point(point) for point in rule_points if getattr(point, "payload", None)]
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
            knowledge_triples=self.graph_store.query(user_turn) or [
                KnowledgeTriple(subject="user", predicate="is", object="human")
            ],
        )

    def episode_count(self) -> int:
        return run_sync(self.aepisode_count())

    async def aepisode_count(self) -> int:
        result = await self.aclient.count(collection_name=self.config.episodes_collection, exact=True)
        return int(getattr(result, "count", 0))

    def store_episode(self, episode: MemoryEpisode) -> None:
        run_sync(self.astore_episode(episode))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
    async def astore_episode(self, episode: MemoryEpisode) -> None:
        point_id = str(uuid.uuid4())
        payload = episode_payload(episode)
        vector = await self._embed_text(episode_document(episode))

        # Check if episode has a valid latent_vector from JEPA
        latent_vector = payload.get("latent_vector", [])
        if latent_vector and self.config.vector_size > 0:
            # Validate latent_vector dimension matches expected vector_size
            if len(latent_vector) == self.config.vector_size:
                # Use the JEPA latent vector directly as the Qdrant vector
                vector = [float(x) for x in latent_vector]
            # If dimension doesn't match, fall back to text embedding (existing behavior)

        if self.codec is not None:
            latent = payload.get("latent_vector", [])
            if latent:
                compressed = self.codec.encode(list(latent))
                payload["latent_vector_compressed"] = base64.b64encode(compressed).decode("ascii")

        await self.aclient.upsert(
            collection_name=self.config.episodes_collection,
            points=[PointStruct(id=point_id, vector=vector, payload=payload)],
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
        episodes = [episode_from_point(p, codec=self.codec) for p in points if getattr(p, "payload", None)]
        consolidator = SleepModeConsolidator(exporter=self.exporter, minimum_frequency=1)
        report = consolidator.consolidate(episodes)

        if report.promoted_rules:
            vectors = await self.embedder.aembed_texts(
                [rule_document(rule) for rule in report.promoted_rules]
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

        for triple in report.graph_updates:
            self.graph_store.upsert(triple)

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
