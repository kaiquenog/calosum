from __future__ import annotations

import uuid
from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from calosum.shared.async_utils import run_sync
from calosum.shared.types import (
    ConsolidationReport,
    KnowledgeTriple,
    MemoryContext,
    MemoryEpisode,
    SemanticRule,
    UserTurn,
    utc_now,
)


@dataclass(slots=True)
class QdrantAdapterConfig:
    url: str = "http://localhost:6333"
    episodes_collection: str = "episodes"
    rules_collection: str = "semantic_rules"


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
        # Usa um threshold dummy vector size de 384 (e.g. all-MiniLM-L6-v2)
        VECTOR_SIZE = 384
        for col_name in [self.config.episodes_collection, self.config.rules_collection]:
            if not self.client.collection_exists(col_name):
                self.client.create_collection(
                    collection_name=col_name,
                    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
                )

    def build_context(self, user_turn: UserTurn, episodic_limit: int = 5) -> MemoryContext:
        return run_sync(self.abuild_context(user_turn, episodic_limit))

    async def abuild_context(self, user_turn: UserTurn, episodic_limit: int = 5) -> MemoryContext:
        """
        Em produção real, calcularíamos o embedding de 'user_turn.user_text'
        para buscar os vetores mais próximos. Neste esqueleto, como não temos
        um modelo de embedding acoplado no construtor de modo síncrono, fazemos um fetch genérico.
        """
        # Exemplo simulando fetch de episódios recentes (sem busca vetorial exata p/ simplificar):
        res, _ = await self.aclient.scroll(
            collection_name=self.config.episodes_collection,
            limit=episodic_limit,
            with_payload=True,
        )
        
        episodes = []
        for point in res:
            p = point.payload or {}
            if "text" in p:
                # Reconstruindo mock do episodio p/ retornar contexto
                episodes.append(
                    MemoryEpisode(
                        episode_id=str(point.id),
                        recorded_at=utc_now(),
                        user_turn=UserTurn(session_id="*", user_text=p["text"], signals=[]),
                        right_state=None,
                        bridge_packet=None,
                        left_result=None,
                    )
                )

        # Regras semanticas fixas para o contexto (recuperando como scroll tbm)
        rules_res, _ = await self.aclient.scroll(
            collection_name=self.config.rules_collection, limit=10, with_payload=True
        )
        
        rules = [
            SemanticRule(
                rule_id=str(r.id),
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
        # Dummy vector embedding genérico para o stub
        dummy_vector = [0.0] * 384
        
        await self.aclient.upsert(
            collection_name=self.config.episodes_collection,
            points=[
                PointStruct(
                    id=point_id,
                    vector=dummy_vector,
                    payload={"text": episode.user_turn.user_text, "session": episode.user_turn.session_id},
                )
            ],
        )

    def sleep_mode(self) -> ConsolidationReport:
        return run_sync(self.asleep_mode())

    async def asleep_mode(self) -> ConsolidationReport:
        # Mock consolidation process that could train LoRA or clean vector DB
        return ConsolidationReport(
            episodes_processed=0,
            rules_extracted=0,
            knowledge_triples_extracted=0,
            telemetry={"status": "Qdrant dual memory consolidation mock executed."},
        )
