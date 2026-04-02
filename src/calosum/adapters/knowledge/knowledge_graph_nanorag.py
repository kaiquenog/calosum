from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx

from calosum.shared.types import KnowledgeTriple, UserTurn


@dataclass(slots=True)
class NanoGraphRAGKnowledgeGraphStore:
    """
    Store de grafo semântico orientado a subgrafo local.

    O Calosum já produz `KnowledgeTriple` estruturado no consolidator, então a
    integração útil aqui não é reextrair entidades com outro LLM, mas manter um
    grafo consultável com expansão multi-hop e persistência local. Quando a
    dependência `nano-graphrag` existir no ambiente, o store sinaliza essa
    compatibilidade na telemetria/configuração; sem a dependência, o fallback é
    uma implementação leve baseada em NetworkX.
    """

    storage_path: Path | None = None
    triples: dict[tuple[str, str, str], KnowledgeTriple] = field(default_factory=dict)
    graph: nx.MultiDiGraph = field(default_factory=nx.MultiDiGraph)
    backend_name: str = field(init=False)
    _last_sync_ns: int | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.backend_name = (
            "nanorag_compatible_networkx"
            if importlib.util.find_spec("nano_graphrag")
            else "networkx_graph_rag_fallback"
        )
        if self.storage_path is not None:
            self.storage_path = Path(self.storage_path)
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            if self.storage_path.exists():
                self._reload_from_disk()

    def upsert(self, triple: KnowledgeTriple) -> None:
        key = (triple.subject, triple.predicate, triple.object)
        current = self.triples.get(key)
        if current is not None and current.weight > triple.weight:
            return

        self.triples[key] = triple
        self._rebuild_edge(triple)
        if self.storage_path is not None:
            self._append_record(triple)
            self._last_sync_ns = self.storage_path.stat().st_mtime_ns

    def all(self) -> list[KnowledgeTriple]:
        self._sync_from_disk()
        return sorted(self.triples.values(), key=lambda item: item.weight, reverse=True)

    def query(self, user_turn: UserTurn, limit: int = 8) -> list[KnowledgeTriple]:
        self._sync_from_disk()
        if not self.triples:
            return []

        terms = _query_terms(user_turn.user_text)
        if not terms:
            return self.all()[:limit]

        seed_nodes = self._seed_nodes(terms)
        expanded_nodes = self._expanded_nodes(seed_nodes)
        ranked = sorted(
            self.triples.values(),
            key=lambda triple: self._triple_score(triple, terms, seed_nodes, expanded_nodes),
            reverse=True,
        )
        return ranked[:limit]

    def _seed_nodes(self, terms: set[str]) -> set[str]:
        matches: set[str] = set()
        for triple in self.triples.values():
            subject_tokens = _query_terms(triple.subject)
            object_tokens = _query_terms(triple.object)
            if subject_tokens.intersection(terms):
                matches.add(triple.subject)
            if object_tokens.intersection(terms):
                matches.add(triple.object)
        return matches

    def _expanded_nodes(self, seed_nodes: set[str]) -> set[str]:
        if not seed_nodes:
            return set()

        expanded = set(seed_nodes)
        undirected = self.graph.to_undirected()
        for node in seed_nodes:
            if not undirected.has_node(node):
                continue
            lengths = nx.single_source_shortest_path_length(undirected, node, cutoff=1)
            expanded.update(lengths.keys())
        return expanded

    def _triple_score(
        self,
        triple: KnowledgeTriple,
        terms: set[str],
        seed_nodes: set[str],
        expanded_nodes: set[str],
    ) -> tuple[float, float]:
        triple_terms = _query_terms(
            f"{triple.subject} {triple.predicate} {triple.object}"
        )
        overlap = len(triple_terms.intersection(terms))
        seed_bonus = 0.35 if triple.subject in seed_nodes or triple.object in seed_nodes else 0.0
        hop_bonus = 0.2 if triple.subject in expanded_nodes or triple.object in expanded_nodes else 0.0
        degree_bonus = min(
            0.15,
            0.03
            * (
                self.graph.degree(triple.subject)
                + self.graph.degree(triple.object)
            ),
        )
        score = overlap + seed_bonus + hop_bonus + degree_bonus + float(triple.weight)
        return score, float(triple.weight)

    def _rebuild_edge(self, triple: KnowledgeTriple) -> None:
        self.graph.remove_edges_from(
            [
                (u, v, key)
                for u, v, key, data in self.graph.edges(keys=True, data=True)
                if u == triple.subject
                and v == triple.object
                and data.get("predicate") == triple.predicate
            ]
        )
        self.graph.add_edge(
            triple.subject,
            triple.object,
            key=triple.predicate,
            predicate=triple.predicate,
            weight=triple.weight,
            source_rule_id=triple.source_rule_id,
        )

    def _reload_from_disk(self) -> None:
        assert self.storage_path is not None
        self.triples = {}
        self.graph = nx.MultiDiGraph()
        for line in self.storage_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            triple = _triple_from_dict(json.loads(line))
            self.triples[(triple.subject, triple.predicate, triple.object)] = triple
            self._rebuild_edge(triple)
        self._last_sync_ns = self.storage_path.stat().st_mtime_ns

    def _sync_from_disk(self) -> None:
        if self.storage_path is None or not self.storage_path.exists():
            return
        current_sync = self.storage_path.stat().st_mtime_ns
        if self._last_sync_ns is None or current_sync > self._last_sync_ns:
            self._reload_from_disk()

    def _append_record(self, triple: KnowledgeTriple) -> None:
        assert self.storage_path is not None
        with self.storage_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_triple_to_dict(triple), ensure_ascii=False, sort_keys=True) + "\n")


def _query_terms(text: str) -> set[str]:
    return {
        token
        for token in text.lower().replace("/", " ").replace("-", " ").split()
        if len(token) >= 3
    }


def _triple_from_dict(data: dict[str, object]) -> KnowledgeTriple:
    return KnowledgeTriple(
        subject=str(data.get("subject", "")),
        predicate=str(data.get("predicate", "")),
        object=str(data.get("object", "")),
        weight=float(data.get("weight", 1.0)),
        source_rule_id=(
            str(data["source_rule_id"])
            if data.get("source_rule_id") is not None
            else None
        ),
    )


def _triple_to_dict(triple: KnowledgeTriple) -> dict[str, object]:
    return {
        "subject": triple.subject,
        "predicate": triple.predicate,
        "object": triple.object,
        "weight": triple.weight,
        "source_rule_id": triple.source_rule_id,
    }
