from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from calosum.shared.models.types import (
    CognitiveWorkspace,
    KnowledgeTriple,
    MemoryEpisode,
    SemanticRule,
    SessionDiagnostic,
    UserTurn,
)
from calosum.shared.utils.serialization import from_primitive, to_primitive


def duckdb_available() -> bool:
    return importlib.util.find_spec("duckdb") is not None


class DuckDBStoreBase:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        import duckdb

        return duckdb.connect(str(self.db_path))

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_graph (
                    subject VARCHAR,
                    predicate VARCHAR,
                    object VARCHAR,
                    weight DOUBLE,
                    source_rule_id VARCHAR,
                    PRIMARY KEY (subject, predicate, object)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS workspaces (
                    session_id VARCHAR PRIMARY KEY,
                    data VARCHAR,
                    updated_at TIMESTAMP DEFAULT NOW()
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS diagnostics (
                    session_id VARCHAR PRIMARY KEY,
                    data VARCHAR,
                    updated_at TIMESTAMP DEFAULT NOW()
                )
                """
            )


class DuckDBSemanticGraphStore(DuckDBStoreBase):
    def upsert(self, triple: KnowledgeTriple) -> None:
        with self._get_connection() as conn:
            conn.execute(
                "DELETE FROM knowledge_graph WHERE subject = ? AND predicate = ? AND object = ?",
                [triple.subject, triple.predicate, triple.object],
            )
            conn.execute(
                "INSERT INTO knowledge_graph(subject, predicate, object, weight, source_rule_id) VALUES (?, ?, ?, ?, ?)",
                [triple.subject, triple.predicate, triple.object, triple.weight, triple.source_rule_id],
            )

    def all(self) -> list[KnowledgeTriple]:
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT subject, predicate, object, weight, source_rule_id FROM knowledge_graph ORDER BY weight DESC"
            ).fetchall()
        return [
            KnowledgeTriple(
                subject=row[0],
                predicate=row[1],
                object=row[2],
                weight=float(row[3]),
                source_rule_id=row[4],
            )
            for row in rows
        ]

    def query(self, user_turn: UserTurn, limit: int = 8) -> list[KnowledgeTriple]:
        terms = set(user_turn.user_text.lower().split())
        ranked = sorted(
            self.all(),
            key=lambda triple: (
                len(terms.intersection(f"{triple.subject} {triple.predicate} {triple.object}".lower().split())),
                triple.weight,
            ),
            reverse=True,
        )
        return ranked[:limit]


class DuckDBEpisodicStore(DuckDBStoreBase):
    def _init_db(self) -> None:
        super()._init_db()
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS episodic_memory (
                    episode_id VARCHAR PRIMARY KEY,
                    data VARCHAR,
                    recorded_at VARCHAR,
                    session_id VARCHAR,
                    user_text VARCHAR
                )
                """
            )

    def add(self, episode: MemoryEpisode) -> None:
        data = json.dumps(to_primitive(episode))
        with self._get_connection() as conn:
            conn.execute("DELETE FROM episodic_memory WHERE episode_id = ?", [episode.episode_id])
            conn.execute(
                "INSERT INTO episodic_memory(episode_id, data, recorded_at, session_id, user_text) VALUES (?, ?, ?, ?, ?)",
                [
                    episode.episode_id,
                    data,
                    episode.recorded_at.isoformat(),
                    episode.user_turn.session_id,
                    episode.user_turn.user_text,
                ],
            )

    def query(self, user_turn: UserTurn, limit: int = 5) -> list[MemoryEpisode]:
        query_terms = set(user_turn.user_text.lower().split())
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT data FROM episodic_memory WHERE session_id = ? ORDER BY recorded_at DESC",
                [user_turn.session_id],
            ).fetchall()
            if not rows:
                rows = conn.execute(
                    "SELECT data FROM episodic_memory ORDER BY recorded_at DESC LIMIT 100"
                ).fetchall()
        episodes = [from_primitive(MemoryEpisode, json.loads(row[0])) for row in rows]
        ranked = sorted(
            episodes,
            key=lambda ep: (
                len(query_terms.intersection(ep.user_turn.user_text.lower().split())),
                ep.recorded_at,
            ),
            reverse=True,
        )
        return ranked[:limit]

    def all(self) -> list[MemoryEpisode]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT data FROM episodic_memory ORDER BY recorded_at DESC").fetchall()
        return [from_primitive(MemoryEpisode, json.loads(row[0])) for row in rows]


class DuckDBSemanticStore(DuckDBStoreBase):
    def _init_db(self) -> None:
        super()._init_db()
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS semantic_rules (
                    rule_id VARCHAR PRIMARY KEY,
                    data VARCHAR,
                    strength DOUBLE
                )
                """
            )

    def upsert(self, rule: SemanticRule) -> None:
        data = json.dumps(to_primitive(rule))
        with self._get_connection() as conn:
            conn.execute("DELETE FROM semantic_rules WHERE rule_id = ?", [rule.rule_id])
            conn.execute(
                "INSERT INTO semantic_rules(rule_id, data, strength) VALUES (?, ?, ?)",
                [rule.rule_id, data, float(rule.strength)],
            )

    def all(self) -> list[SemanticRule]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT data FROM semantic_rules ORDER BY strength DESC").fetchall()
        return [from_primitive(SemanticRule, json.loads(row[0])) for row in rows]


class DuckDBSessionStore(DuckDBStoreBase):
    def save_workspace(self, session_id: str, workspace: CognitiveWorkspace) -> None:
        data = json.dumps(to_primitive(workspace))
        with self._get_connection() as conn:
            conn.execute("DELETE FROM workspaces WHERE session_id = ?", [session_id])
            conn.execute(
                "INSERT INTO workspaces(session_id, data, updated_at) VALUES (?, ?, NOW())",
                [session_id, data],
            )

    def load_workspace(self, session_id: str) -> CognitiveWorkspace | None:
        with self._get_connection() as conn:
            row = conn.execute("SELECT data FROM workspaces WHERE session_id = ?", [session_id]).fetchone()
        if row:
            return from_primitive(CognitiveWorkspace, json.loads(row[0]))
        return None

    def save_diagnostic(self, session_id: str, diagnostic: SessionDiagnostic) -> None:
        data = json.dumps(to_primitive(diagnostic))
        with self._get_connection() as conn:
            conn.execute("DELETE FROM diagnostics WHERE session_id = ?", [session_id])
            conn.execute(
                "INSERT INTO diagnostics(session_id, data, updated_at) VALUES (?, ?, NOW())",
                [session_id, data],
            )

    def load_diagnostic(self, session_id: str) -> SessionDiagnostic | None:
        with self._get_connection() as conn:
            row = conn.execute("SELECT data FROM diagnostics WHERE session_id = ?", [session_id]).fetchone()
        if row:
            return from_primitive(SessionDiagnostic, json.loads(row[0]))
        return None
