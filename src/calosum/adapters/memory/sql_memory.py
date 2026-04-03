from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar, Generic

from calosum.shared.models.types import (
    CognitiveWorkspace,
    KnowledgeTriple,
    MemoryEpisode,
    SemanticRule,
    SessionDiagnostic,
    UserTurn,
)
from calosum.shared.utils.serialization import from_primitive, to_primitive

T = TypeVar("T")

class SQLStoreBase:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_graph (
                    subject TEXT,
                    predicate TEXT,
                    object TEXT,
                    weight REAL,
                    source_rule_id TEXT,
                    PRIMARY KEY (subject, predicate, object)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS workspaces (
                    session_id TEXT PRIMARY KEY,
                    data TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS diagnostics (
                    session_id TEXT PRIMARY KEY,
                    data TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

class SQLiteSemanticGraphStore(SQLStoreBase):
    def upsert(self, triple: KnowledgeTriple) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO knowledge_graph (subject, predicate, object, weight, source_rule_id)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(subject, predicate, object) DO UPDATE SET
                    weight = excluded.weight,
                    source_rule_id = excluded.source_rule_id
                WHERE excluded.weight >= weight
                """,
                (triple.subject, triple.predicate, triple.object, triple.weight, triple.source_rule_id)
            )
            conn.commit()

    def all(self) -> list[KnowledgeTriple]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM knowledge_graph ORDER BY weight DESC").fetchall()
            return [
                KnowledgeTriple(
                    subject=row["subject"],
                    predicate=row["predicate"],
                    object=row["object"],
                    weight=row["weight"],
                    source_rule_id=row["source_rule_id"]
                ) for row in rows
            ]

    def query(self, user_turn: UserTurn, limit: int = 8) -> list[KnowledgeTriple]:
        # Simple term matching for now, similar to in-memory implementation
        terms = set(user_turn.user_text.lower().split())
        all_triples = self.all()
        ranked = sorted(
            all_triples,
            key=lambda triple: (
                len(terms.intersection(f"{triple.subject} {triple.predicate} {triple.object}".lower().split())),
                triple.weight,
            ),
            reverse=True,
        )
        return ranked[:limit]

class SQLiteEpisodicStore(SQLStoreBase):
    def _init_db(self) -> None:
        super()._init_db()
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS episodic_memory (
                    episode_id TEXT PRIMARY KEY,
                    data TEXT,
                    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    session_id TEXT,
                    user_text TEXT
                )
            """)
            conn.commit()

    def add(self, episode: MemoryEpisode) -> None:
        data = json.dumps(to_primitive(episode))
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO episodic_memory (episode_id, data, recorded_at, session_id, user_text) VALUES (?, ?, ?, ?, ?)",
                (episode.episode_id, data, episode.recorded_at.isoformat(), episode.user_turn.session_id, episode.user_turn.user_text)
            )
            conn.commit()

    def query(self, user_turn: UserTurn, limit: int = 5) -> list[MemoryEpisode]:
        query_terms = set(user_turn.user_text.lower().split())
        with self._get_connection() as conn:
            # First, try to get episodes from the same session
            rows = conn.execute(
                "SELECT data FROM episodic_memory WHERE session_id = ? ORDER BY recorded_at DESC",
                (user_turn.session_id,)
            ).fetchall()
            
            if not rows:
                # Fallback to all episodes if session is new
                rows = conn.execute("SELECT data FROM episodic_memory ORDER BY recorded_at DESC LIMIT 100").fetchall()
            
            episodes = [from_primitive(MemoryEpisode, json.loads(row["data"])) for row in rows]
            
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
            return [from_primitive(MemoryEpisode, json.loads(row["data"])) for row in rows]

class SQLiteSemanticStore(SQLStoreBase):
    def _init_db(self) -> None:
        super()._init_db()
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS semantic_rules (
                    rule_id TEXT PRIMARY KEY,
                    data TEXT,
                    strength REAL
                )
            """)
            conn.commit()

    def upsert(self, rule: SemanticRule) -> None:
        data = json.dumps(to_primitive(rule))
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO semantic_rules (rule_id, data, strength) VALUES (?, ?, ?) ON CONFLICT(rule_id) DO UPDATE SET data = excluded.data, strength = excluded.strength",
                (rule.rule_id, data, rule.strength)
            )
            conn.commit()

    def all(self) -> list[SemanticRule]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT data FROM semantic_rules ORDER BY strength DESC").fetchall()
            return [from_primitive(SemanticRule, json.loads(row["data"])) for row in rows]

class SQLiteSessionStore(SQLStoreBase):
    def save_workspace(self, session_id: str, workspace: CognitiveWorkspace) -> None:
        data = json.dumps(to_primitive(workspace))
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO workspaces (session_id, data) VALUES (?, ?) ON CONFLICT(session_id) DO UPDATE SET data = excluded.data, updated_at = CURRENT_TIMESTAMP",
                (session_id, data)
            )
            conn.commit()

    def load_workspace(self, session_id: str) -> CognitiveWorkspace | None:
        with self._get_connection() as conn:
            row = conn.execute("SELECT data FROM workspaces WHERE session_id = ?", (session_id,)).fetchone()
            if row:
                return from_primitive(CognitiveWorkspace, json.loads(row["data"]))
            return None

    def save_diagnostic(self, session_id: str, diagnostic: SessionDiagnostic) -> None:
        data = json.dumps(to_primitive(diagnostic))
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO diagnostics (session_id, data) VALUES (?, ?) ON CONFLICT(session_id) DO UPDATE SET data = excluded.data, updated_at = CURRENT_TIMESTAMP",
                (session_id, data)
            )
            conn.commit()

    def load_diagnostic(self, session_id: str) -> SessionDiagnostic | None:
        with self._get_connection() as conn:
            row = conn.execute("SELECT data FROM diagnostics WHERE session_id = ?", (session_id,)).fetchone()
            if row:
                return from_primitive(SessionDiagnostic, json.loads(row["data"]))
            return None
