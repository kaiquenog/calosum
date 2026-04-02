from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class GeaExperienceStoreConfig:
    path: Path


class SqliteGeaExperienceStore:
    """Persistent experience sharing store for GEA variants."""

    def __init__(self, config: GeaExperienceStoreConfig) -> None:
        self.config = config
        self.config.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def record_experience(
        self,
        *,
        context_type: str,
        variant_id: str,
        score: float,
        reward: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        payload = json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO gea_experience(context_type, variant_id, score, reward, metadata_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (context_type, variant_id, float(score), float(reward), payload),
            )
            conn.commit()

    def variant_prior(self, *, context_type: str, variant_id: str, limit: int = 100) -> float:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT reward FROM gea_experience
                WHERE context_type = ? AND variant_id = ?
                ORDER BY id DESC LIMIT ?
                """,
                (context_type, variant_id, max(1, int(limit))),
            ).fetchall()
        if not rows:
            return 0.0
        rewards = [float(item[0]) for item in rows]
        return sum(rewards) / len(rewards)

    def context_stats(self, *, context_type: str) -> dict[str, Any]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT variant_id, COUNT(*) AS n, AVG(reward) AS avg_reward
                FROM gea_experience
                WHERE context_type = ?
                GROUP BY variant_id
                """,
                (context_type,),
            ).fetchall()
        return {
            "context_type": context_type,
            "variants": [
                {
                    "variant_id": str(variant_id),
                    "samples": int(n),
                    "avg_reward": float(avg_reward or 0.0),
                }
                for variant_id, n, avg_reward in rows
            ],
        }

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS gea_experience (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    context_type TEXT NOT NULL,
                    variant_id TEXT NOT NULL,
                    score REAL NOT NULL,
                    reward REAL NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_gea_context_variant
                ON gea_experience(context_type, variant_id)
                """
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.config.path)
