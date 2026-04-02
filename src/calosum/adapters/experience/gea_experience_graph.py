from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from calosum.shared.ports import ExperienceStorePort

@dataclass
class ExperienceEdge:
    source_variant: str
    target_variant: str
    experience_type: str  # "strategy", "critique", "success_pattern"
    weight: float
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True)
class GeaExperienceGraphConfig:
    path: Path

class GraphGeaExperienceStore(ExperienceStorePort):
    """GEA experience store com grafo de transferência.

    Implementa o evolutionary graph do paper GEA (Weng et al., 2602.04837):
    - Nós = variantes de agentes
    - Arestas = transferência de experiência entre variantes
    - Peso = força da transferência (baseada em similaridade de contexto)
    """

    def __init__(self, config: GeaExperienceGraphConfig) -> None:
        self.config = config
        self.config.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.config.path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS experiences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    variant_id TEXT NOT NULL,
                    context_type TEXT NOT NULL,
                    score REAL,
                    reward REAL,
                    metadata_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS experience_graph (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_variant TEXT NOT NULL,
                    target_variant TEXT NOT NULL,
                    experience_type TEXT NOT NULL,
                    weight REAL DEFAULT 1.0,
                    metadata_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_exp_variant ON experiences(variant_id);
                CREATE INDEX IF NOT EXISTS idx_graph_source ON experience_graph(source_variant);
                CREATE INDEX IF NOT EXISTS idx_graph_target ON experience_graph(target_variant);
            """)
            conn.commit()

    def record_experience(
        self,
        *,
        context_type: str,
        variant_id: str,
        score: float,
        reward: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with sqlite3.connect(self.config.path) as conn:
            conn.execute(
                "INSERT INTO experiences (variant_id, context_type, score, reward, metadata_json) VALUES (?, ?, ?, ?, ?)",
                (variant_id, context_type, float(score), float(reward), json.dumps(metadata or {})),
            )
            conn.commit()

    def add_edge(
        self,
        source: str,
        target: str,
        experience_type: str,
        weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Adiciona aresta de transferência de experiência."""
        with sqlite3.connect(self.config.path) as conn:
            conn.execute(
                "INSERT INTO experience_graph (source_variant, target_variant, experience_type, weight, metadata_json) VALUES (?, ?, ?, ?, ?)",
                (source, target, experience_type, float(weight), json.dumps(metadata or {})),
            )
            conn.commit()

    def variant_prior(
        self,
        *,
        context_type: str,
        variant_id: str,
        limit: int = 100,
    ) -> float:
        """Prior com transferência: score próprio + scores de variantes conectadas."""
        with sqlite3.connect(self.config.path) as conn:
            # Score próprio
            own_row = conn.execute(
                "SELECT AVG(reward) FROM experiences WHERE variant_id = ? AND context_type = ?",
                (variant_id, context_type),
            ).fetchone()
            own = own_row[0] if own_row and own_row[0] is not None else 0.0

            # Scores de variantes conectadas (transferência)
            transfer_row = conn.execute(
                """SELECT AVG(e.reward) * g.weight
                   FROM experience_graph g
                   JOIN experiences e ON e.variant_id = g.source_variant
                   WHERE g.target_variant = ? AND e.context_type = ?""",
                (variant_id, context_type),
            ).fetchone()
            transfer = transfer_row[0] if transfer_row and transfer_row[0] is not None else 0.0

            if transfer > 0.0:
                return float(0.7 * own + 0.3 * transfer)  # Weighted combination
            return float(own)

    def get_transfer_candidates(self, variant_id: str, limit: int = 5) -> list[ExperienceEdge]:
        """Encontra variantes cujas experiências podem ser transferidas."""
        with sqlite3.connect(self.config.path) as conn:
            rows = conn.execute(
                """SELECT source_variant, target_variant, experience_type, weight, metadata_json
                   FROM experience_graph
                   WHERE target_variant = ?
                   ORDER BY weight DESC
                   LIMIT ?""",
                (variant_id, limit),
            ).fetchall()
            return [
                ExperienceEdge(
                    source_variant=r[0],
                    target_variant=r[1],
                    experience_type=r[2],
                    weight=r[3],
                    metadata=json.loads(r[4]) if r[4] else {},
                )
                for r in rows
            ]
