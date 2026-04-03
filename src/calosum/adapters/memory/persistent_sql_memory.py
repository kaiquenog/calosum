from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from calosum.domain.memory.memory import DualMemorySystem
from calosum.adapters.memory.sql_memory import (
    SQLiteEpisodicStore,
    SQLiteSemanticStore,
    SQLiteSemanticGraphStore,
    SQLiteSessionStore,
)
from calosum.adapters.memory.sql_memory_duckdb import (
    DuckDBEpisodicStore,
    DuckDBSemanticGraphStore,
    DuckDBSemanticStore,
    DuckDBSessionStore,
    duckdb_available,
)
from calosum.shared.models.types import (
    CognitiveWorkspace,
    MemoryEpisode,
    SessionDiagnostic,
)


@dataclass(slots=True)
class PersistentDualMemorySystem(DualMemorySystem):
    session_store: SQLiteSessionStore | None = None

    @classmethod
    def from_directory(cls, base_path: str | Path, consolidator=None) -> "PersistentDualMemorySystem":
        base_dir = Path(base_path)
        base_dir.mkdir(parents=True, exist_ok=True)
        db_path = base_dir / "calosum.db"
        
        session_store = SQLiteSessionStore(db_path)
        kwargs = {
            "episodic_store": SQLiteEpisodicStore(db_path),
            "semantic_store": SQLiteSemanticStore(db_path),
            "graph_store": SQLiteSemanticGraphStore(db_path),
            "session_store": session_store,
        }
        if consolidator is not None:
            kwargs["consolidator"] = consolidator
        return cls(**kwargs)

    @classmethod
    def from_duckdb(cls, db_path: str | Path, consolidator=None) -> "PersistentDualMemorySystem":
        duckdb_file = Path(db_path)
        duckdb_file.parent.mkdir(parents=True, exist_ok=True)
        if not duckdb_available():
            return cls.from_directory(duckdb_file.parent, consolidator=consolidator)

        session_store = DuckDBSessionStore(duckdb_file)
        kwargs = {
            "episodic_store": DuckDBEpisodicStore(duckdb_file),
            "semantic_store": DuckDBSemanticStore(duckdb_file),
            "graph_store": DuckDBSemanticGraphStore(duckdb_file),
            "session_store": session_store,
        }
        if consolidator is not None:
            kwargs["consolidator"] = consolidator
        return cls(**kwargs)

    # Override workspace & diagnostic methods to use session_store
    def load_workspace(self, session_id: str) -> CognitiveWorkspace | None:
        if self.session_store:
            return self.session_store.load_workspace(session_id)
        return super().load_workspace(session_id)

    async def aload_workspace(self, session_id: str) -> CognitiveWorkspace | None:
        return self.load_workspace(session_id)

    def save_workspace(self, session_id: str, workspace: CognitiveWorkspace) -> None:
        if self.session_store:
            self.session_store.save_workspace(session_id, workspace)
        else:
            super().save_workspace(session_id, workspace)

    async def asave_workspace(self, session_id: str, workspace: CognitiveWorkspace) -> None:
        self.save_workspace(session_id, workspace)

    def load_diagnostic(self, session_id: str) -> SessionDiagnostic | None:
        if self.session_store:
            return self.session_store.load_diagnostic(session_id)
        return super().load_diagnostic(session_id)

    async def aload_diagnostic(self, session_id: str) -> SessionDiagnostic | None:
        return self.load_diagnostic(session_id)

    def save_diagnostic(self, session_id: str, diagnostic: SessionDiagnostic) -> None:
        if self.session_store:
            self.session_store.save_diagnostic(session_id, diagnostic)
        else:
            super().save_diagnostic(session_id, diagnostic)

    async def asave_diagnostic(self, session_id: str, diagnostic: SessionDiagnostic) -> None:
        self.save_diagnostic(session_id, diagnostic)
