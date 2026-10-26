"""File-level SHA256 incremental skip (performance optimisation, P10.5 T1).

Records which files were last ingested successfully, so a re-ingest of
unchanged files can be skipped without reloading / re-embedding / re-upserting
them. This is a *performance* layer: correctness is guaranteed separately by
the chunk-level content_hash idempotency in the pipeline (CONTEXT.md
"FileIntegrity 增量跳过" — the two coexist, layered). The checker never
touches the filesystem; callers compute and pass in the file_hash (seam S2).

Storage is SQLite with WAL mode (concurrent-safe across connections).
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

DEFAULT_DB_PATH = Path("data") / "db" / "ingestion_history.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS file_integrity (
    file_hash   TEXT PRIMARY KEY,
    status      TEXT NOT NULL,
    file_path   TEXT,
    chunk_count INTEGER DEFAULT 0,
    error_msg   TEXT,
    updated_at  TEXT
)
"""


class FileIntegrityChecker:
    """Persist file-hash -> ingest-status records in SQLite (WAL)."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_schema(self) -> None:
        conn = self._connect()
        try:
            conn.execute(_SCHEMA)
            conn.commit()
        finally:
            conn.close()

    def should_skip(self, file_hash: str) -> bool:
        """True iff a ``success`` record exists for ``file_hash``.

        A ``failed`` record must NOT skip — the file retries on the next
        ingest. Missing record -> False.
        """
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT status FROM file_integrity WHERE file_hash = ?",
                (file_hash,),
            ).fetchone()
            return row is not None and row[0] == "success"
        finally:
            conn.close()

    def mark_success(self, file_hash: str, file_path: str, chunk_count: int = 0) -> None:
        self._upsert(file_hash, "success", file_path, chunk_count, None)

    def mark_failed(self, file_hash: str, error_msg: str) -> None:
        self._upsert(file_hash, "failed", None, 0, error_msg)

    def _upsert(
        self,
        file_hash: str,
        status: str,
        file_path: str | None,
        chunk_count: int,
        error_msg: str | None,
    ) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO file_integrity"
                " (file_hash, status, file_path, chunk_count, error_msg, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (file_hash, status, file_path, chunk_count, error_msg, datetime.now().isoformat(timespec="seconds")),
            )
            conn.commit()
        finally:
            conn.close()

    def remove_record(self, file_hash: str) -> None:
        """Delete the record (idempotent); the file becomes re-ingestable."""
        conn = self._connect()
        try:
            conn.execute("DELETE FROM file_integrity WHERE file_hash = ?", (file_hash,))
            conn.commit()
        finally:
            conn.close()
