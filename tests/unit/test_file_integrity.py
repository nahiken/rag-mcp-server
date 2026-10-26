"""Tests for FileIntegrityChecker — file-level SHA256 incremental skip (T1).

Red/green contract (docs/spec-p10-pipeline.md, ticket T1):
- FileIntegrityChecker persists (file_hash, status, ...) in SQLite (WAL),
  default db path data/db/ingestion_history.db, db_path injectable
- should_skip(file_hash) -> True iff a success record exists; a "failed"
  record must NOT skip (a failed file retries on next ingest)
- mark_success / mark_failed upsert (second mark overwrites the first)
- remove_record(file_hash) is idempotent; after it a file can be re-ingested
- persistence survives a new instance on the same db path
- WAL: two connections can both write (concurrent-safe)

The only seam is FileIntegrityChecker itself (S1): it never touches the
filesystem, callers pass the file_hash in (S2).
"""

from __future__ import annotations

import sqlite3

import pytest

from rag_mcp_server.libs.loader.file_integrity import FileIntegrityChecker


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "ingestion_history.db"


def test_should_skip_false_when_no_record(db_path):
    checker = FileIntegrityChecker(db_path=db_path)
    assert checker.should_skip("hash-abc") is False


def test_mark_success_then_should_skip_true(db_path):
    checker = FileIntegrityChecker(db_path=db_path)
    checker.mark_success("hash-abc", "docs/notes.txt", chunk_count=2)
    assert checker.should_skip("hash-abc") is True


def test_mark_failed_then_should_skip_false(db_path):
    # a failed file must retry on the next ingest, never be skipped
    checker = FileIntegrityChecker(db_path=db_path)
    checker.mark_failed("hash-abc", "pdf parse failed")
    assert checker.should_skip("hash-abc") is False


def test_mark_overwrites_previous_status(db_path):
    checker = FileIntegrityChecker(db_path=db_path)
    checker.mark_failed("hash-abc", "first failure")
    assert checker.should_skip("hash-abc") is False
    checker.mark_success("hash-abc", "docs/notes.txt", chunk_count=1)
    assert checker.should_skip("hash-abc") is True


def test_remove_record_makes_file_reingestable(db_path):
    checker = FileIntegrityChecker(db_path=db_path)
    checker.mark_success("hash-abc", "docs/notes.txt", chunk_count=1)
    assert checker.should_skip("hash-abc") is True
    checker.remove_record("hash-abc")
    assert checker.should_skip("hash-abc") is False
    # idempotent: removing a missing record does not raise
    checker.remove_record("hash-abc")


def test_persistence_across_instances(db_path):
    first = FileIntegrityChecker(db_path=db_path)
    first.mark_success("hash-abc", "docs/notes.txt", chunk_count=2)

    second = FileIntegrityChecker(db_path=db_path)
    assert second.should_skip("hash-abc") is True


def test_wal_two_connections_can_write(db_path):
    # WAL mode: a second connection (a concurrent ingest) sees writes and can
    # write its own records without locking the first one out
    checker = FileIntegrityChecker(db_path=db_path)
    checker.mark_success("hash-a", "docs/a.txt", chunk_count=1)

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            "INSERT OR REPLACE INTO file_integrity (file_hash, status, file_path, chunk_count, error_msg, updated_at)"
            " VALUES (?, 'success', ?, 1, NULL, ?)",
            ("hash-b", "docs/b.txt", "2026-08-25"),
        )
        conn.commit()
    finally:
        conn.close()

    assert checker.should_skip("hash-a") is True
    assert checker.should_skip("hash-b") is True


def test_default_db_path_is_data_db_ingestion_history(db_path):
    # default path per spec; the temp-db tests above inject db_path
    checker = FileIntegrityChecker()
    assert checker.db_path.parts[-3:] == ("data", "db", "ingestion_history.db")
