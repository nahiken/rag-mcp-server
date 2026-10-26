"""Unit tests for the optional persistent BM25 corpus (ADR 0021).

When ``retrieval.bm25_persist`` is on and the collection's chunks.jsonl
snapshot exists, rag_service.query builds the sparse corpus from the snapshot
(no vector-store scan). Otherwise it falls back to the ADR 0003 rebuild.
``DocumentManager.delete`` with an injected ``evidence_dir`` rewrites the
snapshot minus the deleted document.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from rag_mcp_server.application import rag_service
from rag_mcp_server.application.rag_service import ingest, query
from rag_mcp_server.document_manager.manager import DocumentManager
from rag_mcp_server.ingestion.jsonl_store import read_chunks
from rag_mcp_server.ingestion.pipeline import IngestPipeline
from rag_mcp_server.libs.embedding.fake import FakeEmbedding
from rag_mcp_server.libs.vector_store.chroma_store import ChromaVectorStore

COLLECTION = "test_bm25p"


def _env(tmp):
    root = Path(tmp)
    # Chroma collection derived from tmp_path (unique per test) — a fixed name
    # would share ephemeral backend state across tests (known flake).
    col = ("test_bm25p_" + root.name.replace("-", "_"))[:50].rstrip("._-")
    if len(col) < 3:
        col = "test_bm25p_tmp"
    (root / "docs").mkdir()
    (root / "docs" / "notes.txt").write_text(
        "RAG 是检索增强生成，使用关键词检索。\n\nMCP 是模型上下文协议。",
        encoding="utf-8",
    )
    embedding = FakeEmbedding(dim=4)
    store = ChromaVectorStore(persist_dir=None, collection=col)
    pipe = IngestPipeline(
        embedding=embedding, store=store, index_dir=root / "out",
        collection=col,
    )
    ingest(root / "docs", base=root / "docs", index_dir=root / "out",
           pipeline=pipe, trace_dir=root, collection=col)
    return root / "out", col, pipe


def _persist_on(monkeypatch):
    """Force retrieval.bm25_persist=true (snapshot-based sparse corpus)."""
    import rag_mcp_server.config.settings as settings_mod
    from rag_mcp_server.config.settings import DEFAULTS

    def fake_load_settings():
        data = {k: dict(v) if isinstance(v, dict) else v for k, v in DEFAULTS.items()}
        data["retrieval"]["bm25_persist"] = True
        data["trace"] = {"enabled": False, "dir": "logs"}
        return data

    monkeypatch.setattr(settings_mod, "load_settings", fake_load_settings)


def test_persistent_corpus_used_when_enabled(monkeypatch, tmp_path):
    """bm25_persist on + snapshot exists -> query reads chunks.jsonl, not store."""
    _persist_on(monkeypatch)
    out, col, pipe = _env(tmp_path)
    # verify the snapshot exists (written by ingest)
    snapshot = rag_service.evidence_dir(out, col) / "chunks.jsonl"
    assert snapshot.is_file()

    # spy on _corpus_from_store — should NOT be called when persisted path works
    calls = {"rebuild": 0}
    orig = rag_service._corpus_from_store

    def spy(*a, **k):
        calls["rebuild"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(rag_service, "_corpus_from_store", spy)
    results = query("检索", index_dir=out, pipeline=pipe, trace_dir=tmp_path,
                    strategy="sparse", collection=col)
    assert results, "sparse query must hit"
    assert calls["rebuild"] == 0, "persisted corpus should avoid the store rebuild"


def test_persistent_corpus_falls_back_when_snapshot_missing(monkeypatch, tmp_path):
    """bm25_persist on but no snapshot -> rebuild from store (ADR 0003)."""
    _persist_on(monkeypatch)
    root = Path(tmp_path)
    out, col, pipe = _env(tmp_path)
    # remove the snapshot to force fallback
    (rag_service.evidence_dir(out, col) / "chunks.jsonl").unlink()

    results = query("检索", index_dir=out, pipeline=pipe, trace_dir=root,
                    strategy="sparse", collection=col)
    assert results, "fallback rebuild must still hit"


def test_persistent_corpus_with_where_filter_rebuilds(monkeypatch, tmp_path):
    """C2: a filtered query never uses the persisted (unfiltered) snapshot —
    it must rebuild a where-honoring corpus, exactly like the dense route. The
    snapshot exists and bm25_persist is on, yet _corpus_from_store is called."""
    _persist_on(monkeypatch)
    root = Path(tmp_path)
    out, col, pipe = _env(tmp_path)
    # snapshot exists (written by ingest); the filter must bypass it
    assert (rag_service.evidence_dir(out, col) / "chunks.jsonl").is_file()

    calls = {"rebuild": 0}
    orig = rag_service._corpus_from_store

    def spy(*a, **k):
        calls["rebuild"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(rag_service, "_corpus_from_store", spy)
    results = query(
        "检索 source:notes.txt",
        index_dir=out, pipeline=pipe, trace_dir=root,
        strategy="sparse", collection=col,
    )
    assert calls["rebuild"] == 1, "a filtered query must rebuild (ignore snapshot)"
    # the filter narrows the corpus to notes.txt, which still matches
    assert results, "filtered sparse query must still hit"

    # and a hybrid filtered query rebuilds too
    results = query(
        "检索 source:notes.txt",
        index_dir=out, pipeline=pipe, trace_dir=root,
        strategy="hybrid", collection=col,
    )
    assert calls["rebuild"] == 2
    assert results


def test_persistent_corpus_off_rebuilds(monkeypatch, tmp_path):
    """bm25_persist off (default) -> rebuild from store (ADR 0003)."""
    out, col, pipe = _env(tmp_path)  # settings default: bm25_persist False
    results = query("检索", index_dir=out, pipeline=pipe, trace_dir=tmp_path,
                    strategy="sparse", collection=col)
    assert results


def test_delete_rewrites_snapshot_when_evidence_dir_injected(tmp_path):
    """DocumentManager.delete with evidence_dir rewrites chunks.jsonl minus the doc."""
    out, col, pipe = _env(tmp_path)
    manager = DocumentManager(store=pipe.store, evidence_dir=rag_service.evidence_dir(out, col))
    doc_id = f"{col}::notes-txt"
    before = read_chunks(rag_service.evidence_dir(out, col) / "chunks.jsonl")
    assert any(c.document_id == doc_id for c in before)
    removed = manager.delete(doc_id)
    assert removed >= 1
    after = read_chunks(rag_service.evidence_dir(out, col) / "chunks.jsonl")
    assert not any(c.document_id == doc_id for c in after)
    assert len(after) == len(before) - removed


def test_delete_without_evidence_dir_leaves_snapshot_untouched(tmp_path):
    """Default delete (no evidence_dir) leaves chunks.jsonl as-is (ADR 0021 opt-in)."""
    out, col, pipe = _env(tmp_path)
    manager = DocumentManager(store=pipe.store)
    before = read_chunks(rag_service.evidence_dir(out, col) / "chunks.jsonl")
    manager.delete(f"{col}::notes-txt")
    after = read_chunks(rag_service.evidence_dir(out, col) / "chunks.jsonl")
    assert len(after) == len(before)  # unchanged
