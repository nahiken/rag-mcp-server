"""Tests for the DocumentManager seam — document lifecycle (task 12).

Red/green contract (docs/spec-p9-m4m5.md, task 12, ticket 12-T1):
- DocumentManager is constructed with an injected BaseVectorStore
- list() -> [{document_id, source_path, chunk_count}] grouped from get_all()
- detail(document_id) -> list[VectorRecord] (the doc's chunks; missing -> [])
- delete(document_id) -> int (chunks removed; missing doc -> 0, idempotent)
  — clears only Chroma; sparse rebuilds per query so it's automatically consistent
- stats() -> {"documents": n, "chunks": n}

The only test seam is DocumentManager. Tests drive it over a real temporary
Chroma (persist_dir=None), adding records directly — no pipeline involved.
"""

from __future__ import annotations

import pytest

from rag_mcp_server.document_manager.manager import DocumentManager
from rag_mcp_server.libs.vector_store.base import VectorRecord
from rag_mcp_server.libs.vector_store.chroma_store import ChromaVectorStore


def _record(chunk_id: str, document_id: str, source_path: str = "doc.txt", text: str = "") -> VectorRecord:
    return VectorRecord(
        chunk_id=chunk_id,
        metadata={
            "document_id": document_id,
            "source_path": source_path,
            "chunk_id": chunk_id,
            "content_hash": f"hash-{chunk_id}",
        },
        text=text or f"text-{chunk_id}",
    )


@pytest.fixture
def manager():
    """A DocumentManager over a temp Chroma with two docs (one multi-chunk)."""
    store = ChromaVectorStore(persist_dir=None, collection="test_docmgr_iso")
    records = [
        _record("doc-a::0", "doc-a", source_path="a.md", text="A 第一个 chunk"),
        _record("doc-a::1", "doc-a", source_path="a.md", text="A 第二个 chunk"),
        _record("doc-b::0", "doc-b", source_path="b.md", text="B 唯一 chunk"),
    ]
    # 4-dim one-hot so add passes the vector alignment (dim not semantically used)
    embeddings = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    store.add(records, embeddings)
    return DocumentManager(store=store)


def test_list_groups_by_document_id_with_chunk_counts(manager):
    docs = manager.list()
    by_id = {d["document_id"]: d for d in docs}
    assert set(by_id) == {"doc-a", "doc-b"}
    assert by_id["doc-a"]["chunk_count"] == 2
    assert by_id["doc-a"]["source_path"] == "a.md"
    assert by_id["doc-b"]["chunk_count"] == 1


def test_detail_returns_doc_chunks(manager):
    chunks = manager.detail("doc-a")
    assert [c.chunk_id for c in chunks] == ["doc-a::0", "doc-a::1"]
    assert all(c.metadata["document_id"] == "doc-a" for c in chunks)


def test_detail_missing_doc_returns_empty(manager):
    assert manager.detail("does-not-exist") == []


def test_delete_removes_doc_and_returns_chunk_count(manager):
    removed = manager.delete("doc-a")
    assert removed == 2
    # the doc's chunks are gone from the store
    assert manager.detail("doc-a") == []
    # other doc is untouched
    assert manager.detail("doc-b")[0].chunk_id == "doc-b::0"


def test_delete_missing_doc_is_idempotent_returns_zero(manager):
    assert manager.delete("does-not-exist") == 0


def test_delete_then_query_excludes_doc(manager):
    # end-to-end: after delete, a query over the same store must not return the doc
    manager.delete("doc-a")
    store = manager.store
    all_ids = {r.chunk_id for r in store.get_all()}
    assert "doc-a::0" not in all_ids
    assert "doc-a::1" not in all_ids
    assert "doc-b::0" in all_ids


def test_stats_counts_documents_and_chunks(manager):
    assert manager.stats() == {"documents": 2, "chunks": 3}


# -- P10.5 T3 delete 联动（FileIntegrity）-----------------------------------
def test_delete_removes_integrity_record_so_file_reingestable(tmp_path):
    """T3：DocumentManager 注入 integrity 后，delete 会 remove_record。

    端到端：经 pipeline 摄取（record 带 file_hash）→ delete（只清 Chroma +
    remove_record）→ 同一 checker 下重摄同一文件走完整链路（不再被「上次成功」跳过）。
    """
    from rag_mcp_server.ingestion.pipeline import IngestPipeline
    from rag_mcp_server.libs.embedding.fake import FakeEmbedding
    from rag_mcp_server.libs.loader.file_integrity import FileIntegrityChecker

    src = tmp_path / "source"
    src.mkdir()
    (src / "notes.txt").write_text("第一段。\n\n第二段。", encoding="utf-8")
    embedding = FakeEmbedding(dim=4)
    store = ChromaVectorStore(
        persist_dir=str(tmp_path / "storage" / "chroma"), collection="test_t3_docmgr"
    )
    checker = FileIntegrityChecker(db_path=tmp_path / "db" / "history.db")
    pipe = IngestPipeline(
        embedding=embedding,
        store=store,
        index_dir=tmp_path / "storage" / "index",
        integrity=checker,
        collection="test_t3_docmgr",  # ADR 0016: pipeline collection == store collection
    )

    # 摄取：record 带 file_hash（integrity 注入时）
    summary = pipe.ingest(src, base=src)
    assert summary.total_files == 1
    assert summary.total_chunks == 2
    records = store.get_all()
    assert records
    assert all("file_hash" in r.metadata for r in records)

    # 该文件当前被 integrity 标记为成功（should_skip True）
    record_file_hash = records[0].metadata["file_hash"]
    assert checker.should_skip(record_file_hash) is True

    # delete：注入 integrity 的 DocumentManager，删文档 + remove_record
    # ADR 0016: document_id carries the collection prefix
    doc_id = "test_t3_docmgr::notes-txt"
    manager = DocumentManager(store=store, integrity=checker)
    removed = manager.delete(doc_id)
    assert removed == 2
    assert manager.detail(doc_id) == []  # 只清 Chroma
    assert checker.should_skip(record_file_hash) is False  # 记录被 remove

    # 重摄同一文件：不再被 integrity 跳过，走完整链路
    re = pipe.ingest(src, base=src)
    assert re.total_files == 1
    assert re.total_skipped == 0
    assert re.total_chunks == 2


def test_delete_without_integrity_is_unchanged(tmp_path):
    """T3：DocumentManager 不注入 integrity（默认 None）时，delete 行为完全不变。

    仍只清 Chroma；不涉及 remove_record（无 checker 可调）。
    """
    from rag_mcp_server.libs.loader.file_integrity import FileIntegrityChecker

    store = ChromaVectorStore(
        persist_dir=str(tmp_path / "storage" / "chroma"), collection="test_t3_nointegrity"
    )
    records = [
        _record("doc-x::0", "doc-x", source_path="x.md", text="X 唯一 chunk"),
    ]
    store.add(records, [[1.0, 0.0, 0.0, 0.0]])

    manager = DocumentManager(store=store)  # integrity 缺省 None
    removed = manager.delete("doc-x")
    assert removed == 1
    assert manager.detail("doc-x") == []


def test_delete_with_integrity_but_record_without_file_hash_keeps_chroma_clear(tmp_path):
    """FIX-3：注入 checker 但 record 无 file_hash 的 legacy 分支。

    旧版本摄取的存量 record 没有 file_hash 键。delete 时 remove_record 不被触发
    （should_skip 仍是 True，记录残留——已知限制），但 Chroma 必须照常清空。
    """
    from rag_mcp_server.libs.loader.file_integrity import FileIntegrityChecker

    store = ChromaVectorStore(
        persist_dir=str(tmp_path / "storage" / "chroma"), collection="test_t3_legacy"
    )
    # 手工构造无 file_hash 的 record（legacy：4 键契约，无可选键）
    records = [
        _record("doc-y::0", "doc-y", source_path="y.md", text="Y legacy chunk"),
    ]
    store.add(records, [[1.0, 0.0, 0.0, 0.0]])

    checker = FileIntegrityChecker(db_path=tmp_path / "db" / "history.db")
    manager = DocumentManager(store=store, integrity=checker)
    removed = manager.delete("doc-y")
    assert removed == 1
    assert manager.detail("doc-y") == []  # Chroma 照常清空


def test_delete_integrity_cleanup_failure_does_not_abort_chroma_clear(tmp_path, monkeypatch):
    """FIX-1：remove_record 抛异常时，delete 仍清 Chroma（主契约不被 integrity 健康度耦合）。"""
    from rag_mcp_server.libs.loader.file_integrity import FileIntegrityChecker

    store = ChromaVectorStore(
        persist_dir=str(tmp_path / "storage" / "chroma"), collection="test_t3_cleanup_err"
    )
    records = [
        _record("doc-z::0", "doc-z", source_path="z.md", text="Z chunk"),
    ]
    store.add(records, [[1.0, 0.0, 0.0, 0.0]])

    checker = FileIntegrityChecker(db_path=tmp_path / "db" / "history.db")
    # 注入同一 checker，但让 remove_record 抛异常（模拟 SQLite 故障 / DB 损坏）
    manager = DocumentManager(store=store, integrity=checker)

    def _boom(_file_hash):
        raise RuntimeError("db corrupt")

    monkeypatch.setattr(checker, "remove_record", _boom)
    removed = manager.delete("doc-z")
    assert removed == 1
    assert manager.detail("doc-z") == []  # Chroma 照常清空，异常被吞掉


def test_delete_same_file_hash_removes_once(tmp_path):
    """FIX-2：同文件多 chunk 共享 file_hash，remove_record 只调一次（不冗余写 I/O）。"""
    from rag_mcp_server.libs.loader.file_integrity import FileIntegrityChecker

    store = ChromaVectorStore(
        persist_dir=str(tmp_path / "storage" / "chroma"), collection="test_t3_dedup"
    )
    records = [
        _record("doc-w::0", "doc-w", source_path="w.md", text="W 一"),
        _record("doc-w::1", "doc-w", source_path="w.md", text="W 二"),
    ]
    # 两个 chunk 共享同一 file_hash（同文件多 chunk）
    for r in records:
        r.metadata["file_hash"] = "hash-common"
    store.add(records, [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

    checker = FileIntegrityChecker(db_path=tmp_path / "db" / "history.db")
    calls: list[str] = []
    checker.remove_record = lambda h: calls.append(h)  # type: ignore[method-assign]
    manager = DocumentManager(store=store, integrity=checker)
    removed = manager.delete("doc-w")
    assert removed == 2
    assert calls == ["hash-common"]  # 去重后只调一次


def test_empty_store_returns_empty(manager):
    empty = DocumentManager(store=ChromaVectorStore(persist_dir=None, collection="test_docmgr_empty"))
    assert empty.list() == []
    assert empty.stats() == {"documents": 0, "chunks": 0}
    assert empty.delete("anything") == 0


# -- P10.8 ticket 12: stats_detailed ------------------------------------------
def test_stats_detailed_groups_like_list(manager):
    """stats_detailed 与 list 相同的分组/排序，含 chunk_count/source_path。"""
    docs = manager.stats_detailed()
    by_id = {d["document_id"]: d for d in docs}
    assert set(by_id) == {"doc-a", "doc-b"}
    assert by_id["doc-a"]["chunk_count"] == 2
    assert by_id["doc-a"]["source_path"] == "a.md"
    assert by_id["doc-b"]["chunk_count"] == 1


def test_stats_detailed_omits_image_count_without_storage(manager):
    """无 image_storage 注入时，不出现 image_count 键（不声称 0）。"""
    for d in manager.stats_detailed():
        assert "image_count" not in d


def test_stats_detailed_includes_image_count_with_storage(manager, tmp_path):
    """注入 image_storage 时，image_count 出现且 = list_by_document 长度（无图的文档为 0）。"""
    from rag_mcp_server.ingestion.storage.image_storage import ImageStorage

    storage = ImageStorage(
        db_path=tmp_path / "db" / "image_index.db",
        images_root=tmp_path / "images",
    )
    storage.save_image("doc-a_p1_1", b"\x89PNG\r\n\x1a\n" + b"\x00" * 16, mime_type="image/png", document_id="doc-a")
    manager_with_img = DocumentManager(store=manager.store, image_storage=storage)
    by_id = {d["document_id"]: d for d in manager_with_img.stats_detailed()}
    assert by_id["doc-a"]["image_count"] == 1
    assert by_id["doc-b"]["image_count"] == 0


def test_stats_detailed_includes_title_summary_when_present(manager):
    """metadata 有 title/summary 时带上；无则省略。"""
    from rag_mcp_server.libs.vector_store.base import VectorRecord

    store = ChromaVectorStore(persist_dir=None, collection="test_docmgr_detail")
    records = [
        VectorRecord(
            chunk_id="doc-t::0",
            metadata={
                "document_id": "doc-t",
                "source_path": "t.md",
                "chunk_id": "doc-t::0",
                "content_hash": "h",
                "title": "标题",
                "summary": "摘要",
            },
            text="正文",
        ),
        # 无 title/summary 的文档 -> 省略（非空字符串）
        VectorRecord(
            chunk_id="doc-p::0",
            metadata={
                "document_id": "doc-p",
                "source_path": "p.md",
                "chunk_id": "doc-p::0",
                "content_hash": "h",
            },
            text="纯文本",
        ),
    ]
    store.add(records, [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    by_id = {d["document_id"]: d for d in DocumentManager(store=store).stats_detailed()}
    assert by_id["doc-t"]["title"] == "标题"
    assert by_id["doc-t"]["summary"] == "摘要"
    # 「无则省略」：无 title/summary 的文档不出现这两个键
    assert "title" not in by_id["doc-p"]
    assert "summary" not in by_id["doc-p"]


def test_stats_detailed_title_summary_from_first_record_carrying_them(manager):
    """多 chunk 文档中，title/summary 取第一个携带它们的 record（不因同文档无键 record 丢失）。"""
    from rag_mcp_server.libs.vector_store.base import VectorRecord

    store = ChromaVectorStore(persist_dir=None, collection="test_docmgr_detail_first")
    records = [
        # 第一个 chunk 无 title，第二个有 -> 取第二个的
        VectorRecord(
            chunk_id="doc-m::0",
            metadata={
                "document_id": "doc-m",
                "source_path": "m.md",
                "chunk_id": "doc-m::0",
                "content_hash": "h",
            },
            text="第一段",
        ),
        VectorRecord(
            chunk_id="doc-m::1",
            metadata={
                "document_id": "doc-m",
                "source_path": "m.md",
                "chunk_id": "doc-m::1",
                "content_hash": "h",
                "title": "M 标题",
            },
            text="第二段",
        ),
    ]
    store.add(records, [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    by_id = {d["document_id"]: d for d in DocumentManager(store=store).stats_detailed()}
    assert by_id["doc-m"]["title"] == "M 标题"
    assert "summary" not in by_id["doc-m"]


def test_stats_detailed_empty_store_returns_empty(tmp_path):
    empty = DocumentManager(store=ChromaVectorStore(persist_dir=None, collection="test_docmgr_det_empty"))
    assert empty.stats_detailed() == []


def test_stats_detailed_single_get_all_scan(manager):
    """S2 review fix: stats_detailed must do exactly ONE get_all() scan.

    The grouping, chunk counts AND the optional title/summary harvest all come
    from that single pass — no list() + get_all() double scan (CONTEXT
    stats_detailed: 一次 get_all() 扫描同时喂分组与可选键).
    """
    calls = {"get_all": 0}

    class SpyStore:
        def get_all(self):
            calls["get_all"] += 1
            return manager.store.get_all()

        def __getattr__(self, name):
            return getattr(manager.store, name)

    docs = DocumentManager(store=SpyStore()).stats_detailed()
    assert calls["get_all"] == 1
    assert len(docs) == 2
    assert {d["document_id"] for d in docs} == {"doc-a", "doc-b"}


def test_browse_groups_docs_with_chunks_in_one_scan(manager):
    """C6: browse() groups documents AND harvests their chunks in ONE get_all().

    The Dashboard data_browser page must never run N+1 full-store scans (one
    per document, the old list()+detail() shape)."""
    calls = {"get_all": 0}

    class SpyStore:
        def get_all(self):
            calls["get_all"] += 1
            return manager.store.get_all()

        def __getattr__(self, name):
            return getattr(manager.store, name)

    docs = DocumentManager(store=SpyStore()).browse()
    assert calls["get_all"] == 1
    assert len(docs) == 2
    by_id = {d["document_id"]: d for d in docs}
    assert by_id["doc-a"]["chunk_count"] == 2
    assert by_id["doc-a"]["source_path"] == "a.md"
    assert [c["chunk_id"] for c in by_id["doc-a"]["chunks"]] == ["doc-a::0", "doc-a::1"]
    assert by_id["doc-b"]["chunk_count"] == 1
    assert by_id["doc-b"]["chunks"][0]["text"] == "B 唯一 chunk"


def test_browse_empty_store_returns_empty(tmp_path):
    empty = DocumentManager(store=ChromaVectorStore(persist_dir=None, collection="test_docmgr_br_empty"))
    assert empty.browse() == []
