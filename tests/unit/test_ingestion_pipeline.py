"""Tests for the ingestion pipeline seam.

Red/green contract (docs/spec-p9-ingest.md, task 6, tickets 06-T2):
- pipeline entry takes a source path and does resolve -> load -> chunk ->
  embed -> diff -> add -> JSONL evidence
- txt/md/pdf all flow through the same pipeline into the vector store
- re-ingesting identical content is idempotent (all skipped, no dup chunks);
  new content only adds new chunks
- a corrupt/textless PDF is collected as skipped (reason recorded), it does
  not break ingestion of the other files

The only seam under test is the pipeline public entry; tests inject
FakeEmbedding + a temporary Chroma (tmp_path), never hitting Ollama.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3

import pytest

from rag_mcp_server.ingestion.pipeline import IngestPipeline
from rag_mcp_server.ingestion.jsonl_store import read_chunks
from rag_mcp_server.ingestion.transform import ChunkRefiner, MetadataEnricher
from rag_mcp_server.libs.embedding.fake import FakeEmbedding
from rag_mcp_server.libs.vector_store.chroma_store import ChromaVectorStore

from tests.fixtures.make_pdf import empty_content_pdf, minimal_pdf


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@pytest.fixture(autouse=True)
def _isolate_trace(monkeypatch):
    """Isolate trace writes from the repo logs/ (review F2).

    These pipeline-behavior tests call pipeline.ingest without a trace_dir; if
    trace stayed enabled they would write real ingestion traces into the repo
    logs/ dir on every full test run. They are not trace tests (US 14: tests
    isolate the trace dir), so disable trace for the whole module.
    """
    import rag_mcp_server.config.settings as settings_mod
    from rag_mcp_server.config.settings import DEFAULTS

    def fake_load_settings():
        data = {k: dict(v) if isinstance(v, dict) else v for k, v in DEFAULTS.items()}
        data["trace"] = {"enabled": False, "dir": "logs"}
        return data

    monkeypatch.setattr(settings_mod, "load_settings", fake_load_settings)


@pytest.fixture
def pipeline(tmp_path):
    """Pipeline backed by FakeEmbedding + a temp Chroma (offline).

    The source tree, Chroma persist dir and index dir are kept in separate
    subdirectories so ingestion never scans its own storage files.
    """
    embedding = FakeEmbedding(dim=4)
    store = ChromaVectorStore(
        persist_dir=str(tmp_path / "storage" / "chroma"), collection="test_chunks"
    )
    return IngestPipeline(
        embedding=embedding, store=store, index_dir=tmp_path / "storage" / "index",
        collection="test_chunks",
    )


@pytest.fixture
def source_dir(tmp_path):
    """A clean source tree for ingestion, isolated from storage."""
    src = tmp_path / "source"
    src.mkdir()
    return src


def test_pipeline_ingests_txt_and_pdf_into_store_and_jsonl(pipeline, source_dir, tmp_path):
    (source_dir / "notes.txt").write_text("第一段。\n\n第二段。", encoding="utf-8")
    (source_dir / "docs").mkdir()
    (source_dir / "docs" / "guide.pdf").write_bytes(minimal_pdf("Hello RAG PDF"))

    summary = pipeline.ingest(source_dir, base=source_dir)

    assert summary.total_files == 2
    assert summary.total_skipped == 0
    # both txt chunks + the pdf chunk are in the vector store
    records = pipeline.store.get_by_ids(["test_chunks::notes-txt::0", "test_chunks::notes-txt::1", "test_chunks::docs-guide-pdf::0"])
    assert len(records) == 3
    # four-key metadata present (vector store enforces this on add)
    assert records[0].metadata["document_id"] == "test_chunks::notes-txt"
    assert records[0].metadata["content_hash"] == content_hash("第一段。")
    # jsonl evidence written
    chunks = read_chunks(tmp_path / "storage" / "index" / "test_chunks" / "chunks.jsonl")
    assert len(chunks) == 3


def test_pipeline_ingest_is_idempotent_on_identical_reingest(pipeline, source_dir):
    (source_dir / "notes.txt").write_text("第一段。\n\n第二段。", encoding="utf-8")

    first = pipeline.ingest(source_dir, base=source_dir)
    assert first.total_chunks == 2

    second = pipeline.ingest(source_dir, base=source_dir)
    assert second.total_files == 1
    assert second.total_skipped == 0
    assert second.total_chunks == 0  # all skipped by content_hash

    records = pipeline.store.get_by_ids(["test_chunks::notes-txt::0", "test_chunks::notes-txt::1"])
    assert len(records) == 2  # no duplicates


def test_pipeline_new_content_only_adds_new_chunk(pipeline, source_dir):
    source = source_dir / "notes.txt"
    source.write_text("第一段。\n\n第二段。", encoding="utf-8")
    pipeline.ingest(source_dir, base=source_dir)

    # content grows by one paragraph
    source.write_text("第一段。\n\n第二段。\n\n新增段。", encoding="utf-8")
    second = pipeline.ingest(source_dir, base=source_dir)

    assert second.total_chunks == 1  # only the new paragraph added
    records = pipeline.store.get_by_ids(["test_chunks::notes-txt::0", "test_chunks::notes-txt::1", "test_chunks::notes-txt::2"])
    assert len(records) == 3


def test_pipeline_collects_corrupt_pdf_as_skipped(pipeline, source_dir):
    (source_dir / "notes.txt").write_text("正文内容。", encoding="utf-8")
    (source_dir / "broken.pdf").write_bytes(b"%PDF-1.4 fake\n")

    summary = pipeline.ingest(source_dir, base=source_dir)

    assert summary.total_files == 1
    assert summary.total_skipped == 1
    assert "pdf parse failed" in summary.skipped[0]["reason"]
    # the good file still landed
    assert len(pipeline.store.get_by_ids(["test_chunks::notes-txt::0"])) == 1


def test_pipeline_collects_textless_pdf_as_skipped(pipeline, source_dir):
    (source_dir / "notes.txt").write_text("正文内容。", encoding="utf-8")
    (source_dir / "scanned.pdf").write_bytes(empty_content_pdf())

    summary = pipeline.ingest(source_dir, base=source_dir)

    assert summary.total_files == 1
    assert summary.total_skipped == 1
    assert "no extractable text" in summary.skipped[0]["reason"]
    assert len(pipeline.store.get_by_ids(["test_chunks::notes-txt::0"])) == 1


def test_pipeline_report_written(pipeline, source_dir, tmp_path):
    (source_dir / "notes.txt").write_text("正文。", encoding="utf-8")
    pipeline.ingest(source_dir, base=source_dir)

    report_path = tmp_path / "storage" / "index" / "test_chunks" / "ingestion_report.json"
    assert report_path.is_file()
    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["total_files"] == 1
    assert data["total_chunks"] == 1


def test_pipeline_collects_non_utf8_txt_as_skipped_not_fatal(pipeline, source_dir):
    # a non-UTF-8 txt must not crash the whole ingest: it is collected as
    # skipped and the other files still land (spec: 损坏文件不打断摄取)
    (source_dir / "latin1.txt").write_bytes("héllo".encode("latin-1"))
    (source_dir / "notes.txt").write_text("正文内容。", encoding="utf-8")

    summary = pipeline.ingest(source_dir, base=source_dir)

    assert summary.total_files == 1
    assert summary.total_skipped == 1
    assert "cannot read" in summary.skipped[0]["reason"]
    assert len(pipeline.store.get_by_ids(["test_chunks::notes-txt::0"])) == 1


def test_pipeline_changed_chunk_content_is_replaced_not_stale(pipeline, source_dir):
    # a chunk_id that already exists with *different* content (partial edit,
    # order shift) must be re-stored with the new text — Chroma's silent
    # keep-first on re-add must never leave stale content indexed
    source = source_dir / "notes.txt"
    source.write_text("第一段。", encoding="utf-8")
    pipeline.ingest(source_dir, base=source_dir)
    assert pipeline.store.get_by_ids(["test_chunks::notes-txt::0"])[0].text == "第一段。"

    # same id, new content: the second ingest must overwrite, not keep old
    source.write_text("被编辑后的内容。", encoding="utf-8")
    second = pipeline.ingest(source_dir, base=source_dir)

    assert second.total_chunks == 1
    records = pipeline.store.get_by_ids(["test_chunks::notes-txt::0"])
    assert len(records) == 1
    assert records[0].text == "被编辑后的内容。"
    assert records[0].metadata["content_hash"] == content_hash("被编辑后的内容。")
    # JSONL evidence stays aligned with the store (no stale text kept)
    chunks = read_chunks(pipeline.index_dir / "test_chunks" / "chunks.jsonl")
    assert chunks[0].text == "被编辑后的内容。"


def test_pipeline_duplicate_text_across_docs_is_indexed_per_doc(pipeline, source_dir):
    # the dedup key is the chunk_id, not the text: two documents sharing the
    # same paragraph text must BOTH be indexed (a global hash set would skip
    # the second doc entirely)
    (source_dir / "a.txt").write_text("相同段落。", encoding="utf-8")
    (source_dir / "b.txt").write_text("相同段落。", encoding="utf-8")

    summary = pipeline.ingest(source_dir, base=source_dir)

    assert summary.total_chunks == 2
    records = pipeline.store.get_by_ids(["test_chunks::a-txt::0", "test_chunks::b-txt::0"])
    assert len(records) == 2
    assert {r.chunk_id for r in records} == {"test_chunks::a-txt::0", "test_chunks::b-txt::0"}


# -- P10.4 transform wiring -------------------------------------------------
def test_transforms_inject_metadata_into_store_records(tmp_path):
    """With transforms wired, the store records carry the whitelisted keys
    (title/summary/tags/refined_by/enriched_by) on top of the 4 mandatory ones."""
    src = tmp_path / "source"
    src.mkdir()
    (src / "notes.txt").write_text(
        "# RAG 检索原理\n\nRAG 通过检索增强生成，结合向量检索与生成模型。",
        encoding="utf-8",
    )
    embedding = FakeEmbedding(dim=4)
    store = ChromaVectorStore(
        persist_dir=str(tmp_path / "storage" / "chroma"), collection="test_transform"
    )
    pipe = IngestPipeline(
        embedding=embedding,
        store=store,
        index_dir=tmp_path / "storage" / "index",
        transforms=[ChunkRefiner(), MetadataEnricher()],
        collection="test_transform",
    )

    summary = pipe.ingest(src, base=src)
    assert summary.total_chunks >= 1

    records = store.get_by_ids([c.chunk_id for c in read_chunks(tmp_path / "storage" / "index" / "test_transform" / "chunks.jsonl")])
    assert records
    md = records[0].metadata
    # 4 mandatory keys present
    for key in ("document_id", "source_path", "chunk_id", "content_hash"):
        assert key in md
    # whitelisted transform keys forwarded
    assert md["refined_by"] == "rule"
    assert md["enriched_by"] == "rule"
    assert md["title"]
    assert md["summary"]
    assert md["tags"]
    # file_type is NOT forwarded (stays dropped)
    assert "file_type" not in md


def test_default_no_transforms_record_has_exact_four_keys(tmp_path):
    """Default transforms=[] keeps the existing behaviour: record metadata is
    exactly the four mandatory keys (nothing extra forwarded)."""
    src = tmp_path / "source"
    src.mkdir()
    (src / "notes.txt").write_text("第一段。\n\n第二段。", encoding="utf-8")
    embedding = FakeEmbedding(dim=4)
    store = ChromaVectorStore(
        persist_dir=str(tmp_path / "storage" / "chroma"), collection="test_notransform"
    )
    pipe = IngestPipeline(
        embedding=embedding, store=store, index_dir=tmp_path / "storage" / "index",
        collection="test_notransform",
    )

    pipe.ingest(src, base=src)

    records = store.get_by_ids(["test_notransform::notes-txt::0", "test_notransform::notes-txt::1"])
    assert len(records) == 2
    for r in records:
        assert set(r.metadata.keys()) == {"document_id", "source_path", "chunk_id", "content_hash"}


def test_transform_changed_text_recomputes_content_hash(tmp_path):
    """A ChunkRefiner that changes text must change content_hash -> re-embed
    on re-ingest (ADR 0011: content_hash reflects the embedded text)."""
    src = tmp_path / "source"
    src.mkdir()
    (src / "notes.txt").write_text("  有噪声的文本  ", encoding="utf-8")
    embedding = FakeEmbedding(dim=4)
    store = ChromaVectorStore(
        persist_dir=str(tmp_path / "storage" / "chroma"), collection="test_hash"
    )
    pipe = IngestPipeline(
        embedding=embedding,
        store=store,
        index_dir=tmp_path / "storage" / "index",
        transforms=[ChunkRefiner()],
        collection="test_hash",
    )

    pipe.ingest(src, base=src)
    record = store.get_by_ids(["test_hash::notes-txt::0"])[0]
    # refined text (whitespace trimmed) has a different hash than the raw text
    assert record.text == "有噪声的文本"
    assert record.metadata["content_hash"] != content_hash("  有噪声的文本  ")


# -- P10.5 T1 FileIntegrity 增量跳过 -----------------------------------------
def test_integrity_skip_on_unchanged_reingest(tmp_path):
    """Injecting a FileIntegrityChecker: a second ingest of unchanged files is
    skipped at the file level (no re-load/embed/upsert, store untouched, and
    on_progress does NOT fire load/split/embed/upsert for the skipped file)."""
    from rag_mcp_server.libs.loader.file_integrity import FileIntegrityChecker

    src = tmp_path / "source"
    src.mkdir()
    (src / "notes.txt").write_text("第一段。\n\n第二段。", encoding="utf-8")
    embedding = FakeEmbedding(dim=4)
    store = ChromaVectorStore(
        persist_dir=str(tmp_path / "storage" / "chroma"), collection="test_integrity"
    )
    checker = FileIntegrityChecker(db_path=tmp_path / "db" / "history.db")
    pipe = IngestPipeline(
        embedding=embedding,
        store=store,
        index_dir=tmp_path / "storage" / "index",
        integrity=checker,
        collection="test_integrity",
    )

    first = pipe.ingest(src, base=src)
    assert first.total_files == 1
    assert first.total_chunks == 2

    events: list[tuple[str, int, int]] = []
    second = pipe.ingest(src, base=src, on_progress=lambda s, c, t: events.append((s, c, t)))

    # the unchanged file was skipped at the integrity level
    assert second.total_files == 0
    assert second.total_skipped == 1
    assert "unchanged (integrity)" in second.skipped[0]["reason"]
    # store unchanged (no dup chunks)
    assert len(store.get_by_ids(["test_integrity::notes-txt::0", "test_integrity::notes-txt::1"])) == 2
    # on_progress fired nothing for the skipped file
    assert events == []


def test_integrity_record_metadata_carries_file_hash(tmp_path):
    """T3：integrity 注入时，VectorRecord.metadata 带 file_hash 键（delete 联动用）。

    只有 checker 注入时写该可选键；delete 靠它 remove_record（无需 base/读文件反推）。
    """
    from rag_mcp_server.libs.loader.file_integrity import FileIntegrityChecker

    src = tmp_path / "source"
    src.mkdir()
    (src / "notes.txt").write_text("第一段。\n\n第二段。", encoding="utf-8")
    embedding = FakeEmbedding(dim=4)
    store = ChromaVectorStore(
        persist_dir=str(tmp_path / "storage" / "chroma"), collection="test_t3_filehash"
    )
    checker = FileIntegrityChecker(db_path=tmp_path / "db" / "history.db")
    pipe = IngestPipeline(
        embedding=embedding,
        store=store,
        index_dir=tmp_path / "storage" / "index",
        integrity=checker,
        collection="test_t3_filehash",
    )

    pipe.ingest(src, base=src)

    records = store.get_by_ids(["test_t3_filehash::notes-txt::0", "test_t3_filehash::notes-txt::1"])
    assert len(records) == 2
    for r in records:
        assert "file_hash" in r.metadata
        assert isinstance(r.metadata["file_hash"], str)
        assert len(r.metadata["file_hash"]) == 64  # sha256 hexdigest
    # 同一文件的多个 chunk 共享同一 file_hash
    assert records[0].metadata["file_hash"] == records[1].metadata["file_hash"]


def test_integrity_changed_file_is_reingested(tmp_path):
    """A file whose content changed since the last success is NOT skipped: it
    is re-ingested and the record is refreshed."""
    from rag_mcp_server.libs.loader.file_integrity import FileIntegrityChecker

    src = tmp_path / "source"
    src.mkdir()
    source = src / "notes.txt"
    source.write_text("第一段。\n\n第二段。", encoding="utf-8")
    embedding = FakeEmbedding(dim=4)
    store = ChromaVectorStore(
        persist_dir=str(tmp_path / "storage" / "chroma"), collection="test_integrity_chg"
    )
    checker = FileIntegrityChecker(db_path=tmp_path / "db" / "history.db")
    pipe = IngestPipeline(
        embedding=embedding,
        store=store,
        index_dir=tmp_path / "storage" / "index",
        integrity=checker,
        collection="test_integrity_chg",
    )

    pipe.ingest(src, base=src)
    # content grows: different file hash -> not skipped
    source.write_text("第一段。\n\n第二段。\n\n新增段。", encoding="utf-8")
    second = pipe.ingest(src, base=src)

    assert second.total_files == 1
    assert second.total_skipped == 0
    assert second.total_chunks == 1  # only the new paragraph added
    records = store.get_by_ids(["test_integrity_chg::notes-txt::0", "test_integrity_chg::notes-txt::1", "test_integrity_chg::notes-txt::2"])
    assert len(records) == 3


def test_integrity_load_failure_marked_failed_retries_next_time(tmp_path):
    """A load failure marks the file failed, so the next ingest retries it
    instead of skipping (failed != success)."""
    from rag_mcp_server.libs.loader.file_integrity import FileIntegrityChecker

    src = tmp_path / "source"
    src.mkdir()
    broken = src / "broken.pdf"
    broken.write_bytes(b"%PDF-1.4 fake\n")
    (src / "notes.txt").write_text("正文内容。", encoding="utf-8")
    embedding = FakeEmbedding(dim=4)
    store = ChromaVectorStore(
        persist_dir=str(tmp_path / "storage" / "chroma"), collection="test_integrity_fail"
    )
    checker = FileIntegrityChecker(db_path=tmp_path / "db" / "history.db")
    pipe = IngestPipeline(
        embedding=embedding,
        store=store,
        index_dir=tmp_path / "storage" / "index",
        integrity=checker,
        collection="test_integrity_fail",
    )

    first = pipe.ingest(src, base=src)
    assert first.total_files == 1  # only notes.txt
    assert first.total_skipped == 1  # broken.pdf load-failed

    # second ingest: notes.txt is integrity-skipped (success, unchanged);
    # broken.pdf is NOT skipped by integrity (failed != success) and is
    # retried — it fails again with the load reason, not "unchanged"
    second = pipe.ingest(src, base=src)
    assert second.total_files == 0
    assert second.total_skipped == 2  # notes.txt integrity-skip + broken.pdf retry
    reasons = [s["reason"] for s in second.skipped]
    assert "unchanged (integrity)" in reasons
    assert any("pdf parse failed" in r for r in reasons)


def test_integrity_mark_success_records_added_chunks_not_chunked(tmp_path):
    """chunk_count 语义（修复 Bug 2/4）：mark_success 记「实际入库 chunk 数」。

    文件部分更新时（内容增长），chunker 切出的 chunk 数（doc_chunks）与本次
    实际新增/变更的 chunk 数（added）不同。integrity 记录必须镜像 store 真实
    状态 —— 记 added，与 summary.total_chunks 语义一致 —— 而不是 doc_chunks。
    """
    from rag_mcp_server.libs.loader.file_integrity import FileIntegrityChecker

    src = tmp_path / "source"
    src.mkdir()
    source = src / "notes.txt"
    source.write_text("第一段。", encoding="utf-8")
    embedding = FakeEmbedding(dim=4)
    store = ChromaVectorStore(
        persist_dir=str(tmp_path / "storage" / "chroma"), collection="test_integrity_cnt"
    )
    checker = FileIntegrityChecker(db_path=tmp_path / "db" / "history.db")
    pipe = IngestPipeline(
        embedding=embedding,
        store=store,
        index_dir=tmp_path / "storage" / "index",
        integrity=checker,
        collection="test_integrity_cnt",
    )

    first = pipe.ingest(src, base=src)
    assert first.total_chunks == 1

    # 内容增长：chunker 现在切出 2 个 chunk，但只有新增的第 2 个进 store
    source.write_text("第一段。\n\n第二段。", encoding="utf-8")
    second = pipe.ingest(src, base=src)
    assert second.total_chunks == 1  # only the new paragraph

    # file_hash 含文件身份（relative path）+ 内容（F1 修复后的语义）
    relative = "notes.txt"
    file_hash = hashlib.sha256(
        relative.encode("utf-8") + b"\0" + source.read_bytes()
    ).hexdigest()
    conn = sqlite3.connect(str(tmp_path / "db" / "history.db"))
    try:
        row = conn.execute(
            "SELECT chunk_count FROM file_integrity WHERE file_hash = ?", (file_hash,)
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    # chunk_count == 本次 added 数（1），不是 chunker 切出的 doc_chunks 数（2）
    assert row[0] == 1


def test_integrity_unexpected_error_marks_failed_and_reingests(tmp_path):
    """非 DocumentLoadError 异常（修复 Bug 3）：mark_failed + 重抛。

    若 transform 等步骤抛异常，旧的「success」记录必须被 mark_failed 覆盖，
    否则下次摄取 should_skip 命中过期记录、文件被错误跳过、新内容永不入索引。
    """
    import pytest as _pytest

    from rag_mcp_server.libs.loader.file_integrity import FileIntegrityChecker
    from rag_mcp_server.libs.llm.fake import FakeLLM

    class BoomRefiner(ChunkRefiner):
        """A ChunkRefiner whose transform() raises (transform-level, not per-chunk)."""

        def transform(self, chunks, trace=None):
            raise RuntimeError("transform exploded")

    src = tmp_path / "source"
    src.mkdir()
    (src / "notes.txt").write_text("第一段。\n\n第二段。", encoding="utf-8")
    embedding = FakeEmbedding(dim=4)
    store = ChromaVectorStore(
        persist_dir=str(tmp_path / "storage" / "chroma"), collection="test_integrity_bug"
    )
    checker = FileIntegrityChecker(db_path=tmp_path / "db" / "history.db")
    pipe = IngestPipeline(
        embedding=embedding,
        store=store,
        index_dir=tmp_path / "storage" / "index",
        integrity=checker,
        transforms=[BoomRefiner(llm=FakeLLM(), use_llm=False)],
        collection="test_integrity_bug",
    )

    # 第一次 ingest：transform 异常冒泡 -> mark_failed（不是残留 success）
    with _pytest.raises(RuntimeError):
        pipe.ingest(src, base=src)

    # 关闭 boom transform：同一 checker 再次摄取必须重摄（failed != success）
    pipe2 = IngestPipeline(
        embedding=embedding,
        store=store,
        index_dir=tmp_path / "storage" / "index",
        integrity=checker,
        collection="test_integrity_bug",
    )
    retry = pipe2.ingest(src, base=src)
    assert retry.total_files == 1
    assert retry.total_chunks == 2
    assert len(store.get_by_ids(["test_integrity_bug::notes-txt::0", "test_integrity_bug::notes-txt::1"])) == 2


def test_integrity_same_content_different_paths_both_ingested(tmp_path):
    """F1 回归：同内容不同路径的两个文件必须都入库。

    旧实现 file_hash 只有内容维度（sha256(read_bytes)），同目录两个内容相同的
    文件会在首次摄取时让第二个被误判「未变更」跳过——chunk 永不到达 store 且被
    记为 unchanged (integrity) 掩盖丢数据。修复后 file_hash 含文件身份
    （relative path），两文件各自入库、互不覆盖。
    """
    from rag_mcp_server.libs.loader.file_integrity import FileIntegrityChecker

    src = tmp_path / "source"
    src.mkdir()
    (src / "a.txt").write_text("相同段落。", encoding="utf-8")
    (src / "b.txt").write_text("相同段落。", encoding="utf-8")
    embedding = FakeEmbedding(dim=4)
    store = ChromaVectorStore(
        persist_dir=str(tmp_path / "storage" / "chroma"), collection="test_integrity_same"
    )
    checker = FileIntegrityChecker(db_path=tmp_path / "db" / "history.db")
    pipe = IngestPipeline(
        embedding=embedding,
        store=store,
        index_dir=tmp_path / "storage" / "index",
        integrity=checker,
        collection="test_integrity_same",
    )

    summary = pipe.ingest(src, base=src)

    assert summary.total_files == 2
    assert summary.total_skipped == 0  # 无 unchanged (integrity) 误跳
    assert summary.total_chunks == 2
    records = store.get_by_ids(["test_integrity_same::a-txt::0", "test_integrity_same::b-txt::0"])
    assert len(records) == 2  # 两个文档都入库，互不覆盖


def test_integrity_hash_io_failure_does_not_abort_ingest(tmp_path, monkeypatch):
    """F2 回归：file_hash 计算（read_bytes）I/O 失败不得中止整个摄取。

    旧实现 read_bytes() 在 try 外，文件不可读（权限/删除）会抛 OSError 直接崩掉
    整次 ingest；注入 checker 前同场景会由 load_document 转 DocumentLoadError
    优雅跳过。修复后 hash 读失败 -> file_hash=None -> 走正常 load 流程，
    load_document 的 read_text 失败转 DocumentLoadError -> skip，其余文件仍入库。
    """
    import pathlib

    from rag_mcp_server.libs.loader.file_integrity import FileIntegrityChecker

    src = tmp_path / "source"
    src.mkdir()
    (src / "unreadable.txt").write_text("无法读取", encoding="utf-8")
    (src / "notes.txt").write_text("正文内容。", encoding="utf-8")
    embedding = FakeEmbedding(dim=4)
    store = ChromaVectorStore(
        persist_dir=str(tmp_path / "storage" / "chroma"), collection="test_integrity_io"
    )
    checker = FileIntegrityChecker(db_path=tmp_path / "db" / "history.db")
    pipe = IngestPipeline(
        embedding=embedding,
        store=store,
        index_dir=tmp_path / "storage" / "index",
        integrity=checker,
        collection="test_integrity_io",
    )

    def _boom(path_method):
        def wrapper(self, *args, **kwargs):
            if self.name == "unreadable.txt":
                raise OSError("cannot read file")
            return path_method(self, *args, **kwargs)

        return wrapper

    monkeypatch.setattr(pathlib.Path, "read_bytes", _boom(pathlib.Path.read_bytes))
    monkeypatch.setattr(pathlib.Path, "read_text", _boom(pathlib.Path.read_text))

    summary = pipe.ingest(src, base=src)

    assert summary.total_files == 1  # notes.txt 入库
    assert summary.total_skipped == 1  # unreadable.txt 走 skip，不崩
    assert "cannot read" in summary.skipped[0]["reason"]
    assert len(store.get_by_ids(["test_integrity_io::notes-txt::0"])) == 1
