"""Tests for the on_progress callback (ticket 06, P10.5).

Grill-with-docs contract (Q1-Q7):
- stage set = load/split/transform/embed/upsert/skip (mirrors trace stage
  names exactly); transform fires only when transforms are configured
  (default [] -> no transform stage, matching the 4-stage trace shape)
- current/total are FILE-level: total = len(files), current = 1-based file
  index; every stage of the same file shares the same (current, total)
- empty documents (0 chunks) and fully-idempotent re-ingests still fire
  embed/upsert (mirroring _upsert_chunks' empty/idempotent branches) so the
  progress bar moves monotonically and stays aligned with the trace
- a document whose load fails fires ("skip", cur, total) (it was attempted);
  resolve-level unsupported files are NOT in the queue and fire nothing
- callback exceptions are swallowed (log warning), never break the ingest
- on_progress=None leaves behaviour completely unchanged
"""

from __future__ import annotations

import pytest

from rag_mcp_server.ingestion.pipeline import IngestPipeline
from rag_mcp_server.ingestion.transform import ChunkRefiner
from rag_mcp_server.libs.embedding.fake import FakeEmbedding
from rag_mcp_server.libs.vector_store.chroma_store import ChromaVectorStore


@pytest.fixture(autouse=True)
def _isolate_trace(monkeypatch):
    """Isolate trace writes from the repo logs/ (review F2; same as pipeline tests)."""
    import rag_mcp_server.config.settings as settings_mod
    from rag_mcp_server.config.settings import DEFAULTS

    def fake_load_settings():
        data = {k: dict(v) if isinstance(v, dict) else v for k, v in DEFAULTS.items()}
        data["trace"] = {"enabled": False, "dir": "logs"}
        return data

    monkeypatch.setattr(settings_mod, "load_settings", fake_load_settings)


@pytest.fixture
def pipeline(tmp_path):
    embedding = FakeEmbedding(dim=4)
    store = ChromaVectorStore(
        persist_dir=str(tmp_path / "storage" / "chroma"), collection="test_progress"
    )
    return IngestPipeline(
        embedding=embedding, store=store, index_dir=tmp_path / "storage" / "index",
        collection="test_progress",  # ADR 0016
    )


@pytest.fixture
def source_dir(tmp_path):
    src = tmp_path / "source"
    src.mkdir()
    return src


def _file_stages_of(events, cur):
    """The (stage, total) pairs recorded for the file with 1-based index ``cur``."""
    return [(stage, total) for stage, c, total in events if c == cur]


def test_on_progress_fires_all_six_stages_with_file_level_counts(pipeline, source_dir):
    # 2 files: one clean txt + one corrupt pdf. The txt walks the full
    # 5-stage happy path; the corrupt pdf is attempted -> load fails -> skip.
    (source_dir / "notes.txt").write_text("第一段。\n\n第二段。", encoding="utf-8")
    (source_dir / "broken.pdf").write_bytes(b"%PDF-1.4 fake\n")

    events: list[tuple[str, int, int]] = []
    summary = pipeline.ingest(source_dir, base=source_dir, on_progress=lambda s, c, t: events.append((s, c, t)))

    assert summary.total_files == 1
    assert summary.total_skipped == 1

    # file-level counts: total == 2 files, 1-based current 1 and 2. The txt
    # and the broken pdf sort alphabetically (broken.pdf < notes.txt), so we
    # assert the composition per current rather than which file is which.
    assert sorted(_file_stages_of(events, 1)) == [
        ("load", 2),
        ("skip", 2),
    ]
    assert sorted(_file_stages_of(events, 2)) == [
        ("embed", 2),
        ("load", 2),
        ("split", 2),
        ("upsert", 2),
    ]
    # exactly two distinct currents, total == 2 throughout
    assert {c for _, c, _ in events} == {1, 2}
    assert all(t == 2 for _, _, t in events)


def test_on_progress_skipped_resolve_files_fire_nothing(pipeline, source_dir):
    # an unsupported file is filtered at resolve time, never enters the
    # queue -> no progress event, but still reported in summary.skipped
    (source_dir / "readme.docx").write_bytes(b"not a docx")
    (source_dir / "notes.txt").write_text("正文。", encoding="utf-8")

    events: list[tuple[str, int, int]] = []
    summary = pipeline.ingest(source_dir, base=source_dir, on_progress=lambda s, c, t: events.append((s, c, t)))

    assert summary.total_files == 1
    assert summary.total_skipped == 1  # docx collected by resolve
    assert summary.skipped[0]["path"].endswith(".docx")
    assert _file_stages_of(events, 1) == [("load", 1), ("split", 1), ("embed", 1), ("upsert", 1)]


def test_on_progress_transform_stage_fires_only_when_transforms_configured(tmp_path):
    src = tmp_path / "source"
    src.mkdir()
    (src / "notes.txt").write_text("  有噪声的文本  ", encoding="utf-8")
    embedding = FakeEmbedding(dim=4)
    store = ChromaVectorStore(
        persist_dir=str(tmp_path / "storage" / "chroma"), collection="test_progress_tr"
    )
    pipe = IngestPipeline(
        embedding=embedding,
        store=store,
        index_dir=tmp_path / "storage" / "index",
        transforms=[ChunkRefiner()],
    )

    events: list[tuple[str, int, int]] = []
    pipe.ingest(src, base=src, on_progress=lambda s, c, t: events.append((s, c, t)))

    assert [stage for stage, _, _ in events] == ["load", "split", "transform", "embed", "upsert"]
    # transform shares the same file-level (current, total)
    assert _file_stages_of(events, 1) == [
        ("load", 1),
        ("split", 1),
        ("transform", 1),
        ("embed", 1),
        ("upsert", 1),
    ]


def test_on_progress_empty_and_idempotent_still_fire_embed_upsert(pipeline, source_dir):
    # empty document (0 chunks) + a re-ingest of identical content both walk
    # the full stage shape: embed/upsert fire even though nothing was added
    (source_dir / "blank.txt").write_text("", encoding="utf-8")

    events: list[tuple[str, int, int]] = []
    first = pipeline.ingest(source_dir, base=source_dir, on_progress=lambda s, c, t: events.append((s, c, t)))
    assert first.total_chunks == 0
    assert _file_stages_of(events, 1) == [
        ("load", 1),
        ("split", 1),
        ("embed", 1),
        ("upsert", 1),
    ]

    # idempotent re-ingest of an already-indexed file
    (source_dir / "blank.txt").write_text("唯一段落。", encoding="utf-8")
    events2: list[tuple[str, int, int]] = []
    second = pipeline.ingest(source_dir, base=source_dir, on_progress=lambda s, c, t: events2.append((s, c, t)))
    assert second.total_chunks == 1
    assert _file_stages_of(events2, 1) == [
        ("load", 1),
        ("split", 1),
        ("embed", 1),
        ("upsert", 1),
    ]

    events3: list[tuple[str, int, int]] = []
    third = pipeline.ingest(source_dir, base=source_dir, on_progress=lambda s, c, t: events3.append((s, c, t)))
    assert third.total_chunks == 0  # identical content skipped
    assert _file_stages_of(events3, 1) == [
        ("load", 1),
        ("split", 1),
        ("embed", 1),
        ("upsert", 1),
    ]


def test_on_progress_none_leaves_behaviour_unchanged(pipeline, source_dir):
    # on_progress default None must be a total no-op (existing callers/tests)
    (source_dir / "notes.txt").write_text("第一段。\n\n第二段。", encoding="utf-8")

    summary = pipeline.ingest(source_dir, base=source_dir)
    assert summary.total_files == 1
    assert summary.total_chunks == 2
    records = pipeline.store.get_by_ids(["test_progress::notes-txt::0", "test_progress::notes-txt::1"])
    assert len(records) == 2


def test_on_progress_callback_exception_is_swallowed(pipeline, source_dir, caplog):
    # a broken callback must not break the ingest (progress is observation)
    (source_dir / "notes.txt").write_text("正文内容。", encoding="utf-8")

    def boom(stage, current, total):
        raise RuntimeError("progress bar broke")

    with caplog.at_level("WARNING", logger="rag_mcp_server.ingestion.pipeline"):
        summary = pipeline.ingest(source_dir, base=source_dir, on_progress=boom)

    assert summary.total_files == 1
    assert summary.total_chunks == 1
    # the record still landed
    assert len(pipeline.store.get_by_ids(["test_progress::notes-txt::0"])) == 1
    assert any("on_progress" in r.message for r in caplog.records)
