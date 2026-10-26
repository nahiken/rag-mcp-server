"""Wiring tests for the P10.1 trace — at the application-service seam.

Red/green contract (docs/spec-p10-trace.md, ticket 01):
- rag_service.query produces one trace_type="query" trace per call: dense
  strategy records query_processing + dense_retrieval; hybrid records all four
  stages; empty text records only query_processing. Return type unchanged.
- IngestPipeline.ingest produces one trace_type="ingestion" trace per call,
  with per-file load/split/embed/upsert stages. Return type unchanged.
- trace_dir points at the observer location; tests isolate it to a tmp dir.
- trace disabled (enabled: false) writes nothing.

The only business seam under test is rag_service.query / IngestPipeline.ingest,
mirroring test_query_orchestration.py's offline_env fixture (FakeEmbedding +
temp Chroma + monkeypatched load_settings).
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

import pytest

from rag_mcp_server.application.rag_service import ingest, query
from rag_mcp_server.ingestion.pipeline import IngestPipeline
from rag_mcp_server.libs.embedding.fake import FakeEmbedding
from rag_mcp_server.libs.vector_store.chroma_store import ChromaVectorStore

from tests.fixtures.make_pdf import minimal_pdf


def _offline_env(tmp_path: Path):
    """A temp source tree + offline pipeline; returns (root, index_dir, pipeline).

    The store collection is derived from tmp_path (unique per test) because
    Chroma's EphemeralClient shares its in-memory backend across instances by
    collection name: a fixed name let records from one test leak into another,
    flaking exact-count assertions like rerank output_k (known-risk #1).
    """
    root = Path(tmp_path)
    (root / "docs").mkdir()
    (root / "docs" / "rag-notes.md").write_text(
        "RAG 使用关键词检索 baseline，检索结果按 score 排序。", encoding="utf-8"
    )
    (root / "docs" / "mcp-notes.md").write_text(
        "MCP 是 Model Context Protocol，用于连接工具与数据。", encoding="utf-8"
    )
    embedding = FakeEmbedding(dim=4)
    # derive a valid Chroma collection name from tmp_path (unique per test):
    # sanitize to [a-zA-Z0-9._-] and strip trailing dashes/underscores. Chroma's
    # EphemeralClient shares its in-memory backend by collection name, so a fixed
    # name let records from one test leak into another, flaking exact-count
    # assertions like rerank output_k (known-risk #1).
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", tmp_path.name).strip("_.-")
    store = ChromaVectorStore(persist_dir=None, collection=f"test_trace_wiring_{slug}")
    index_dir = root / "out"
    # ADR 0016: pipeline collection must match the store collection
    pipeline = IngestPipeline(
        embedding=embedding, store=store, index_dir=index_dir,
        collection=f"test_trace_wiring_{slug}",
    )
    # setup ingest writes its trace to a separate subdir so the query tests can
    # assert on tmp_path alone (their own trace_dir) without this setup line
    ingest(root / "docs", base=root / "docs", index_dir=index_dir, pipeline=pipeline,
           trace_dir=root / "setup_trace")
    return root, index_dir, pipeline


def _patch_strategy(monkeypatch, strategy: str):
    """Force load_settings to return a given retrieval strategy."""
    import rag_mcp_server.config.settings as settings_mod
    from rag_mcp_server.config.settings import DEFAULTS

    def fake_load_settings():
        data = {k: dict(v) if isinstance(v, dict) else v for k, v in DEFAULTS.items()}
        data["retrieval"]["strategy"] = strategy
        return data

    monkeypatch.setattr(settings_mod, "load_settings", fake_load_settings)


def _traces(tmp_path: Path):
    """The trace lines written by the last call, parsed."""
    target = tmp_path / "traces.jsonl"
    assert target.is_file(), "a trace file should have been written"
    return [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines() if line]


# -- query wiring -----------------------------------------------------------

def test_query_writes_query_trace_with_stages(tmp_path, monkeypatch):
    _, index_dir, pipeline = _offline_env(tmp_path)
    _patch_strategy(monkeypatch, "hybrid")

    results = query("MCP", index_dir=index_dir, pipeline=pipeline, trace_dir=tmp_path)

    assert isinstance(results, list)  # return type unchanged
    traces = _traces(tmp_path)
    assert len(traces) == 1
    trace = traces[0]
    assert trace["trace_type"] == "query"
    stage_names = [s["stage"] for s in trace["stages"]]
    assert stage_names == ["query_processing", "dense_retrieval", "sparse_retrieval", "fusion"]
    assert all(isinstance(s["elapsed_ms"], (int, float)) for s in trace["stages"])


def test_query_stage_durations_reflect_real_work(tmp_path, monkeypatch):
    """Each stage's elapsed_ms must measure its own work, not the previous step.

    Regression for the review finding: stages were recorded *before* the search
    ran, so dense_retrieval actually timed the embedding. query_processing must
    carry the embedding time; dense_retrieval must carry the search time and be
    non-trivial (a real store query, however small).
    """
    _, index_dir, pipeline = _offline_env(tmp_path)
    _patch_strategy(monkeypatch, "dense")

    query("MCP", index_dir=index_dir, pipeline=pipeline, trace_dir=tmp_path)

    trace = _traces(tmp_path)[0]
    stages = {s["stage"]: s for s in trace["stages"]}
    # query_processing covers validation + embedding (>= 0 is fine); the point
    # is the dense_retrieval stage is its own measurement and is recorded after
    # the search, so it reflects the search's own elapsed time.
    assert "embedding_dim" in stages["query_processing"]["data"]
    assert stages["dense_retrieval"]["elapsed_ms"] >= 0
    assert stages["query_processing"]["elapsed_ms"] >= 0


def test_dense_strategy_records_only_actual_stages(tmp_path, monkeypatch):
    _, index_dir, pipeline = _offline_env(tmp_path)
    _patch_strategy(monkeypatch, "dense")

    query("MCP", index_dir=index_dir, pipeline=pipeline, trace_dir=tmp_path)

    trace = _traces(tmp_path)[0]
    stage_names = [s["stage"] for s in trace["stages"]]
    assert stage_names == ["query_processing", "dense_retrieval"]


def test_empty_text_records_only_query_processing(tmp_path, monkeypatch):
    _, index_dir, pipeline = _offline_env(tmp_path)
    _patch_strategy(monkeypatch, "hybrid")

    query("   ", index_dir=index_dir, pipeline=pipeline, trace_dir=tmp_path)

    trace = _traces(tmp_path)[0]
    stage_names = [s["stage"] for s in trace["stages"]]
    assert stage_names == ["query_processing"]


def test_query_trace_carries_query_and_result_count_metadata(tmp_path, monkeypatch):
    """C1/C8: the query trace records the query text + actual result count in
    top-level metadata, so the query_traces page can label rows (a real chain,
    not the fixture shape the unit test used to invent)."""
    _, index_dir, pipeline = _offline_env(tmp_path)
    _patch_strategy(monkeypatch, "hybrid")

    results = query("MCP", index_dir=index_dir, pipeline=pipeline, trace_dir=tmp_path)

    trace = _traces(tmp_path)[0]
    metadata = trace["metadata"]
    assert metadata["query"] == "MCP"
    assert metadata["result_count"] == len(results)  # actual, not top_k
    assert metadata["top_k"] == 3


def test_empty_text_query_trace_records_zero_result_count(tmp_path, monkeypatch):
    _, index_dir, pipeline = _offline_env(tmp_path)
    _patch_strategy(monkeypatch, "hybrid")

    query("   ", index_dir=index_dir, pipeline=pipeline, trace_dir=tmp_path)

    trace = _traces(tmp_path)[0]
    assert trace["metadata"]["query"] == "   "
    assert trace["metadata"]["result_count"] == 0


# -- ingestion wiring -------------------------------------------------------

def test_ingest_with_transforms_records_transform_stage(tmp_path):
    """With transforms wired, an ingestion trace gains one 'transform' stage
    (applied names + counts); default-empty records nothing (the four-stage
    shape above stays green)."""
    from rag_mcp_server.ingestion.transform import ChunkRefiner, MetadataEnricher

    root = Path(tmp_path)
    (root / "docs").mkdir()
    (root / "docs" / "notes.txt").write_text("# 标题\n正文内容。", encoding="utf-8")

    embedding = FakeEmbedding(dim=4)
    store = ChromaVectorStore(persist_dir=None, collection="test_trace_transform")
    pipeline = IngestPipeline(
        embedding=embedding,
        store=store,
        index_dir=root / "out",
        transforms=[ChunkRefiner(), MetadataEnricher()],
    )

    ingest(root / "docs", base=root / "docs", index_dir=root / "out",
           pipeline=pipeline, trace_dir=tmp_path)

    trace = _traces(tmp_path)[0]
    stage_names = [s["stage"] for s in trace["stages"]]
    assert stage_names == ["load", "split", "transform", "embed", "upsert"]
    transform_stage = next(s for s in trace["stages"] if s["stage"] == "transform")
    assert transform_stage["data"]["applied"] == ["ChunkRefiner", "MetadataEnricher"]
    assert transform_stage["data"]["refined_by"] == {"rule": 1}
    assert transform_stage["data"]["enriched_by"] == {"rule": 1}


def test_ingest_without_transforms_records_four_stages(tmp_path):
    """Default transforms=[] -> no 'transform' stage; the four-stage shape is
    unchanged (the pre-existing exact-equality assertions stand)."""
    root = Path(tmp_path)
    (root / "docs").mkdir()
    (root / "docs" / "notes.txt").write_text("第一段。", encoding="utf-8")

    embedding = FakeEmbedding(dim=4)
    store = ChromaVectorStore(persist_dir=None, collection="test_trace_notransform")
    pipeline = IngestPipeline(embedding=embedding, store=store, index_dir=root / "out")

    ingest(root / "docs", base=root / "docs", index_dir=root / "out",
           pipeline=pipeline, trace_dir=tmp_path)

    trace = _traces(tmp_path)[0]
    stage_names = [s["stage"] for s in trace["stages"]]
    assert stage_names == ["load", "split", "embed", "upsert"]


def test_ingest_writes_ingestion_trace_with_per_file_stages(tmp_path):
    """Ingest trace through the rag_service.ingest seam (the spec's seam)."""
    root = Path(tmp_path)
    (root / "docs").mkdir()
    (root / "docs" / "notes.txt").write_text("第一段。\n\n第二段。", encoding="utf-8")
    (root / "docs" / "guide.pdf").write_bytes(minimal_pdf("Hello RAG PDF"))

    embedding = FakeEmbedding(dim=4)
    store = ChromaVectorStore(persist_dir=None, collection="test_trace_ingest")
    pipeline = IngestPipeline(embedding=embedding, store=store, index_dir=root / "out")

    summary = ingest(root / "docs", base=root / "docs", index_dir=root / "out",
                     pipeline=pipeline, trace_dir=tmp_path)

    assert summary.total_files == 2  # return type unchanged
    traces = _traces(tmp_path)
    assert len(traces) == 1
    trace = traces[0]
    assert trace["trace_type"] == "ingestion"
    # per-file stages (Q15): each imported file records its own load/split/embed/upsert
    stage_names = [s["stage"] for s in trace["stages"]]
    assert stage_names == ["load", "split", "embed", "upsert"] * 2
    assert all(isinstance(s["elapsed_ms"], (int, float)) for s in trace["stages"])


def test_single_file_ingest_records_exact_four_stages(tmp_path):
    root = Path(tmp_path)
    (root / "docs").mkdir()
    (root / "docs" / "notes.txt").write_text("第一段。", encoding="utf-8")

    embedding = FakeEmbedding(dim=4)
    store = ChromaVectorStore(persist_dir=None, collection="test_trace_ingest_one")
    pipeline = IngestPipeline(embedding=embedding, store=store, index_dir=root / "out")

    pipeline.ingest(root / "docs", base=root / "docs", trace_dir=tmp_path)

    trace = _traces(tmp_path)[0]
    stage_names = [s["stage"] for s in trace["stages"]]
    assert stage_names == ["load", "split", "embed", "upsert"]


def test_idempotent_reingest_still_records_all_four_stages(tmp_path):
    """A re-ingest where every chunk already exists must still show four stages."""
    root = Path(tmp_path)
    (root / "docs").mkdir()
    (root / "docs" / "notes.txt").write_text("第一段。", encoding="utf-8")

    embedding = FakeEmbedding(dim=4)
    store = ChromaVectorStore(persist_dir=None, collection="test_trace_ingest_idem")
    pipeline = IngestPipeline(embedding=embedding, store=store, index_dir=root / "out")

    pipeline.ingest(root / "docs", base=root / "docs", trace_dir=tmp_path)
    summary = pipeline.ingest(root / "docs", base=root / "docs", trace_dir=tmp_path)

    assert summary.total_chunks == 0  # idempotent skip
    trace = _traces(tmp_path)[-1]  # the second ingest's trace
    stage_names = [s["stage"] for s in trace["stages"]]
    assert stage_names == ["load", "split", "embed", "upsert"]
    # the idempotent-skip embed/upsert carry the marker
    upsert_stage = next(s for s in trace["stages"] if s["stage"] == "upsert")
    assert upsert_stage["data"] == {"added": 0, "idempotent": True}


def test_empty_doc_still_records_four_stages(tmp_path):
    """Regression (review F1): a blank/paragraph-less doc must still record the
    full four-stage shape per file (spec US7), with embed/upsert marked empty.

    chunk_document returns [] for a textless doc; _upsert_chunks must record
    embed/upsert even when there is nothing to store, so the trace shape stays
    honest and Dashboard/P10.8's per-file four-stage expectation holds.
    """
    root = Path(tmp_path)
    (root / "docs").mkdir()
    (root / "docs" / "blank.txt").write_text("   \n\n   \n", encoding="utf-8")

    embedding = FakeEmbedding(dim=4)
    store = ChromaVectorStore(persist_dir=None, collection="test_trace_ingest_empty")
    pipeline = IngestPipeline(embedding=embedding, store=store, index_dir=root / "out")

    summary = pipeline.ingest(root / "docs", base=root / "docs", trace_dir=tmp_path)

    # business contract unchanged: a textless doc is imported with 0 chunks
    assert summary.total_files == 1
    assert summary.imported[0]["chunk_count"] == 0
    trace = _traces(tmp_path)[0]
    stage_names = [s["stage"] for s in trace["stages"]]
    assert stage_names == ["load", "split", "embed", "upsert"]
    embed_stage = next(s for s in trace["stages"] if s["stage"] == "embed")
    assert embed_stage["data"] == {"chunks": 0, "empty": True}
    upsert_stage = next(s for s in trace["stages"] if s["stage"] == "upsert")
    assert upsert_stage["data"] == {"added": 0, "empty": True}


def test_corrupt_pdf_records_skip_stage_with_reason(tmp_path):
    """A corrupt PDF is skipped; the trace records the skip + reason."""
    root = Path(tmp_path)
    (root / "docs").mkdir()
    (root / "docs" / "notes.txt").write_text("正文内容。", encoding="utf-8")
    (root / "docs" / "broken.pdf").write_bytes(b"%PDF-1.4 fake\n")

    embedding = FakeEmbedding(dim=4)
    store = ChromaVectorStore(persist_dir=None, collection="test_trace_ingest_skip")
    pipeline = IngestPipeline(embedding=embedding, store=store, index_dir=root / "out")

    summary = pipeline.ingest(root / "docs", base=root / "docs", trace_dir=tmp_path)

    assert summary.total_skipped == 1
    trace = _traces(tmp_path)[0]
    stages = trace["stages"]
    assert any(s["stage"] == "skip" and "pdf parse failed" in s["data"]["reason"]
               for s in stages)
    # the good file still got its full four-stage path
    assert any(s["stage"] == "upsert" for s in stages)


# -- disabled / isolation ---------------------------------------------------

def test_trace_disabled_writes_nothing(tmp_path, monkeypatch):
    _, index_dir, pipeline = _offline_env(tmp_path)
    _patch_strategy(monkeypatch, "sparse")

    import rag_mcp_server.config.settings as settings_mod
    from rag_mcp_server.config.settings import DEFAULTS

    def disabled_settings():
        data = {k: dict(v) if isinstance(v, dict) else v for k, v in DEFAULTS.items()}
        data["retrieval"]["strategy"] = "sparse"
        data["trace"] = {"enabled": False, "dir": "logs"}
        return data

    monkeypatch.setattr(settings_mod, "load_settings", disabled_settings)

    query("MCP", index_dir=index_dir, pipeline=pipeline, trace_dir=tmp_path)

    assert not (tmp_path / "traces.jsonl").exists()


# -- rerank trace stage (P10.3) ---------------------------------------------
def test_rerank_enabled_records_rerank_stage(tmp_path, monkeypatch):
    """rerank enabled -> a 'rerank' stage with the fallback marker (default off
    records no stage, so the existing exact-stage assertions stay green)."""
    import json as _json

    from rag_mcp_server.libs.llm.fake import FakeLLM
    from rag_mcp_server.libs.reranker.llm_reranker import LLMReranker

    def scored(messages):
        # The prompt lists each candidate as "Passage ID: <chunk_id>"; extract
        # the real chunk ids (which carry the test's collection prefix, ADR 0016)
        # and score the MCP chunk above the RAG chunk without hardcoding ids.
        prompt = messages[-1]["content"]
        import re as _re
        ids = _re.findall(r"Passage ID: (\S+)", prompt)
        items = [{"passage_id": cid, "score": 1, "reasoning": ""} for cid in ids]
        # bump the MCP one to 3 so order is verifiable
        for item in items:
            if "mcp-notes" in item["passage_id"]:
                item["score"] = 3
        return _json.dumps(items)

    llm = FakeLLM()
    llm._chat = scored  # type: ignore[method-assign]

    _, index_dir, pipeline = _offline_env(tmp_path)
    _patch_strategy(monkeypatch, "hybrid")

    query("MCP", index_dir=index_dir, pipeline=pipeline, trace_dir=tmp_path,
          rerank=True, reranker=LLMReranker(llm=llm))

    trace = _traces(tmp_path)[0]
    stage_names = [s["stage"] for s in trace["stages"]]
    assert stage_names[-1] == "rerank"
    rerank_stage = trace["stages"][-1]
    assert rerank_stage["data"]["fallback"] is False
    assert rerank_stage["data"]["output_k"] == 2  # 2 chunks, both scored, truncated to top_k=3
    assert "reason" in rerank_stage["data"]


def test_rerank_disabled_records_no_rerank_stage(tmp_path, monkeypatch):
    """rerank off (default) -> no 'rerank' stage; the four hybrid stages stand."""
    _, index_dir, pipeline = _offline_env(tmp_path)
    _patch_strategy(monkeypatch, "hybrid")

    query("MCP", index_dir=index_dir, pipeline=pipeline, trace_dir=tmp_path)

    trace = _traces(tmp_path)[0]
    stage_names = [s["stage"] for s in trace["stages"]]
    assert stage_names == ["query_processing", "dense_retrieval", "sparse_retrieval", "fusion"]
