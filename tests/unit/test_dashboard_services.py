"""Tests for the Dashboard services layer — pure data logic, no streamlit.

Red/green contract (docs/spec-p10-dashboard.md):
- services/trace_service.load_traces reads logs/traces.jsonl -> list[dict], tolerating
  missing files and non-JSON lines (skip, never raise)
- services/trace_service.aggregate_by_file groups ingestion stages per file into
  StageTiming rows (load/split/transform/embed/upsert elapsed_ms + total), all
  files returned (top-N charting is the page's job)
- skip / integrity_skip files are marked as skipped with zeroed timings — no
  fabricated stage durations, never a fake "fast file" (T4)
- services/data_service.summary maps DocumentManager.stats() {documents, chunks}
  into number-card structure
- pages/ and services/ top-level never import streamlit (import contract, so pytest
  can hit the logic directly and the six-page extension point stays streamlit-free)

Prior art: tests/unit/test_trace.py (trace JSONL parsing), tests/unit/test_settings.py
(defaults fallback), tests/unit/test_document_manager.py (stats over a temp store).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_mcp_server.observability.dashboard.services import data_service, trace_service


def _trace(*stages: tuple[str, dict]) -> dict:
    return {
        "trace_id": "t",
        "trace_type": "ingestion",
        "started_at": "2026-08-25T00:00:00Z",
        "finished_at": "2026-08-25T00:00:01Z",
        "total_elapsed_ms": 1.0,
        "stages": [{"stage": s, "data": d, "elapsed_ms": 10.0} for s, d in stages],
    }


def _write_jsonl(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# -- load_traces ------------------------------------------------------------
def test_load_traces_empty_file_returns_empty_list(tmp_path):
    p = _write_jsonl(tmp_path / "traces.jsonl", [])
    assert trace_service.load_traces(p) == []


def test_load_traces_missing_file_returns_empty_list(tmp_path):
    assert trace_service.load_traces(tmp_path / "nope.jsonl") == []


def test_load_traces_parses_each_line_as_dict(tmp_path):
    p = _write_jsonl(
        tmp_path / "traces.jsonl",
        [
            '{"trace_id": "a", "trace_type": "ingestion", "stages": []}',
            '{"trace_id": "b", "trace_type": "query", "stages": []}',
        ],
    )
    traces = trace_service.load_traces(p)
    assert [t["trace_id"] for t in traces] == ["a", "b"]


def test_load_traces_skips_corrupt_json_lines(tmp_path):
    p = _write_jsonl(
        tmp_path / "traces.jsonl",
        [
            '{"trace_id": "ok", "trace_type": "ingestion", "stages": []}',
            "this is not json {{{",
            "",
            '{"trace_id": "ok2", "trace_type": "ingestion", "stages": []}',
        ],
    )
    traces = trace_service.load_traces(p)
    assert [t["trace_id"] for t in traces] == ["ok", "ok2"]


# -- aggregate_by_file ------------------------------------------------------
def test_aggregate_by_file_empty_returns_empty(tmp_path):
    p = _write_jsonl(tmp_path / "traces.jsonl", [])
    assert trace_service.aggregate_by_file(trace_service.load_traces(p)) == []


def test_aggregate_by_file_groups_per_file_and_sums_stages(tmp_path):
    p = _write_jsonl(
        tmp_path / "traces.jsonl",
        [
            # two files inside one ingestion trace; each load..upsert is its own file
            json.dumps(
                _trace(
                    ("load", {"file": "a.md"}),
                    ("split", {"chunks": 2}),
                    ("embed", {"chunks": 2}),
                    ("upsert", {"added": 2}),
                    ("load", {"file": "b.pdf"}),
                    ("split", {"chunks": 1}),
                    ("embed", {"chunks": 1}),
                    ("upsert", {"added": 1}),
                ),
                ensure_ascii=False,
            )
        ],
    )
    rows = trace_service.aggregate_by_file(trace_service.load_traces(p))
    assert len(rows) == 2
    by_file = {r.file: r for r in rows}
    assert set(by_file) == {"a.md", "b.pdf"}
    assert by_file["a.md"].timings["load_ms"] == 10.0 and by_file["a.md"].timings["upsert_ms"] == 10.0
    assert by_file["a.md"].total_ms == 40.0
    assert by_file["b.pdf"].timings["split_ms"] == 10.0 and by_file["b.pdf"].total_ms == 40.0


def test_aggregate_by_file_marks_skip_files(tmp_path):
    p = _write_jsonl(
        tmp_path / "traces.jsonl",
        [
            json.dumps(
                _trace(
                    ("load", {"file": "good.md"}),
                    ("split", {"chunks": 1}),
                    ("embed", {"chunks": 1}),
                    ("upsert", {"added": 1}),
                    ("load", {"file": "bad.pdf"}),
                    ("skip", {"file": "bad.pdf", "reason": "DocumentLoadError"}),
                ),
                ensure_ascii=False,
            )
        ],
    )
    rows = trace_service.aggregate_by_file(trace_service.load_traces(p))
    by_file = {r.file: r for r in rows}
    assert by_file["good.md"].skipped is False
    assert by_file["bad.pdf"].skipped is True
    # T4: a skip is honest — its (real, non-zero) load timing is zeroed so the
    # row reads as "skipped, untimed" and never ranks as a fast file.
    assert by_file["bad.pdf"].timings == {k: 0.0 for k in trace_service.STAGE_MS_KEYS}
    assert by_file["bad.pdf"].total_ms == 0.0


def test_aggregate_by_file_marks_integrity_skip_files(tmp_path):
    p = _write_jsonl(
        tmp_path / "traces.jsonl",
        [json.dumps(_trace(("integrity_skip", {"file": "unchanged.md", "reason": "unchanged"})))],
    )
    rows = trace_service.aggregate_by_file(trace_service.load_traces(p))
    assert len(rows) == 1
    assert rows[0].file == "unchanged.md" and rows[0].skipped is True
    assert rows[0].total_ms == 0.0


def test_aggregate_by_file_sorts_by_total_desc(tmp_path):
    p = _write_jsonl(
        tmp_path / "traces.jsonl",
        [
            json.dumps(
                _trace(
                    ("load", {"file": "slow.md"}), ("split", {"chunks": 1}),
                    ("embed", {"chunks": 1}), ("upsert", {"added": 1}),
                    ("load", {"file": "fast.md"}), ("split", {"chunks": 1}),
                    ("embed", {"chunks": 1}), ("upsert", {"added": 1}),
                )
            )
        ],
    )
    rows = trace_service.aggregate_by_file(trace_service.load_traces(p))
    assert [r.file for r in rows] == ["slow.md", "fast.md"]


def test_aggregate_by_file_returns_all_files_no_top_n(tmp_path):
    """T2: aggregate returns the FULL set — top-N charting is the page's job, so
    no file (incl. skipped ones) can be silently dropped from the detail/skip
    views."""
    p = _write_jsonl(
        tmp_path / "traces.jsonl",
        [
            json.dumps(
                _trace(
                    *[s for f in ("a.md", "b.md", "c.md") for s in
                        (("load", {"file": f}), ("split", {"chunks": 1}),
                         ("embed", {"chunks": 1}), ("upsert", {"added": 1}))]
                )
            )
        ],
    )
    rows = trace_service.aggregate_by_file(trace_service.load_traces(p))
    assert [r.file for r in rows] == ["a.md", "b.md", "c.md"]  # all 3, no truncation


def test_stage_ms_keys_are_single_source_of_stage_list():
    """S3: the stage list is encoded once — StageTiming.timings keys and the
    public STAGE_MS_KEYS derive from _STAGE_KEYS; a new pipeline stage adds one
    place."""
    assert trace_service.STAGE_MS_KEYS == (
        "load_ms", "split_ms", "transform_ms", "embed_ms", "upsert_ms",
    )
    row = trace_service.StageTiming(
        file="a.md",
        timings={k: 1.0 for k in trace_service.STAGE_MS_KEYS},
        total_ms=5.0,
        skipped=False,
    )
    d = row.to_dict()
    for key in trace_service.STAGE_MS_KEYS:
        assert key in d


def test_skipped_files_sort_last_even_with_load_timing(tmp_path):
    """T4 regression: a skip file whose load genuinely took time must not rank
    as a fast file — zeroed timings put it last, so it never crowds a real
    file out of the chart's top N."""
    p = _write_jsonl(
        tmp_path / "traces.jsonl",
        [
            json.dumps(
                _trace(
                    ("load", {"file": "slow.md"}), ("split", {"chunks": 1}),
                    ("embed", {"chunks": 1}), ("upsert", {"added": 1}),
                    ("load", {"file": "bad.pdf"}),  # real load time below
                    ("skip", {"file": "bad.pdf", "reason": "DocumentLoadError"}),
                )
            )
        ],
    )
    rows = trace_service.aggregate_by_file(trace_service.load_traces(p))
    assert [r.file for r in rows] == ["slow.md", "bad.pdf"]  # skip last
    assert rows[-1].skipped is True and rows[-1].total_ms == 0.0


def test_aggregate_by_file_ignores_query_traces(tmp_path):
    p = _write_jsonl(
        tmp_path / "traces.jsonl",
        [
            '{"trace_id": "q", "trace_type": "query", "stages": [{"stage": "fusion", "data": {}, "elapsed_ms": 1.0}]}',
            json.dumps(_trace(("load", {"file": "a.md"}), ("upsert", {"added": 1}))),
        ],
    )
    rows = trace_service.aggregate_by_file(trace_service.load_traces(p))
    assert [r.file for r in rows] == ["a.md"]


# -- data_service.summary ---------------------------------------------------
def test_summary_maps_stats_to_cards():
    cards = data_service.summary({"documents": 3, "chunks": 42})
    assert cards == [
        {"label": "documents", "value": 3},
        {"label": "chunks", "value": 42},
    ]


def test_knowledge_stats_builds_store_and_reads_stats(tmp_path, monkeypatch):
    """knowledge_stats wires create_vector_store + DocumentManager.stats and returns counts."""
    from rag_mcp_server.libs.vector_store.base import VectorRecord

    import rag_mcp_server.observability.dashboard.services.data_service as ds

    def fake_create(provider, persist_dir, collection):
        store = _Store()
        store.add(
            VectorRecord(
                chunk_id="d1::0",
                text="x",
                metadata={"document_id": "d1", "source_path": "a.md"},
            )
        )
        return store

    monkeypatch.setattr(ds, "create_vector_store", fake_create)

    stats = data_service.knowledge_stats({"vector_store": {"provider": "chroma"}})
    assert stats == {"documents": 1, "chunks": 1}


def test_knowledge_stats_degrades_to_zero_on_build_failure(monkeypatch):
    """A store that cannot be built yields zero cards, never an exception (US 7)."""
    import rag_mcp_server.observability.dashboard.services.data_service as ds

    def boom(provider, persist_dir, collection):
        raise ValueError("no index")

    monkeypatch.setattr(ds, "create_vector_store", boom)
    assert data_service.knowledge_stats({}) == {"documents": 0, "chunks": 0}


class _Store:
    def __init__(self):
        self._records = []

    def add(self, record):
        self._records.append(record)

    def get_all(self):
        return list(self._records)


# -- import contract: services/pages never import streamlit -----------------
def test_services_do_not_import_streamlit():
    """Importing the services package must not pull streamlit in.

    This is the P10.8 contract that keeps the six-page extension point (ADR 0014)
    streamlit-free: pure data logic testable by pytest, renderable by any frontend.

    Run in a subprocess so prior imports in this pytest process (the e2e AppTest
    suite imports streamlit) cannot mask a leak.
    """
    import subprocess
    import sys

    code = (
        "import sys\n"
        "from rag_mcp_server.observability.dashboard.services import data_service, trace_service\n"
        "from rag_mcp_server.observability.dashboard.pages import overview, ingestion_traces\n"
        "assert 'streamlit' not in sys.modules, 'streamlit leaked into services/pages'\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=".",
        env={"PYTHONPATH": "src"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


# -- aggregate_queries (ADR 0014 extension: query_traces page) ---------------
def _query_trace(query: str, *stages: tuple[str, float], result_count: int = 3) -> dict:
    return {
        "trace_id": "q",
        "trace_type": "query",
        # C8: rag_service writes the ACTUAL result count + top_k at finish; the
        # query_traces page reads result_count (falling back to top_k for old
        # traces written before the field existed).
        "metadata": {"query": query, "result_count": result_count, "top_k": 3},
        "stages": [{"stage": s, "elapsed_ms": ms} for s, ms in stages],
    }


def test_aggregate_queries_empty_returns_empty(tmp_path):
    rows = trace_service.aggregate_queries(trace_service.load_traces(tmp_path / "traces.jsonl"))
    assert rows == []


def test_aggregate_queries_sums_stages_per_query():
    traces = [
        _query_trace("检索", ("query_processing", 5.0), ("dense_retrieval", 10.0), result_count=2),
        _query_trace("MCP", ("query_processing", 2.0), result_count=0),
    ]
    rows = trace_service.aggregate_queries(traces)
    assert len(rows) == 2
    # query "检索" -> total 15ms
    first = rows[0]
    assert first.query == "检索"  # sorted by total desc
    assert first.total_ms == 15.0
    assert first.timings["query_processing_ms"] == 5.0
    assert first.timings["dense_retrieval_ms"] == 10.0
    assert first.result_count == 2  # C8: the ACTUAL returned count, not top_k
    assert rows[1].result_count == 0  # no results -> 0


def test_aggregate_queries_falls_back_to_top_k_for_old_traces():
    """C8: traces written before result_count existed fall back to metadata.top_k."""
    trace = {
        "trace_id": "q",
        "trace_type": "query",
        "metadata": {"query": "旧查询", "top_k": 3},
        "stages": [{"stage": "query_processing", "elapsed_ms": 1.0}],
    }
    rows = trace_service.aggregate_queries([trace])
    assert rows[0].query == "旧查询"
    assert rows[0].result_count == 3


def test_aggregate_queries_ignores_non_query_traces():
    ingestion = _trace(("load", {"file": "a.md"}))
    rows = trace_service.aggregate_queries([ingestion])
    assert rows == []


def test_aggregate_queries_skips_empty_query():
    traces = [_query_trace("", ("query_processing", 1.0))]
    rows = trace_service.aggregate_queries(traces)
    assert rows == []
