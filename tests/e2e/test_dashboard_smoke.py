"""End-to-end smoke tests for the Streamlit Dashboard.

P10.8 acceptance (docs/spec-p10-dashboard.md): each page renders without
exception under AppTest, and the empty-traces path renders a friendly empty
state instead of raising.

AppTest runs a self-contained script (from_string) in a bare-mode script runner
and collects the elements emitted. Each script imports the page module (which is
streamlit-free at top level, ADR 0014) and calls the render function with
explicit hermetic args. The pure logic (trace parsing / stats) is covered in
tests/unit/test_dashboard_services.py; these tests only prove the render path is
wired and does not throw.

Hermeticity: the pages read the repo settings/logs when pointed at the repo, so
every test passes explicit settings / a temp trace dir (no repo logs/ writes, no
real vector store). This mirrors the repo's isolation discipline
(spec-p10-trace US 14: tests never pollute repo logs/).

Prior art: tests/e2e/test_phase9.py (offline full-loop); tests/unit/test_trace_wiring.py
(settings isolation for the trace dir). AppTest.from_string is the first
streamlit.testing use in this repo.
"""

from __future__ import annotations

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

from rag_mcp_server.config.settings import DEFAULTS


def _settings(trace_dir, trace_enabled: bool = True) -> dict:
    """Repo settings with trace dir AND vector store persist dir pointed at tmp.

    No repo logs/ writes and no real ``data/chroma`` ever opened by tests
    (S1): the overview's knowledge_stats builds the store from these settings,
    so leaving persist_dir at the default would touch the repo's real Chroma.
    The store is built over an empty tmp dir, so stats read as zero cards —
    exactly the degraded-card path, hermetically.
    """
    data = {k: dict(v) if isinstance(v, dict) else v for k, v in DEFAULTS.items()}
    data["trace"] = {"enabled": trace_enabled, "dir": str(trace_dir)}
    data["vector_store"] = {
        "provider": "chroma",
        "persist_dir": str(trace_dir / "chroma"),
        "collection": "rag_chunks",
    }
    return data


def _app_test(script: str) -> AppTest:
    return AppTest.from_string(script, default_timeout=10).run()


def _overview(tmp_path, trace_enabled: bool = True) -> AppTest:
    settings = _settings(tmp_path, trace_enabled)
    script = (
        "from rag_mcp_server.observability.dashboard.pages.overview import render_overview\n"
        f"render_overview(settings={settings!r})\n"
    )
    return _app_test(script)


def _traces(tmp_path) -> AppTest:
    # trace_dir = the directory that contains traces.jsonl (write_trace semantics)
    script = (
        "from rag_mcp_server.observability.dashboard.pages.ingestion_traces import render_ingestion_traces\n"
        f"render_ingestion_traces(trace_dir={json.dumps(str(tmp_path))})\n"
    )
    return _app_test(script)


def _write_trace(tmp_path, stages: list[dict]) -> None:
    (tmp_path / "traces.jsonl").write_text(
        json.dumps(
            {
                "trace_id": "t",
                "trace_type": "ingestion",
                "started_at": "2026-08-25T00:00:00Z",
                "finished_at": "2026-08-25T00:00:01Z",
                "total_elapsed_ms": 1.0,
                "stages": stages,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_overview_page_renders_config_and_stats(tmp_path):
    settings = _settings(tmp_path)
    at = _overview(tmp_path)
    assert not at.exception, f"overview raised: {at.exception}"
    assert at.title and "总览" in at.title[0].value
    # config cards carry the configured llm model
    assert any(str(settings["llm"]["model"]) in el.value for el in at.markdown)
    # knowledge base stats -> number cards. The store is built over an empty tmp
    # dir (S1 hermeticity), so stats read as zero documents/chunks — degraded
    # cards, never the repo's real data/chroma.
    labels = [m.label for m in at.metric]
    assert {"documents", "chunks"} <= set(labels)


def test_overview_warns_when_trace_disabled(tmp_path):
    at = _overview(tmp_path, trace_enabled=False)
    assert not at.exception
    assert any("追踪已关闭" in el.value for el in at.warning)


def test_traces_page_empty_state_when_no_trace_file(tmp_path):
    at = _traces(tmp_path)
    assert not at.exception, f"traces raised: {at.exception}"
    assert any("暂无 ingestion traces" in el.value for el in at.markdown)


def test_traces_page_renders_rows_with_trace_file(tmp_path):
    _write_trace(
        tmp_path,
        [
            {"stage": "load", "data": {"file": "a.md"}, "elapsed_ms": 10.0},
            {"stage": "split", "data": {"chunks": 1}, "elapsed_ms": 10.0},
            {"stage": "embed", "data": {"chunks": 1}, "elapsed_ms": 10.0},
            {"stage": "upsert", "data": {"added": 1}, "elapsed_ms": 10.0},
        ],
    )
    at = _traces(tmp_path)
    assert not at.exception, f"traces raised: {at.exception}"
    assert at.dataframe, "expected a dataframe row for the ingested file"


def test_traces_page_renders_skip_rows(tmp_path):
    _write_trace(
        tmp_path,
        [
            {"stage": "load", "data": {"file": "a.md"}, "elapsed_ms": 10.0},
            {"stage": "split", "data": {"chunks": 1}, "elapsed_ms": 10.0},
            {"stage": "embed", "data": {"chunks": 1}, "elapsed_ms": 10.0},
            {"stage": "upsert", "data": {"added": 1}, "elapsed_ms": 10.0},
            {"stage": "skip", "data": {"file": "bad.pdf", "reason": "DocumentLoadError"}, "elapsed_ms": 0.0},
        ],
    )
    at = _traces(tmp_path)
    assert not at.exception
    assert any("bad.pdf" in el.value and "跳过" in el.value for el in at.markdown)


def test_traces_page_all_files_skipped_renders(tmp_path):
    """All files skipped: no chart bars, but the page still renders the skipped
    block + detail rows instead of the "no timings" placeholder or an error."""
    _write_trace(
        tmp_path,
        [
            {"stage": "skip", "data": {"file": "bad1.pdf", "reason": "DocumentLoadError"}, "elapsed_ms": 0.0},
            {"stage": "skip", "data": {"file": "bad2.pdf", "reason": "DocumentLoadError"}, "elapsed_ms": 0.0},
        ],
    )
    at = _traces(tmp_path)
    assert not at.exception, f"traces raised: {at.exception}"
    assert any("bad1.pdf" in el.value and "跳过" in el.value for el in at.markdown)
    assert any("bad2.pdf" in el.value and "跳过" in el.value for el in at.markdown)
    assert at.dataframe, "expected detail rows for the skipped files"


def test_traces_page_skip_survives_chart_top_n(tmp_path):
    """T2: files beyond the chart top-N still appear — skipped files never
    vanish from the detail/skip views even when 20+ files precede them."""
    stages: list[dict] = []
    for i in range(25):
        stages.append({"stage": "load", "data": {"file": f"f{i:02d}.md"}, "elapsed_ms": 10.0})
        stages.append({"stage": "split", "data": {"chunks": 1}, "elapsed_ms": 10.0})
        stages.append({"stage": "upsert", "data": {"added": 1}, "elapsed_ms": 10.0})
    stages.append({"stage": "skip", "data": {"file": "bad.pdf", "reason": "DocumentLoadError"}, "elapsed_ms": 0.0})
    _write_trace(tmp_path, stages)

    at = _traces(tmp_path)
    assert not at.exception, f"traces raised: {at.exception}"
    # the skip file is past the top 20 chart bars but must stay visible
    assert any("bad.pdf" in el.value and "跳过" in el.value for el in at.markdown)
    # and the full detail dataframe keeps every file (26 rows, incl. the skip)
    assert at.dataframe, "expected a detail dataframe"
    assert "bad.pdf" in set(at.dataframe[0].value["file"])


def test_launcher_friendly_hint_when_streamlit_missing():
    """T1 (US 9): scripts/dashboard.py prints the friendly hint and exits 1 when
    streamlit is missing, instead of a raw ImportError traceback.

    The import of streamlit is forced to fail via a meta-path blocker so the
    test is deterministic whether or not streamlit is installed (dev extras).
    """
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    launcher = repo_root / "scripts" / "dashboard.py"

    blocker = (
        "import importlib.abc, runpy, sys\n"
        "class _Blocker(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname == 'streamlit' or fullname.startswith('streamlit.'):\n"
        "            raise ModuleNotFoundError(f\"No module named '{fullname}'\")\n"
        "sys.meta_path.insert(0, _Blocker())\n"
        f"runpy.run_path({str(launcher)!r}, run_name='__main__')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", blocker],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1, f"returncode={result.returncode}, stderr={result.stderr}"
    assert "需要 streamlit" in result.stderr, f"stderr: {result.stderr}"
    assert "Traceback" not in result.stderr, result.stderr
