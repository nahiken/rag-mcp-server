"""Smoke tests for the two new Dashboard pages (ADR 0014 extension point).

query_traces and data_browser render without exception under AppTest, and the
empty-traces path renders a friendly empty state instead of raising. Pure logic
(aggregate_queries / browse_documents) is covered in unit tests; these only
prove the render path is wired.

Same hermetic pattern as test_dashboard_smoke.py: render functions take explicit
args (trace_dir / settings), so no repo logs/ writes and no real data/chroma is
ever touched. The vector store is built over an empty tmp dir (collection
"default", ADR 0016 D22), so data_browser reads an empty state.
"""

from __future__ import annotations

import json

from streamlit.testing.v1 import AppTest

from rag_mcp_server.config.settings import DEFAULTS


def _settings(trace_dir) -> dict:
    data = {k: dict(v) if isinstance(v, dict) else v for k, v in DEFAULTS.items()}
    data["trace"] = {"enabled": True, "dir": str(trace_dir)}
    data["vector_store"] = {
        "provider": "chroma",
        "persist_dir": str(trace_dir / "chroma"),
        "collection": "default",  # ADR 0016 D22
    }
    return data


def _query_traces(tmp_path) -> AppTest:
    script = (
        "from rag_mcp_server.observability.dashboard.pages.query_traces import render_query_traces\n"
        f"render_query_traces(trace_dir={json.dumps(str(tmp_path))})\n"
    )
    return AppTest.from_string(script, default_timeout=10).run()


def _data_browser(tmp_path) -> AppTest:
    settings = _settings(tmp_path)
    script = (
        "from rag_mcp_server.observability.dashboard.pages.data_browser import render_data_browser\n"
        f"render_data_browser(settings={settings!r})\n"
    )
    return AppTest.from_string(script, default_timeout=10).run()


def test_query_traces_page_renders(tmp_path):
    """query_traces renders without exception over an empty traces dir."""
    at = _query_traces(tmp_path)
    assert not at.exception, f"query_traces raised: {at.exception}"


def test_data_browser_page_renders_empty_state(tmp_path):
    """data_browser renders the empty state over an empty store."""
    at = _data_browser(tmp_path)
    assert not at.exception, f"data_browser raised: {at.exception}"


def _ingestion_manager() -> AppTest:
    script = (
        "from rag_mcp_server.observability.dashboard.pages.ingestion_manager import render_ingestion_manager\n"
        "render_ingestion_manager()\n"
    )
    return AppTest.from_string(script, default_timeout=10).run()


def _evaluation_panel() -> AppTest:
    script = (
        "from rag_mcp_server.observability.dashboard.pages.evaluation_panel import render_evaluation_panel\n"
        "render_evaluation_panel()\n"
    )
    return AppTest.from_string(script, default_timeout=10).run()


def test_ingestion_manager_page_renders():
    """ingestion_manager renders the upload form without exception."""
    at = _ingestion_manager()
    assert not at.exception, f"ingestion_manager raised: {at.exception}"


def test_evaluation_panel_page_renders():
    """evaluation_panel renders the form without exception (no run yet)."""
    at = _evaluation_panel()
    assert not at.exception, f"evaluation_panel raised: {at.exception}"
