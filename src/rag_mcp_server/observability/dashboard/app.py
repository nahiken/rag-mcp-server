"""Dashboard entry point — Streamlit app (the seam to the runtime).

ADR 0014: ``pages`` and ``services`` never import streamlit at module top
level (render functions import it lazily in their bodies); this module is the
single seam that boots the runtime. streamlit is an optional/dev dependency
(P10.8): when it is missing we print a friendly hint to stderr and exit cleanly
instead of raising an ImportError traceback (US 9).

Start with either:
    streamlit run src/rag_mcp_server/observability/dashboard/app.py
    python scripts/dashboard.py
"""

from __future__ import annotations


def _run() -> None:
    try:
        import streamlit as st
    except ModuleNotFoundError:
        import sys

        sys.stderr.write(
            "Dashboard 需要 streamlit（dev 依赖）。请安装：\n"
            "    pip install \"streamlit>=1.36\"\n"
            "或：pip install -e \".[dev]\"\n"
        )
        raise SystemExit(1)

    from rag_mcp_server.observability.dashboard.pages.data_browser import render_data_browser
    from rag_mcp_server.observability.dashboard.pages.evaluation_panel import render_evaluation_panel
    from rag_mcp_server.observability.dashboard.pages.ingestion_manager import render_ingestion_manager
    from rag_mcp_server.observability.dashboard.pages.ingestion_traces import render_ingestion_traces
    from rag_mcp_server.observability.dashboard.pages.overview import render_overview
    from rag_mcp_server.observability.dashboard.pages.query_traces import render_query_traces

    # st.navigation + st.Page(fn=...) route to the render functions (streamlit
    # >= 1.36). The page module is imported lazily on selection.
    pages = [
        st.Page(render_overview, title="总览", url_path="overview", default=True),
        st.Page(render_ingestion_traces, title="摄取追踪", url_path="ingestion_traces"),
        st.Page(render_query_traces, title="查询追踪", url_path="query_traces"),
        st.Page(render_data_browser, title="数据浏览", url_path="data_browser"),
        st.Page(render_ingestion_manager, title="摄取管理", url_path="ingestion_manager"),
        st.Page(render_evaluation_panel, title="评估面板", url_path="evaluation_panel"),
    ]
    page = st.navigation(pages)
    page.run()

_run()
