"""Query traces page — per-query stage timings from traces.jsonl (ADR 0014 ext).

Pure data via ``trace_service.aggregate_queries`` (streamlit-free), rendered
with streamlit's bundled Altair. Empty/missing traces shows a friendly state.
Query traces record stages query_processing / dense_retrieval / sparse_retrieval
/ fusion / rerank (rag_service.query).
"""

from __future__ import annotations

from pathlib import Path

from rag_mcp_server.config.settings import load_settings
from rag_mcp_server.observability.dashboard.services.trace_service import (
    QUERY_STAGE_MS_KEYS,
    aggregate_queries,
    load_traces,
)
from rag_mcp_server.observability.logger import TRACES_FILENAME

_CHART_TOP_N = 20


def _traces_path(trace_dir: Path | None) -> Path:
    if trace_dir is not None:
        return Path(trace_dir) / TRACES_FILENAME
    settings = load_settings()
    return Path(settings.get("trace", {}).get("dir", "logs")) / TRACES_FILENAME


def render_query_traces(trace_dir: Path | str | None = None) -> None:
    """Render the query traces page (stages + waterfall per query)."""
    import streamlit as st

    st.title("RAG Knowledge Hub — 查询追踪")

    path = _traces_path(Path(trace_dir) if trace_dir is not None else None)
    traces = load_traces(path)
    rows = aggregate_queries(traces)

    if not rows:
        st.markdown(
            "暂无 query traces。请先运行一次查询：\n\n"
            "```bash\nPYTHONPATH=src python -m rag_mcp_server query 检索\n"
            "```\n\n"
            "查询完成后刷新本页即可看到每阶段的耗时。"
        )
        return

    import pandas as pd

    chart_rows = rows[:_CHART_TOP_N]
    chart_df = pd.DataFrame([r.to_dict() for r in chart_rows])
    if not chart_df.empty:
        melted = chart_df.melt(
            id_vars=["query", "total_ms"],
            value_vars=[c for c in QUERY_STAGE_MS_KEYS if c in chart_df.columns],
            var_name="stage_ms",
            value_name="elapsed_ms",
        )
        melted["stage"] = melted["stage_ms"].str.removesuffix("_ms")
        melted = melted[melted["elapsed_ms"] > 0]

        if not melted.empty:
            import altair as alt

            chart = (
                alt.Chart(melted)
                .mark_bar()
                .encode(
                    y=alt.Y("query:N", sort="-x", title="查询"),
                    x=alt.X("elapsed_ms:Q", title="耗时 (ms)"),
                    color=alt.Color("stage:N", title="阶段"),
                    tooltip=["query:N", "stage:N", "elapsed_ms:Q"],
                )
                .properties(title="每查询阶段耗时（top N）", height=min(len(chart_df) * 24 + 60, 600))
            )
            st.altair_chart(chart, width="stretch")
        else:
            st.markdown("已执行的查询无有效阶段耗时。")

    st.subheader("明细")
    st.dataframe([r.to_dict() for r in rows])
