"""Ingestion traces page — per-file stage stacked bars from traces.jsonl.

Pure data via ``trace_service`` (load_traces + aggregate_by_file), rendered with
streamlit's bundled Altair (no extra chart dependency, ADR 0014). Empty or
missing traces file shows a friendly empty state instead of an exception
(US 7). ``skip``/``integrity_skip`` files are surfaced as a markdown row rather
than fabricated timings.

Stage columns/labels come from ``trace_service`` (STAGE_MS_KEYS) — the single
source of the pipeline stage list (S3): adding a stage touches trace_service
only, and the chart + dataframe derive from it here.
"""

from __future__ import annotations

from pathlib import Path

from rag_mcp_server.config.settings import load_settings
from rag_mcp_server.observability.dashboard.services.trace_service import (
    STAGE_MS_KEYS,
    aggregate_by_file,
    group_files_by_stage_data,
    load_traces,
)
from rag_mcp_server.observability.logger import TRACES_FILENAME

# Number of files shown as chart bars. Detail dataframe and the "skipped files"
# block always use the FULL row set, so files past top_n never disappear (T2).
_CHART_TOP_N = 20
# Altair axis label per stage column (already "<stage>_ms" from the single source).
_STAGE_LABELS = {ms_key: ms_key.removesuffix("_ms") for ms_key in STAGE_MS_KEYS}


def _traces_path(trace_dir: Path | None) -> Path:
    if trace_dir is not None:
        # The caller passes the directory that *contains* traces.jsonl — the same
        # semantics as write_trace(trace_dir). AppTest passes a temp dir directly.
        return Path(trace_dir) / TRACES_FILENAME
    settings = load_settings()
    return Path(settings.get("trace", {}).get("dir", "logs")) / TRACES_FILENAME


def render_ingestion_traces(trace_dir: Path | str | None = None) -> None:
    """Render the ingestion traces page.

    ``trace_dir`` is optional for testability (AppTest drives this function with
    an explicit dir so the render is hermetic); the app calls it with no args and
    it falls back to the repo trace configuration.
    """
    import streamlit as st

    st.title("RAG Knowledge Hub — 摄取追踪")

    path = _traces_path(Path(trace_dir) if trace_dir is not None else None)
    traces = load_traces(path)
    rows = aggregate_by_file(traces)

    if not rows:
        st.markdown(
            "暂无 ingestion traces。请先运行一次摄取：\n\n"
            "```bash\nPYTHONPATH=src python -m pytest tests/e2e/test_phase9.py -q\n"
            "# 或 CLI：\nPYTHONPATH=src python -m rag_mcp_server.cli ingest <目录>\n"
            "```\n\n"
            "摄取完成后刷新本页即可看到每个文件的阶段耗时瀑布。"
        )
        return

    # Melt the per-stage columns into long form for Altair stacked bars.
    import pandas as pd

    rows_dicts = [r.to_dict() for r in rows]
    # Only files that actually loaded get bars (top_n), skipped ones are listed
    # below; the detail dataframe keeps every file (T2).
    chart_rows = [r for r in rows if not r.skipped][:_CHART_TOP_N]
    chart_df = pd.DataFrame([r.to_dict() for r in chart_rows])
    if not chart_df.empty:
        melted = chart_df.melt(
            id_vars=["file", "total_ms"],
            value_vars=[c for c in STAGE_MS_KEYS if c in chart_df.columns],
            var_name="stage_ms",
            value_name="elapsed_ms",
        )
        melted["stage"] = melted["stage_ms"].map(_STAGE_LABELS)
        melted = melted[melted["elapsed_ms"] > 0]

        if not melted.empty:
            import altair as alt

            chart = (
                alt.Chart(melted)
                .mark_bar()
                .encode(
                    y=alt.Y("file:N", sort="-x", title="文件"),
                    x=alt.X("elapsed_ms:Q", title="耗时 (ms)"),
                    color=alt.Color("stage:N", title="阶段"),
                    tooltip=["file:N", "stage:N", "elapsed_ms:Q"],
                )
                .properties(title="每文件阶段耗时（top N）", height=min(len(chart_df) * 24 + 60, 600))
            )
            st.altair_chart(chart, width="stretch")
        else:
            st.markdown("已摄取的文件无有效阶段耗时。")

    skipped_rows = [r for r in rows if r.skipped]
    if skipped_rows:
        st.subheader("跳过的文件")
        st.markdown(
            "\n".join(f"- `{r.file}`（跳过）" for r in skipped_rows)
        )

    # ── 逐条摄取记录（含 ✂️ 分块全文回放）────────────────────────────
    # Each ingestion trace becomes an expander. A trace is run-scoped (one
    # ingest() may cover many files), so each FILE inside the trace gets its own
    # expander: header = file + chunk count + total time; body = key metrics +
    # per-stage timing + the split stage's per-chunk full texts. Only traces
    # recorded after the pipeline started carrying chunk text (rich split data)
    # show the ✂️ list; older traces degrade to a hint instead of an empty block.
    # The count control makes the newest-N cap explicit (no silent truncation).
    rich_traces = [
        t for t in traces
        if t.get("trace_type") == "ingestion"
        and isinstance(t.get("stages"), list)
        and t.get("stages")
    ]
    rich_traces.sort(key=lambda t: str(t.get("started_at", "")), reverse=True)

    if rich_traces:
        st.subheader("逐条摄取记录")
        # The count control makes the newest-N cap explicit (no silent
        # truncation). Default covers all records when the total is under the
        # smallest option; otherwise show the latest 50.
        _SHOW_OPTIONS = [10, 25, 50, 100, 200]
        default_show = (
            len(rich_traces)
            if len(rich_traces) in _SHOW_OPTIONS
            else next((n for n in _SHOW_OPTIONS if n >= len(rich_traces)), _SHOW_OPTIONS[-1])
        )
        n_traces = st.select_slider(
            "显示最近多少条摄取记录",
            options=_SHOW_OPTIONS,
            value=default_show,
            help="仅控制本页回放条数；上方瀑布与下方明细始终覆盖全部记录。",
        )
        for trace in rich_traces[:n_traces]:
            trace_id = trace.get("trace_id", "?")
            started = str(trace.get("started_at", ""))[:19]
            for file_entry in group_files_by_stage_data(trace):
                stages = file_entry.get("stages", [])
                by_stage = {s.get("stage"): s.get("data") for s in stages if isinstance(s, dict)}
                load_d = by_stage.get("load", {})
                split_d = by_stage.get("split", {})
                embed_d = by_stage.get("embed", {})
                file_name = file_entry.get("file") or trace_id
                chunk_count = split_d.get("chunk_count", 0)
                if not load_d and not split_d:
                    # a skip/integrity_skip-only entry: no chunk metrics to show
                    reason = next(
                        (s.get("data", {}).get("reason") for s in stages
                         if s.get("stage") in ("skip", "integrity_skip") and s.get("data")),
                        "跳过",
                    )
                    st.markdown(f"⏭️ **{file_name}** · 跳过（{reason}）")
                    continue
                total_ms = trace.get("total_elapsed_ms", 0)
                total_ms_label = f"{float(total_ms):.0f}" if isinstance(total_ms, (int, float)) else "—"
                header = f"📄 **{file_name}** · {chunk_count} 块 · {started}"
                with st.expander(header):
                    # key metrics
                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("字数", f"{load_d.get('text_length', 0):,}")
                    c2.metric("分块", chunk_count)
                    c3.metric("平均块长", f"{split_d.get('avg_chunk_size', 0)}")
                    c4.metric("向量", embed_d.get("chunks", embed_d.get("added", "—")))
                    c5.metric("该 run 总耗时", total_ms_label)

                    # per-stage timing row (guard dirty non-numeric elapsed_ms)
                    timings: list[str] = []
                    for s in stages:
                        if not isinstance(s, dict):
                            continue
                        ms = s.get("elapsed_ms")
                        if isinstance(ms, (int, float)):
                            timings.append(f"`{s.get('stage')}` {ms:.0f}ms")
                    if timings:
                        st.caption(f"阶段耗时：{' · '.join(timings)}")

                    # ✂️ split-stage chunk texts
                    chunks = split_d.get("chunks")
                    if isinstance(chunks, list) and chunks:
                        st.markdown("**✂️ 分块结果**")
                        for i, chunk in enumerate(chunks):
                            if not isinstance(chunk, dict):
                                continue
                            cid = chunk.get("chunk_id", f"#{i + 1}")
                            char_len = chunk.get("char_len", len(str(chunk.get("text", ""))))
                            label = f"📝 Chunk #{i + 1} — `{cid}` — {char_len} 字"
                            with st.expander(label, expanded=(i < 2)):
                                st.text(chunk.get("text", ""))
                    elif split_d:
                        st.caption("（该条记录未含分块全文 —— 旧版 trace，重新摄取后即可回放）")

    st.subheader("明细")
    st.dataframe(rows_dicts)
