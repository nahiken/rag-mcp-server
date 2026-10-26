"""Overview page — configuration cards + knowledge base stats.

Pure rendering: config comes from ``load_settings()`` (already merged over the
code defaults), stats from ``data_service.knowledge_stats`` (vector store, or
zero cards when the store cannot be built). ``trace.enabled=false`` surfaces a
warning so the empty traces page is not a surprise (US 3).

streamlit is imported inside ``render_overview`` (not at module top level, ADR
0014) so importing this module never pulls in the runtime.
"""

from __future__ import annotations

from typing import Any

from rag_mcp_server.config.settings import load_settings
from rag_mcp_server.observability.dashboard.services.data_service import knowledge_stats, summary


def _section_label(section: dict[str, Any], key: str) -> str:
    value = section.get(key, "")
    return str(value) if value is not None else ""


def render_overview(settings: dict[str, Any] | None = None) -> None:
    """Render the overview page.

    ``settings`` is optional for testability (AppTest drives this function with
    explicit settings so the render is hermetic); the app calls it with no args
    and it falls back to the repo configuration.
    """
    import streamlit as st

    if settings is None:
        settings = load_settings()

    st.title("RAG Knowledge Hub — 总览")

    trace = settings.get("trace", {})
    if trace.get("enabled") is False:
        st.warning("追踪已关闭（settings `trace.enabled: false`），ingestion_traces 页将无数据。")

    st.subheader("配置")
    c1, c2, c3 = st.columns(3)

    llm = settings.get("llm", {})
    c1.markdown(
        "**LLM**\n\n"
        f"- provider: `{_section_label(llm, 'provider')}`\n"
        f"- base_url: `{_section_label(llm, 'base_url')}`\n"
        f"- model: `{_section_label(llm, 'model')}`"
    )

    embedding = settings.get("embedding", {})
    c2.markdown(
        "**Embedding**\n\n"
        f"- provider: `{_section_label(embedding, 'provider')}`\n"
        f"- base_url: `{_section_label(embedding, 'base_url')}`\n"
        f"- model: `{_section_label(embedding, 'model')}`"
    )

    rerank = settings.get("rerank", {})
    c3.markdown(
        "**Reranker**\n\n"
        f"- enabled: `{rerank.get('enabled', False)}`\n"
        f"- provider: `{_section_label(rerank, 'provider')}`"
    )

    st.subheader("知识库")
    stats = knowledge_stats(settings)
    cards = summary(stats)
    cols = st.columns(max(len(cards), 1))
    for col, card in zip(cols, cards):
        col.metric(card["label"], card["value"])
