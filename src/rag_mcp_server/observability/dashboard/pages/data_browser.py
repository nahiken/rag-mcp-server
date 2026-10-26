"""Data browser page — browse documents and their chunks (ADR 0014 ext).

Pure data via ``data_service.browse_documents`` (streamlit-free), rendered with
streamlit. No "clear all" danger zone (Q8 — kept as an extension point).
"""

from __future__ import annotations

from typing import Any

from rag_mcp_server.config.settings import load_settings
from rag_mcp_server.observability.dashboard.services.data_service import browse_documents


def render_data_browser(settings: dict[str, Any] | None = None) -> None:
    """Render the data browser page (docs -> chunk cards + metadata).

    ``settings`` is optional for testability (AppTest passes an explicit dict);
    the app calls it with no args and it falls back to the repo config.
    """
    import streamlit as st

    st.title("RAG Knowledge Hub — 数据浏览")

    settings = settings if settings is not None else load_settings()
    docs = browse_documents(settings)

    if not docs:
        st.markdown(
            "知识库为空或向量库不可用。请先摄取文档：\n\n"
            "```bash\nPYTHONPATH=src python -m rag_mcp_server ingest <目录>\n"
            "```"
        )
        return

    st.markdown(f"共 **{len(docs)}** 篇文档")
    for doc in docs:
        with st.expander(f"{doc['document_id']}（chunks={doc['chunk_count']}）"):
            st.markdown(f"**来源：** `{doc['source_path']}`")
            chunks = doc.get("chunks", [])
            for chunk in chunks:
                st.markdown(f"**{chunk['chunk_id']}**")
                st.text(chunk.get("text", ""))
                if chunk.get("metadata"):
                    with st.expander("metadata", expanded=False):
                        st.json(chunk["metadata"])
