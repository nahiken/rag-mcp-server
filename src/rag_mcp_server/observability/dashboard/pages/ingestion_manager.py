"""Ingestion manager page — upload files and ingest with real progress (ADR 0014 ext).

This page wires streamlit's file uploader to ``rag_service.ingest`` and renders
the on_progress events (load/split/transform/embed/upsert) live. Uploads are
written to a temp dir before ingest so streamlit's UploadedFile never reaches
the loader; the ingest TARGET is the repo's real index_dir, so the evidence
snapshot persists (C3).
"""

from __future__ import annotations

from pathlib import Path


def render_ingestion_manager() -> None:
    """Render the ingestion manager page (upload -> ingest -> progress)."""
    import tempfile

    import streamlit as st

    st.title("RAG Knowledge Hub — 摄取管理")

    from rag_mcp_server.application import rag_service
    from rag_mcp_server.config.settings import load_settings

    settings = load_settings()
    col_name = settings.get("vector_store", {}).get("collection", "default")

    uploaded = st.file_uploader(
        "上传文档（txt / md / pdf）",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
    )
    if not uploaded:
        st.info("上传文件后点击「开始摄取」。")
        return

    if st.button("开始摄取", type="primary"):
        # Uploads are written to a temp dir (streamlit UploadedFile is a
        # file-like object; the loader needs a real path) — but the ingest
        # TARGET is the repo's real index_dir (data/index), so the evidence /
        # chunks.jsonl snapshot and any persisted BM25 corpus survive after the
        # button handler returns (C3). The vector store persists via settings
        # persist_dir regardless of this page.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index_dir = rag_service.DEFAULT_INDEX_DIR
            for up in uploaded:
                (root / up.name).write_bytes(up.getbuffer())
            progress = st.progress(0.0, text="等待开始…")
            stage_text = st.empty()

            def on_progress(stage: str, current: int, total: int) -> None:
                progress.progress(current / max(total, 1))
                stage_text.markdown(f"**{stage}** — {current}/{total}")

            try:
                summary = rag_service.ingest(
                    root,
                    base=root,
                    index_dir=index_dir,
                    on_progress=on_progress,
                    collection=col_name,
                )
            except Exception as exc:  # noqa: BLE001 — a failed ingest shows a readable error
                st.error(f"摄取失败：{exc}")
                return

            st.success(f"导入 {summary.total_files} 个文件，{summary.total_chunks} 个 chunk；跳过 {summary.total_skipped}。")
            if summary.skipped:
                st.markdown("**跳过：**")
                for item in summary.skipped:
                    st.markdown(f"- `{item['path']}`（{item['reason']}）")
