"""Evaluation panel page — run the offline Custom gate and show results (ADR 0014 ext).

Wires the offline Custom evaluator (hit_rate/mrr, ADR 0013) via EvalRunner over
the golden test set. Ragas results are intentionally NOT surfaced (Q9 — Ragas is
an optional Answer extension, not the default offline gate). Pure logic lives in
observability/evaluation; this page only assembles search_fn + evaluator and
renders the report.
"""

from __future__ import annotations

from pathlib import Path


def render_evaluation_panel() -> None:
    """Render the evaluation panel page (run golden gate, show metrics)."""
    import json

    import streamlit as st

    st.title("RAG Knowledge Hub — 评估面板")

    from rag_mcp_server.application import rag_service
    from rag_mcp_server.config.settings import load_settings
    from rag_mcp_server.libs.evaluator.custom import CustomEvaluator
    from rag_mcp_server.observability.evaluation.eval_runner import (
        EvalRunner,
        load_golden_test_set,
    )

    settings = load_settings()
    default_golden = Path("tests/fixtures/golden_test_set.json")

    golden_path = st.text_input("Golden test set 路径", value=str(default_golden))
    strategy = st.selectbox("检索策略", ["sparse", "hybrid", "dense"])
    top_k = st.number_input("top_k", min_value=1, max_value=20, value=3, step=1)

    if st.button("运行评估", type="primary"):
        path = Path(golden_path)
        if not path.is_file():
            st.error(f"golden test set 不存在：{path}")
            return
        try:
            test_set = load_golden_test_set(path)
        except Exception as exc:  # noqa: BLE001 — a bad golden file is a readable error
            st.error(f"golden test set 解析失败：{exc}")
            return

        col = settings.get("vector_store", {}).get("collection", "default")

        def search_fn(query: str):
            return rag_service.query(query, top_k=top_k, strategy=strategy, collection=col)

        evaluator = CustomEvaluator()
        runner = EvalRunner()
        try:
            report = runner.run(test_set, evaluator, search_fn)
        except Exception as exc:  # noqa: BLE001 — a failed run is a readable error
            st.error(f"评估失败：{exc}")
            return

        metrics = {k: round(v, 4) for k, v in report.items() if k != "per_query"}
        st.subheader("聚合指标")
        st.json(metrics)

        st.subheader("单查询明细")
        st.dataframe(report.get("per_query", []))
