"""Unit tests for retrieval-time filters (ADR 0016) through rag_service.query.

Filters are applied in retrieval (Chroma `where` / filtered corpus), not
post-hoc — RetrievalResult.metadata only carries image_refs (P10.6 T3), so the
filter must happen on the store/chunk level before results are shaped.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rag_mcp_server.application.rag_service import ingest, query
from rag_mcp_server.ingestion.pipeline import IngestPipeline
from rag_mcp_server.libs.embedding.fake import FakeEmbedding
from rag_mcp_server.libs.vector_store.chroma_store import ChromaVectorStore


def _pipeline(index_dir: Path, collection: str = "filter_collection") -> IngestPipeline:
    embedding = FakeEmbedding(dim=4)
    store = ChromaVectorStore(persist_dir=None, collection=collection)
    return IngestPipeline(embedding=embedding, store=store, index_dir=index_dir, collection=collection)


def _seed(root: Path, index_dir: Path, collection: str = "filter_collection") -> IngestPipeline:
    """Ingest two documents with distinct source_path / file_type / tags."""
    root.mkdir(parents=True, exist_ok=True)
    pipeline = _pipeline(index_dir, collection)
    (root / "finance.md").write_text("财务季度报表，营收与利润数据。", encoding="utf-8")
    (root / "tech.md").write_text("技术架构说明，RAG 与检索系统。", encoding="utf-8")
    ingest(root, base=root, index_dir=index_dir, pipeline=pipeline, trace_dir=root, collection=collection)
    return pipeline


class QueryFilterTests(unittest.TestCase):
    def test_dense_filter_by_source_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index_dir = root / "out"
            pipeline = _seed(root, index_dir)
            # filter to finance.md only — tech.md chunks must not appear
            results = query(
                "检索", index_dir=index_dir, top_k=10, pipeline=pipeline, trace_dir=root,
                strategy="dense", filters={"source_path": "finance.md"},
            )
            self.assertTrue(results)
            for r in results:
                self.assertEqual(r.source_path, "finance.md")

    def test_sparse_filter_by_source_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index_dir = root / "out"
            pipeline = _seed(root, index_dir)
            # BM25 matches exact tokens: "RAG" is a token in tech.md (jieba
            # groups "检索系统" into one token, so "检索" would not match).
            results = query(
                "RAG", index_dir=index_dir, top_k=10, pipeline=pipeline, trace_dir=root,
                strategy="sparse", filters={"source_path": "tech.md"},
            )
            self.assertTrue(results)
            for r in results:
                self.assertEqual(r.source_path, "tech.md")

    def test_hybrid_filter_by_source_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index_dir = root / "out"
            pipeline = _seed(root, index_dir)
            results = query(
                "检索", index_dir=index_dir, top_k=10, pipeline=pipeline, trace_dir=root,
                strategy="hybrid", filters={"source_path": "finance.md"},
            )
            self.assertTrue(results)
            for r in results:
                self.assertEqual(r.source_path, "finance.md")

    def test_no_filter_returns_both(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index_dir = root / "out"
            pipeline = _seed(root, index_dir)
            results = query(
                "检索", index_dir=index_dir, top_k=10, pipeline=pipeline, trace_dir=root,
                strategy="dense", filters=None,
            )
            sources = {r.source_path for r in results}
            self.assertTrue({"finance.md", "tech.md"}.issubset(sources))

    def test_collection_param_selects_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index_dir = root / "out"
            # ingest into two collections with different content
            _seed(root / "c1", index_dir, collection="collection_a")
            _seed(root / "c2", index_dir, collection="collection_b")
            p1 = _pipeline(index_dir, "collection_a")
            p2 = _pipeline(index_dir, "collection_b")
            # collection scope is proven by the document_id prefix
            r1 = query("检索", index_dir=index_dir, top_k=10, pipeline=p1, trace_dir=root, strategy="dense", collection="collection_a")
            self.assertTrue(r1)
            self.assertTrue(all(r.chunk_id.startswith("collection_a::") for r in r1))
            r2 = query("检索", index_dir=index_dir, top_k=10, pipeline=p2, trace_dir=root, strategy="dense", collection="collection_b")
            self.assertTrue(all(r.chunk_id.startswith("collection_b::") for r in r2))

    # -- 语法接线（spec T1：过滤语法解析在 service 层）--------------------------
    def test_query_text_source_syntax_wires_into_filter(self) -> None:
        """查询文本里的 `source:xxx` 语法应被 service 层解析并过滤（ADR 0016 D18）。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index_dir = root / "out"
            pipeline = _seed(root, index_dir)
            # 语法在查询文本里，不传 filters 参数——验证 service 层接线
            results = query(
                "source:finance.md 财务", index_dir=index_dir, top_k=10,
                pipeline=pipeline, trace_dir=root, strategy="dense",
            )
            self.assertTrue(results)
            for r in results:
                self.assertEqual(r.source_path, "finance.md")
            # clean_text 不应含 source: 语法（选择器被剥离，不污染检索）
            self.assertNotIn("source:", results[0].text if results else "")

    def test_query_text_collection_syntax_selects_collection(self) -> None:
        """查询文本里的 `collection:xxx` 语法应选集合（参数优先，无参数时生效）。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index_dir = root / "out"
            _seed(root / "c1", index_dir, collection="collection_a")
            _seed(root / "c2", index_dir, collection="collection_b")
            # 不传 collection 参数，用文本语法选 collection_a
            results = query(
                "collection:collection_a 检索", index_dir=index_dir, top_k=10,
                pipeline=_pipeline(index_dir, "collection_a"), trace_dir=root,
                strategy="dense",
            )
            self.assertTrue(results)
            for r in results:
                self.assertTrue(r.chunk_id.startswith("collection_a::"), r.chunk_id)

    def test_explicit_filters_win_over_text_syntax(self) -> None:
        """显式 filters 参数优先于文本语法（per-key，ADR 0016 D18）。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index_dir = root / "out"
            pipeline = _seed(root, index_dir)
            # 文本语法说 source:tech.md，但显式 filters 指定 finance.md——后者胜
            results = query(
                "source:tech.md 检索", index_dir=index_dir, top_k=10,
                pipeline=pipeline, trace_dir=root, strategy="dense",
                filters={"source_path": "finance.md"},
            )
            self.assertTrue(results)
            for r in results:
                self.assertEqual(r.source_path, "finance.md")


if __name__ == "__main__":
    unittest.main()
