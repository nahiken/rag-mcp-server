"""Tests for the hybrid retrieval seam — DenseRetriever + HybridRetriever.

Red/green contract (docs/spec-p9-hybrid.md, task 8, ticket 08-T2):
- DenseRetriever is a thin adapter over a BaseVectorStore: search(query_embedding,
  top_k) maps VectorRecord -> RetrievalResult, source_path from metadata, score
  straight from the store's similarity, sources=["dense"]
- rrf_fuse is the pure RRF algorithm: fusion score = Σ 1/(rrf_k + rank); equal
  scores tie-break by chunk_id ascending (stable ordering)
- HybridRetriever runs dense + sparse in parallel (each takes top_k*3
  candidates), fuses with rrf_fuse, truncates back to top_k
- fusion score goes into score; sources lists the single routes that actually
  appeared (["dense"], ["sparse"], or ["dense","sparse"])
- no explicit fallback: sparse no-hit -> RRF naturally degrades to pure dense
  ordering; store empty -> []
- empty embedding/empty query -> []

Seams: rrf_fuse (pure algorithm), DenseRetriever over a real BaseVectorStore
(temporary Chroma), HybridRetriever wiring store + injected sparse corpus.
All offline.
"""

from __future__ import annotations

import pytest

from rag_mcp_server.core.types import Chunk, RetrievalResult
from rag_mcp_server.libs.vector_store.base import VectorRecord
from rag_mcp_server.libs.vector_store.chroma_store import ChromaVectorStore
from rag_mcp_server.retrieval.dense import DenseRetriever
from rag_mcp_server.retrieval.hybrid import HybridRetriever, rrf_fuse
from rag_mcp_server.retrieval.sparse_bm25 import SparseBM25


def make_chunk(document_id: str, text: str, order: int = 0, source_path: str = "notes.txt") -> Chunk:
    return Chunk(
        document_id=document_id,
        text=text,
        source_path=source_path,
        order=order,
        metadata={"file_type": "txt"},
    )


def _record(chunk_id: str, text: str, source_path: str = "doc.txt") -> VectorRecord:
    return VectorRecord(
        chunk_id=chunk_id,
        metadata={
            "document_id": chunk_id.split("::")[0],
            "source_path": source_path,
            "chunk_id": chunk_id,
            "content_hash": f"hash-{chunk_id}",
        },
        text=text,
    )


@pytest.fixture
def store():
    """Temp Chroma populated with three chunks at distinct one-hot positions."""
    records = [
        _record("rag::0", "RAG 使用关键词检索 baseline", source_path="docs/rag.md"),
        _record("mcp::0", "MCP 是 Model Context Protocol", source_path="docs/mcp.md"),
        _record("both::0", "检索与 MCP 都提到", source_path="docs/both.md"),
    ]
    embeddings = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    store = ChromaVectorStore(persist_dir=None, collection="test_hybrid_isolated")
    store.add(records, embeddings)
    return store


# -- rrf_fuse (pure algorithm) ---------------------------------------------
def test_rrf_fuse_scores_sum_over_routes():
    # a appears in both routes (at rank 2 in each) -> higher fusion score
    fused = rrf_fuse([["b", "a"], ["c", "a"]], top_k=5, rrf_k=60)
    # a = 1/62 + 1/62 ; b = 1/61 ; c = 1/61
    assert fused[0][0] == "a"
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 62, rel=1e-9)


def test_rrf_fuse_tie_breaks_by_chunk_id_ascending():
    # b and c both = 1/61 -> tie; chunk_id ascending -> b before c
    fused = rrf_fuse([["b", "a"], ["c", "a"]], top_k=5, rrf_k=60)
    assert [i for i, _ in fused] == ["a", "b", "c"]


def test_rrf_fuse_truncates_to_top_k():
    fused = rrf_fuse([["a", "b", "c", "d"], ["a"]], top_k=2, rrf_k=60)
    assert len(fused) == 2
    assert fused[0][0] == "a"


def test_rrf_fuse_empty_routes_returns_empty():
    assert rrf_fuse([[], []], top_k=5, rrf_k=60) == []


# -- DenseRetriever ---------------------------------------------------------
def test_dense_maps_store_results_to_retrieval_results(store):
    retriever = DenseRetriever(store)
    results = retriever.search([0.0, 1.0, 0.0, 0.0], top_k=3)
    assert isinstance(results[0], RetrievalResult)
    assert results[0].chunk_id == "mcp::0"
    assert results[0].source_path == "docs/mcp.md"
    assert results[0].text.startswith("MCP")
    assert isinstance(results[0].score, float)
    assert results[0].sources == ["dense"]


def test_dense_returns_most_similar_first(store):
    retriever = DenseRetriever(store)
    results = retriever.search([1.0, 0.0, 0.0, 0.0], top_k=3)
    assert results[0].chunk_id == "rag::0"
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_dense_empty_embedding_returns_empty(store):
    assert DenseRetriever(store).search([], top_k=3) == []


def test_hybrid_empty_embedding_returns_empty(store):
    # empty query embedding must short-circuit to [] (dense contract), even when
    # sparse would otherwise hit — an un-embedded query is not a sparse search
    hybrid = HybridRetriever(
        dense=DenseRetriever(store),
        sparse=SparseBM25([make_chunk("mcp", "MCP 是 Model Context Protocol")]),
        rrf_k=60,
    )
    assert hybrid.search([], "MCP", top_k=3) == []


# -- HybridRetriever: RRF fusion -------------------------------------------
def test_hybrid_combines_dense_and_sparse_contributions(store):
    sparse_chunks = [
        make_chunk("rag", "RAG 使用关键词检索 baseline", source_path="docs/rag.md"),
        make_chunk("both", "检索与 MCP 都提到", source_path="docs/both.md"),
    ]
    hybrid = HybridRetriever(dense=DenseRetriever(store), sparse=SparseBM25(sparse_chunks), rrf_k=60)
    # dense prefers mcp::0 (embedding [0,1,0,0]); sparse term "MCP" hits both::0
    results = hybrid.search([0.0, 1.0, 0.0, 0.0], query_text="MCP", top_k=3)
    assert results, "hybrid must return fused results"
    # both::0 is hit by BOTH dense (rank 2/3) and sparse (rank 1) -> highest RRF
    assert results[0].chunk_id == "both::0"
    assert results[0].sources == ["dense", "sparse"]
    assert all(isinstance(r.score, float) for r in results)
    assert len(results) <= 3


def test_hybrid_sparse_no_hit_degrades_to_dense(store):
    # a query term that only exists in the dense corpus, not in sparse chunks
    sparse_chunks = [make_chunk("other", "完全不相关的段落", source_path="other.txt")]
    hybrid = HybridRetriever(dense=DenseRetriever(store), sparse=SparseBM25(sparse_chunks), rrf_k=60)
    results = hybrid.search([1.0, 0.0, 0.0, 0.0], query_text="MCP", top_k=3)
    # sparse corpus has no MCP, dense ranks rag::0 first -> pure dense ordering
    assert results[0].chunk_id == "rag::0"
    assert results[0].sources == ["dense"]


def test_hybrid_no_hit_returns_empty():
    # an EMPTY store means neither route can produce anything -> []
    empty = ChromaVectorStore(persist_dir=None, collection="test_hybrid_empty")
    hybrid = HybridRetriever(dense=DenseRetriever(empty), sparse=SparseBM25([]), rrf_k=60)
    assert hybrid.search([0.0, 0.0, 0.0, 1.0], query_text="anything", top_k=3) == []


def test_hybrid_sources_reflect_both_routes(store):
    # a chunk that is hit by BOTH dense and sparse must report ["dense","sparse"]
    sparse_chunks = [
        make_chunk("rag", "RAG 使用关键词检索 baseline", source_path="docs/rag.md"),
        make_chunk("mcp", "MCP 是 Model Context Protocol", source_path="docs/mcp.md"),
    ]
    hybrid = HybridRetriever(dense=DenseRetriever(store), sparse=SparseBM25(sparse_chunks), rrf_k=60)
    results = hybrid.search([0.0, 1.0, 0.0, 0.0], query_text="MCP", top_k=3)
    mcp = [r for r in results if r.chunk_id == "mcp::0"]
    assert mcp and mcp[0].sources == ["dense", "sparse"]


def test_hybrid_truncates_back_to_top_k(store):
    sparse_chunks = [
        make_chunk("rag", "RAG 使用关键词检索 baseline", source_path="docs/rag.md"),
        make_chunk("mcp", "MCP 是 Model Context Protocol", source_path="docs/mcp.md"),
        make_chunk("both", "检索与 MCP 都提到", source_path="docs/both.md"),
    ]
    hybrid = HybridRetriever(dense=DenseRetriever(store), sparse=SparseBM25(sparse_chunks), rrf_k=60)
    results = hybrid.search([0.0, 1.0, 0.0, 0.0], query_text="MCP", top_k=2)
    assert len(results) == 2
