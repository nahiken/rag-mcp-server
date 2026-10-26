"""Tests for the SparseBM25 seam — sparse (BM25) retrieval external behaviour.

Red/green contract (docs/spec-p9-hybrid.md, task 7, ticket 07-T1):
- true BM25: Chinese via jieba segmentation, English word tokens; k1=1.5, b=0.75
- Chinese and English keywords both hit relevant chunks
- top_k limits results; score=0 chunks are excluded
- empty query / empty corpus / no match -> empty list, never an exception
- results carry chunk_id/source_path/score/text, sources=["sparse"]

The only test seam is SparseBM25: the corpus is injected (list[Chunk]) and the
retriever never touches the store — the orchestrator builds the corpus from
store.get_all() per query (ADR 0003, rebuilt-from-vstore).
"""

from __future__ import annotations

import pytest

from rag_mcp_server.core.types import Chunk, RetrievalResult
from rag_mcp_server.retrieval.sparse_bm25 import SparseBM25


def make_chunk(document_id: str, text: str, order: int = 0, source_path: str = "notes.txt") -> Chunk:
    return Chunk(
        document_id=document_id,
        text=text,
        source_path=source_path,
        order=order,
        metadata={"file_type": "txt"},
    )


@pytest.fixture
def chunks() -> list[Chunk]:
    return [
        make_chunk(
            "rag-notes",
            "RAG 使用关键词检索 baseline，检索结果按 score 排序。",
            order=0,
            source_path="docs/rag-notes.md",
        ),
        make_chunk(
            "mcp-notes",
            "MCP 是 Model Context Protocol，用于连接工具与数据。",
            order=0,
            source_path="docs/mcp-notes.md",
        ),
        make_chunk(
            "rag-notes",
            "关键词检索会返回 top 3 结果，并附带引用来源。",
            order=1,
            source_path="docs/rag-notes.md",
        ),
    ]


def _search(chunks: list[Chunk], query: str, top_k: int = 3) -> list[RetrievalResult]:
    return SparseBM25(chunks).search(query, top_k=top_k)


def test_cn_keyword_hits_relevant_chunk(chunks: list[Chunk]) -> None:
    results = _search(chunks, "检索")
    assert results, "中文关键词应命中 chunk"
    assert results[0].chunk_id == "rag-notes::0"


def test_en_keyword_hits_relevant_chunk(chunks: list[Chunk]) -> None:
    results = _search(chunks, "MCP")
    assert results
    assert results[0].chunk_id == "mcp-notes::0"


def test_en_acronym_token_kept_whole(chunks: list[Chunk]) -> None:
    # "MCP" must match as a whole token, not be split into M/C/P
    results = _search(chunks, "MCP")
    assert any("MCP" in r.text for r in results)


def test_chunk_with_more_keyword_hits_scores_higher(chunks: list[Chunk]) -> None:
    # "检索" appears twice in chunk0 ("关键词检索" + "检索结果"), once in chunk2
    results = _search(chunks, "检索")
    assert results[0].chunk_id == "rag-notes::0"
    assert results[0].score > results[1].score


def test_top_k_limits_results(chunks: list[Chunk]) -> None:
    results = _search(chunks, "检索", top_k=1)
    assert len(results) == 1


def test_zero_score_chunks_are_excluded(chunks: list[Chunk]) -> None:
    results = _search(chunks, "完全不存在的词zzz")
    assert results == []


def test_no_result_returns_empty_list_not_exception(chunks: list[Chunk]) -> None:
    results = _search(chunks, "zzz-not-found")
    assert isinstance(results, list)
    assert results == []


def test_empty_query_returns_empty_list(chunks: list[Chunk]) -> None:
    assert _search(chunks, "") == []
    assert _search(chunks, "   ") == []


def test_empty_corpus_returns_empty_list() -> None:
    assert _search([], "检索") == []


def test_corpus_of_zero_term_chunks_does_not_divide_by_zero() -> None:
    # chunks whose text tokenizes to zero terms: avgdl = 0 must not crash _score
    chunks = [
        make_chunk("punct", "!!! ... ---"),
        make_chunk("spaces", "   \t  "),
    ]
    assert _search(chunks, "检索") == []


def test_negative_top_k_fails_fast() -> None:
    chunks = [make_chunk("a", "RAG alpha"), make_chunk("b", "RAG beta")]
    with pytest.raises(ValueError):
        _search(chunks, "RAG", top_k=-1)
    with pytest.raises(ValueError):
        _search(chunks, "RAG", top_k=0)


def test_cjk_punctuation_is_dropped_from_tokens() -> None:
    from rag_mcp_server.retrieval.sparse_bm25 import tokenize

    assert tokenize("用于连接，工具。") == ["用于", "连接", "工具"]


def test_cjk_punctuation_query_matches_nothing() -> None:
    # a query consisting only of CJK punctuation must not hit any chunk
    chunks = [make_chunk("a", "连接工具。"), make_chunk("b", "其他内容")]
    assert _search(chunks, "。") == []


def test_result_carries_fields_and_sources(chunks: list[Chunk]) -> None:
    results = _search(chunks, "MCP")
    result = results[0]
    assert result.source_path == "docs/mcp-notes.md"
    assert result.text.startswith("MCP")
    assert isinstance(result.score, float)
    assert result.sources == ["sparse"]


def test_each_result_traces_back_to_a_real_chunk(chunks: list[Chunk]) -> None:
    """引用来源可追溯：返回的 chunk_id/source_path 必须对应真实 chunk。"""
    results = _search(chunks, "检索")
    by_id = {c.chunk_id: c for c in chunks}
    for result in results:
        assert result.chunk_id in by_id
        chunk = by_id[result.chunk_id]
        assert result.source_path == chunk.source_path
        assert result.text == chunk.text
