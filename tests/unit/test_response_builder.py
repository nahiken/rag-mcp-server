"""Unit tests for the Markdown rich-response builder (ADR 0017).

Covers build_citations / build_markdown / build_references_json — pure
functions that MCP query tools use to return a Markdown report with [n]
markers plus a machine-readable JSON references block.
"""
from __future__ import annotations

import json

from rag_mcp_server.core.response.response_builder import (
    build_citations,
    build_markdown,
    build_references_json,
)
from rag_mcp_server.core.types import RetrievalResult


def _result(chunk_id: str, source: str, text: str, score: float = 0.9, page=None) -> RetrievalResult:
    meta = {"page": page} if page is not None else {}
    return RetrievalResult(
        chunk_id=chunk_id,
        source_path=source,
        score=score,
        text=text,
        sources=["dense"],
        metadata=meta,
    )


def test_build_citations_numbered_and_source():
    results = [
        _result("d::0", "a.md", "第一段内容。", score=0.9),
        _result("d::1", "b.md", "第二段内容。", score=0.8),
    ]
    citations = build_citations(results)
    assert [c.index for c in citations] == [1, 2]
    assert citations[0].source == "a.md"
    assert citations[0].chunk_id == "d::0"
    assert citations[1].source == "b.md"


def test_build_citations_page_from_metadata():
    results = [_result("d::0", "a.md", "内容", score=0.9, page=3)]
    c = build_citations(results)[0]
    assert c.page == 3
    assert c.to_dict()["page"] == 3


def test_build_citations_snippet_truncated():
    long_text = "词 " * 400  # > SNIPPET_MAX (300)
    results = [_result("d::0", "a.md", long_text, score=0.9)]
    c = build_citations(results)[0]
    assert len(c.snippet) <= 300


def test_build_citations_empty():
    assert build_citations([]) == []


def test_build_markdown_has_markers_and_references():
    results = [
        _result("d::0", "a.md", "第一段内容。", score=0.9),
        _result("d::1", "b.md", "第二段内容。", score=0.8),
    ]
    citations = build_citations(results)
    md = build_markdown("检索", results, citations)
    assert "## 检索结果" in md
    assert "**检索**" in md
    assert "### [1]" in md
    assert "### [2]" in md
    assert "## 引用来源" in md
    assert "[1] `a.md`" in md
    assert "[2] `b.md`" in md


def test_build_references_json_is_valid_json():
    results = [_result("d::0", "a.md", "内容", score=0.91)]
    citations = build_citations(results)
    block = build_references_json(citations, "检索", len(results))
    # the block embeds a fenced JSON; extract between ```json and ```
    payload_str = block.split("```json\n")[1].rsplit("\n```", 1)[0]
    payload = json.loads(payload_str)
    assert payload["query"] == "检索"
    assert payload["result_count"] == 1
    assert payload["citations"][0]["chunk_id"] == "d::0"
    assert payload["citations"][0]["source"] == "a.md"
    assert payload["citations"][0]["score"] == round(0.91, 4)
