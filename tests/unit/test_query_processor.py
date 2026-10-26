"""Unit tests for query preprocessing (filter syntax parsing).

Covers parse_filters / parse_collection / build_where (ADR 0016 D3/D18):
- metadata filters (source:/tag:/doc_type:) become a where clause
- collection: is a collection selector, stripped from the search text
- unknown key:value tokens stay in the search text as keywords
- build_where returns None for empty filters (existing behaviour unchanged)
"""
from __future__ import annotations

from rag_mcp_server.retrieval.query_processor import (
    build_where,
    parse_collection,
    parse_filters,
)


def test_parse_filters_extracts_source_path():
    clean, filters = parse_filters("source:docs/guide.md 什么是RAG")
    assert clean == "什么是RAG"
    assert filters == {"source_path": "docs/guide.md"}


def test_parse_filters_aliases_source():
    _, filters = parse_filters("src:a.md 检索")
    assert filters == {"source_path": "a.md"}
    _, filters = parse_filters("s:a.md 检索")
    assert filters == {"source_path": "a.md"}


def test_parse_filters_doc_type():
    clean, filters = parse_filters("doc_type:md 检索")
    assert clean == "检索"
    assert filters == {"file_type": "md"}
    _, filters = parse_filters("type:pdf 检索")
    assert filters == {"file_type": "pdf"}


def test_parse_filters_tags_in_clause():
    _, filters = parse_filters("tag:a,b 检索")
    assert filters == {"tags": {"$in": ["a", "b"]}}
    _, filters = parse_filters("tags:x tag:y 检索")
    # both tag: tokens accumulate into one $in clause
    assert filters == {"tags": {"$in": ["x", "y"]}}


def test_parse_filters_unknown_token_stays_keyword():
    clean, filters = parse_filters("color:red 检索")
    # "color:" is not a recognized filter -> stays a search keyword
    assert "color:red" in clean
    assert filters == {}


def test_parse_filters_no_filters():
    clean, filters = parse_filters("什么是 RAG 检索")
    assert clean == "什么是 RAG 检索"
    assert filters == {}


def test_parse_filters_empty_query():
    clean, filters = parse_filters("")
    assert clean == ""
    assert filters == {}


def test_parse_collection_extracts_selector():
    assert parse_collection("collection:finance 报表") == "finance"
    assert parse_collection("col:docs 检索") == "docs"
    assert parse_collection("c:legal 检索") == "legal"


def test_parse_collection_none_when_absent():
    assert parse_collection("普通查询") is None


def test_collection_removed_from_clean_query():
    # collection: must not leak into the search text (it is a selector); other
    # metadata filters (source:) are also stripped
    clean, filters = parse_filters("collection:finance source:a.md 报表")
    assert clean == "报表"
    assert "collection:finance" not in clean
    assert filters == {"source_path": "a.md"}
    assert parse_collection("collection:finance source:a.md 报表") == "finance"


def test_build_where_none_for_empty():
    assert build_where({}) is None
    assert build_where(None) is None


def test_build_where_passthrough():
    assert build_where({"source_path": "a.md"}) == {"source_path": "a.md"}
    assert build_where({"tags": {"$in": ["x"]}}) == {"tags": {"$in": ["x"]}}
