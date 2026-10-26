"""Unit tests for the Agentic RAG tools (ADR 0019).

The five tools are plain MCP-free functions taking injected seams (index dir,
collection, llm) — so list/preview are tested against a temp index + real
chunks, and verify_fact/self_check are tested with a FakeLLM (deterministic,
offline). A server-level test asserts all five are registered.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from rag_mcp_server.application.rag_service import ingest
from rag_mcp_server.ingestion.pipeline import IngestPipeline
from rag_mcp_server.libs.embedding.fake import FakeEmbedding
from rag_mcp_server.libs.llm.fake import FakeLLM
from rag_mcp_server.libs.vector_store.chroma_store import ChromaVectorStore
from rag_mcp_server.mcp_server.tools.agentic import (
    _parse_sufficient,
    _parse_verdict,
    list_documents,
    list_directory,
    preview_document,
    self_check,
    verify_fact,
)

COLLECTION = "test_agentic"


@pytest.fixture
def env():
    """A temp index ingested with two docs; returns (index_dir, collection, pipeline).

    The Chroma collection is derived from tmp_path (unique per test) because
    Chroma's EphemeralClient shares its in-memory backend by collection name —
    a fixed name let records from one test leak into another (known flake).
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # Chroma collection name must be 3-512 [a-zA-Z0-9._-], start AND end
        # with a letter/digit; derive from tmp_path and strip trailing non-alnum.
        col = ("test_agentic_" + root.name.replace("-", "_"))[:50].rstrip("._-")
        if len(col) < 3:
            col = "test_agentic_tmp"
        (root / "docs").mkdir()
        (root / "docs" / "finance.md").write_text(
            "财务季度报表，营收与利润数据，季度销售分析。", encoding="utf-8"
        )
        (root / "docs" / "tech.md").write_text(
            "技术架构说明，RAG 与检索系统，语义检索。", encoding="utf-8"
        )
        embedding = FakeEmbedding(dim=4)
        store = ChromaVectorStore(persist_dir=None, collection=col)
        pipe = IngestPipeline(
            embedding=embedding, store=store, index_dir=root / "out",
            collection=col,
        )
        ingest(root / "docs", base=root / "docs", index_dir=root / "out",
               pipeline=pipe, trace_dir=root, collection=col)
        yield root / "out", col, pipe


# -- list_directory ----------------------------------------------------------
def test_list_directory_lists_collections(env):
    out, col, _ = env
    text = list_directory(str(out))
    assert COLLECTION in text
    assert "集合" in text


def test_list_directory_empty_is_friendly(tmp_path):
    text = list_directory(str(tmp_path / "empty"))
    assert "尚未导入" in text


# -- list_documents ----------------------------------------------------------
def test_list_documents_lists_docs_with_chunk_counts(env):
    out, col, _ = env
    text = list_documents(str(out), col)
    assert "finance-md" in text or "finance" in text
    assert "tech-md" in text or "tech" in text
    assert "chunks=" in text


def test_list_documents_not_ingested_is_friendly(env):
    out, col, _ = env
    text = list_documents(str(out), "nonexistent_collection")
    assert "尚未导入" in text


# -- preview_document --------------------------------------------------------
def test_preview_document_shows_chunks_truncated(env):
    out, col, _ = env
    text = preview_document(f"{col}::finance-md", str(out), col, max_chunks=5)
    assert "finance-md" in text
    assert "财务" in text


def test_preview_document_missing_doc_is_friendly(env):
    out, col, _ = env
    text = preview_document("does-not-exist", str(out), col)
    assert "未找到文档" in text


# -- verify_fact -------------------------------------------------------------
def test_verify_fact_supported_with_fake_llm(env):
    out, col, pipe = env
    llm = FakeLLM(responses={
        "": "{}",  # echo path unused; responses keyed on full content below
    })
    llm._chat = lambda messages: '{"supported": true, "reason": "证据支持该断言"}'
    text = verify_fact("财务", str(out), col, top_k=3, llm=llm, pipeline=pipe)
    assert "支持" in text
    assert "财务季度报表" in text


def test_verify_fact_not_supported_returns_evidence(env):
    out, col, pipe = env
    llm = FakeLLM()
    llm._chat = lambda messages: '{"supported": false, "reason": "证据与此断言无关"}'
    text = verify_fact("不存在的断言", str(out), col, top_k=3, llm=llm, pipeline=pipe)
    assert "不支持" in text
    assert "证据" in text


def test_verify_fact_llm_failure_degrades(env):
    out, col, pipe = env

    def boom(messages):
        raise RuntimeError("llm down")

    llm = FakeLLM()
    llm._chat = boom
    text = verify_fact("财务", str(out), col, top_k=3, llm=llm, pipeline=pipe)
    assert "无法判读" in text
    assert "证据" in text  # degraded response still shows evidence


def test_verify_fact_no_evidence_is_friendly(env):
    out, col, pipe = env
    # an empty collection (no evidence dir) -> no evidence -> friendly, no LLM
    text = verify_fact("zxqjkw", str(out), "empty_agentic_col", top_k=3, llm=FakeLLM(), pipeline=pipe)
    assert "没有检索到" in text or "尚未导入" in text


# -- self_check --------------------------------------------------------------
def test_self_check_sufficient_with_fake_llm(env):
    out, col, pipe = env
    llm = FakeLLM()
    llm._chat = lambda messages: '{"sufficient": true, "gap": ""}'
    text = self_check("财务", str(out), col, top_k=3, llm=llm, pipeline=pipe)
    assert "充分" in text
    assert "证据" in text


def test_self_check_insufficient_with_gap(env):
    out, col, pipe = env
    llm = FakeLLM()
    llm._chat = lambda messages: '{"sufficient": false, "gap": "缺少历史对比数据"}'
    text = self_check("财务", str(out), col, top_k=3, llm=llm, pipeline=pipe)
    assert "不足" in text
    assert "缺少历史对比数据" in text


def test_self_check_llm_failure_degrades(env):
    out, col, pipe = env

    def boom(messages):
        raise RuntimeError("llm down")

    llm = FakeLLM()
    llm._chat = boom
    text = self_check("财务", str(out), col, top_k=3, llm=llm, pipeline=pipe)
    assert "无法判读" in text
    assert "证据" in text


# -- strict JSON parsing -----------------------------------------------------
def test_parse_verdict_strict():
    assert _parse_verdict('{"supported": true, "reason": "ok"}') == {
        "supported": True, "reason": "ok",
    }
    assert _parse_verdict('```json\n{"supported": false, "reason": "x"}\n```') == {
        "supported": False, "reason": "x",
    }
    with pytest.raises(ValueError):
        _parse_verdict("not json")
    with pytest.raises(ValueError):
        _parse_verdict('{"supported": "yes", "reason": "x"}')  # non-bool


def test_parse_sufficient_strict():
    assert _parse_sufficient('{"sufficient": true, "gap": ""}') == {
        "sufficient": True, "gap": "",
    }
    with pytest.raises(ValueError):
        _parse_sufficient('{"sufficient": "yes"}')


# -- registration ------------------------------------------------------------
def test_server_registers_all_five_tools():
    import asyncio

    from rag_mcp_server.mcp_server.server import server

    names = {t.name for t in asyncio.run(server.list_tools())}
    assert {"list_directory", "list_documents", "preview_document",
            "verify_fact", "self_check"}.issubset(names)
