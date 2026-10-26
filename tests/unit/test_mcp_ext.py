"""Tests for the MCP tool seam — query_knowledge_hub + semantic_search.

Red/green contract (docs/spec-p9-m4m5.md, task 10, ticket 10-T1):
- query_knowledge_hub returns vector-retrieval results with traceable citations
- query_knowledge_hub accepts strategy=None / top_k=None (None -> config default)
- semantic_search is a new tool forcing dense strategy, same human-readable format
- both tools reuse a _result_text_blocks helper and the "not ingested" guard
- invalid strategy propagates as an error (fail-fast); no-hit returns friendly string
- tools never duplicate business logic (only pass through rag_service.query)
- P10.6 T3: tools return list[ContentBlock]; a result with image_refs appends
  ImageContent blocks; text-only results stay unchanged (US10/US11)

The only test seam is the MCP tool public entry (async functions registered via
@server.tool). Tests call the async tool functions directly — no stdio client
needed (the e2e stdio verification lives in test_real_mcp_server.py).
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from rag_mcp_server.core.errors import ConfigError
from rag_mcp_server.core.types import RetrievalResult
from rag_mcp_server.ingestion.pipeline import IngestPipeline
from rag_mcp_server.libs.embedding.fake import FakeEmbedding
from rag_mcp_server.libs.vector_store.chroma_store import ChromaVectorStore
from rag_mcp_server.mcp_server.server import (
    _is_ingested,
    get_document_summary,
    list_collections,
    query_knowledge_hub,
    semantic_search,
)


@pytest.fixture
def env(monkeypatch):
    """A temp index + fake pipeline shared by the tool tests (offline).

    Trace is disabled for this module: the tool tests call the real
    rag_service.query through the MCP server (which passes no trace_dir), so an
    enabled trace would write repo logs/ on every run. These are tool tests,
    not trace tests (US 14: tests isolate the trace dir).
    """
    import rag_mcp_server.config.settings as settings_mod
    from rag_mcp_server.config.settings import DEFAULTS

    def fake_load_settings():
        data = {k: dict(v) if isinstance(v, dict) else v for k, v in DEFAULTS.items()}
        data["trace"] = {"enabled": False, "dir": "logs"}
        return data

    monkeypatch.setattr(settings_mod, "load_settings", fake_load_settings)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "docs").mkdir()
        (root / "docs" / "rag-notes.md").write_text(
            "RAG 使用关键词检索 baseline，检索结果按 score 排序。",
            encoding="utf-8",
        )
        (root / "docs" / "mcp-notes.md").write_text(
            "MCP 是 Model Context Protocol，用于连接工具与数据。",
            encoding="utf-8",
        )
        embedding = FakeEmbedding(dim=4)
        store = ChromaVectorStore(persist_dir=None, collection="test_mcp_ext_iso")
        pipeline = IngestPipeline(
            embedding=embedding, store=store, index_dir=root / "out", collection="test_mcp_ext_iso"
        )
        from rag_mcp_server.application.rag_service import ingest

        ingest(root / "docs", base=root / "docs", index_dir=root / "out", pipeline=pipeline)
        yield root / "out"


def _run(coro):
    return asyncio.run(coro)


def _text(run_result):
    """Flatten a list[ContentBlock] into its text blocks (for str-era assertions)."""
    return "".join(getattr(b, "text", str(b)) for b in run_result)


# -- tools/list shape -------------------------------------------------------
def test_semantic_search_is_registered_as_a_tool():
    from rag_mcp_server.mcp_server.server import server

    tool_names = {t.name for t in _run(server.list_tools())}
    assert "query_knowledge_hub" in tool_names
    assert "semantic_search" in tool_names


# -- query_knowledge_hub ----------------------------------------------------
def test_query_knowledge_hub_returns_traceable_citations(env, monkeypatch):
    """工具把 rag_service.query 结果格式化为 Markdown 富响应（ADR 0017）。"""
    index_dir = env
    from rag_mcp_server.application import rag_service
    from rag_mcp_server.core.types import RetrievalResult

    def fake_query(text, index_dir=None, top_k=None, pipeline=None, strategy=None, filters=None, collection="default"):
        return [
            RetrievalResult(
                chunk_id="default::rag-notes-md::0",
                source_path="rag-notes.md",
                score=0.5,
                text="RAG 使用关键词检索 baseline",
                sources=["hybrid"],
            )
        ]

    monkeypatch.setattr(rag_service, "query", fake_query)
    text = _text(_run(query_knowledge_hub("检索", index_dir=str(index_dir), collection="test_mcp_ext_iso")))
    assert "检索" in text
    assert "## 检索结果" in text          # Markdown report header (ADR 0017)
    assert "[1] `rag-notes.md`" in text    # citation marker + source
    # the JSON references block carries the chunk id (not just a loose substring)
    assert "default::rag-notes-md::0" in text


def test_query_knowledge_hub_no_hit_returns_friendly_message(env, monkeypatch):
    from rag_mcp_server.application import rag_service

    monkeypatch.setattr(rag_service, "query", lambda *a, **k: [])
    text = _text(_run(query_knowledge_hub("qxzvw", index_dir=str(env), collection="test_mcp_ext_iso")))
    assert "没有找到匹配" in text


def test_query_knowledge_hub_not_ingested_is_friendly(env):
    # a fresh index dir with no chunks.jsonl -> "请先运行 ingest"
    with tempfile.TemporaryDirectory() as tmp:
        text = _text(_run(query_knowledge_hub("检索", index_dir=str(Path(tmp) / "empty"))))
        assert "尚未导入" in text


def test_query_knowledge_hub_passes_strategy_and_top_k_through(env, monkeypatch):
    """显式 strategy/top_k 必须透传给 rag_service.query（None 用配置）。"""
    from rag_mcp_server.application import rag_service

    received = {}
    monkeypatch.setattr(
        rag_service,
        "query",
        lambda *a, **k: received.update(a=a, k=k) or [],
    )
    _run(query_knowledge_hub("检索", index_dir=str(env), collection="test_mcp_ext_iso", strategy="sparse", top_k=2))
    assert received["k"]["strategy"] == "sparse"
    assert received["k"]["top_k"] == 2


def test_query_knowledge_hub_defaults_use_config(env, monkeypatch):
    """strategy/top_k 未传时必须用 None（让 rag_service.query 读配置）。"""
    from rag_mcp_server.application import rag_service

    received = {}
    monkeypatch.setattr(
        rag_service,
        "query",
        lambda *a, **k: received.update(a=a, k=k) or [],
    )
    _run(query_knowledge_hub("检索", index_dir=str(env), collection="test_mcp_ext_iso"))
    assert received["k"]["strategy"] is None
    assert received["k"]["top_k"] is None


def test_query_knowledge_hub_invalid_strategy_fails_fast(env, monkeypatch):
    """非法 strategy 以 ConfigError 传播（fail-fast，不捕不藏）。"""
    from rag_mcp_server.application import rag_service

    def boom(text, index_dir=None, top_k=None, pipeline=None, strategy=None, filters=None, collection="default"):
        raise ConfigError(f"未知检索策略：{strategy}")

    monkeypatch.setattr(rag_service, "query", boom)
    with pytest.raises(ConfigError):
        _run(query_knowledge_hub("检索", index_dir=str(env), collection="test_mcp_ext_iso", strategy="bogus"))


# -- semantic_search --------------------------------------------------------
def test_semantic_search_returns_results(env, monkeypatch):
    from rag_mcp_server.application import rag_service
    from rag_mcp_server.core.types import RetrievalResult

    monkeypatch.setattr(
        rag_service,
        "query",
        lambda *a, **k: [
            RetrievalResult(
                chunk_id="doc::0", source_path="notes.md", score=0.5, text="检索内容", sources=["dense"]
            )
        ],
    )
    text = _text(_run(semantic_search("检索", index_dir=str(env), collection="test_mcp_ext_iso")))
    assert "检索" in text
    assert "## 检索结果" in text          # rich Markdown (ADR 0017)
    assert "[1] `notes.md`" in text


def test_semantic_search_forces_dense_strategy(env, monkeypatch):
    """semantic_search must always route through dense, ignoring config hybrid."""
    index_dir = env
    from rag_mcp_server.application import rag_service

    calls = []
    original = rag_service.query

    # Route the real query through an isolated in-memory store (the env already
    # ingested) instead of the default settings store: the spy passes
    # pipeline=None, and leaving it None would build a persistent Chroma over
    # the settings persist_dir (P2 — tests must never touch repo data/chroma).
    # Reusing the env's ephemeral pipeline keeps the query real and hermetic.
    from rag_mcp_server.ingestion.pipeline import IngestPipeline
    from rag_mcp_server.libs.embedding.fake import FakeEmbedding
    from rag_mcp_server.libs.vector_store.chroma_store import ChromaVectorStore

    isolated = IngestPipeline(
        embedding=FakeEmbedding(dim=4),
        store=ChromaVectorStore(persist_dir=None, collection="test_mcp_ext_iso"),
        index_dir=Path(index_dir),
    )

    def spy_query(text, index_dir=None, top_k=None, pipeline=None, strategy=None, filters=None, collection="default"):
        calls.append(strategy)
        return original(
            text,
            index_dir=index_dir,
            top_k=top_k,
            pipeline=isolated if pipeline is None else pipeline,
            strategy=strategy,
            filters=filters,
            collection=collection,
        )

    monkeypatch.setattr(rag_service, "query", spy_query)
    _run(semantic_search("检索", index_dir=str(index_dir), collection="test_mcp_ext_iso"))
    assert calls == ["dense"], f"semantic_search 必须强制 dense，实际 {calls}"


# -- _result_blocks / _is_ingested helpers ---------------------------------
def test_result_blocks_renders_markdown_report():
    from rag_mcp_server.mcp_server.server import _result_blocks
    from rag_mcp_server.core.types import RetrievalResult

    results = [
        RetrievalResult(
            chunk_id="doc::0", source_path="notes.md", score=0.5, text="内容", sources=["dense"]
        )
    ]
    blocks = _result_blocks("检索", results)
    # first block is the Markdown report; second is the JSON references block
    assert len(blocks) >= 2
    text = blocks[0].text
    assert "## 检索结果" in text
    assert "### [1]" in text
    assert "[1] `notes.md`" in text
    # references JSON block is valid JSON
    import json
    refs = blocks[1].text.split("```json\n")[1].rsplit("\n```", 1)[0]
    payload = json.loads(refs)
    assert payload["citations"][0]["chunk_id"] == "doc::0"


def test_is_ingested_true_when_chunks_jsonl_exists(env):
    # ADR 0016: evidence is under index_dir/{collection}/chunks.jsonl
    assert _is_ingested(Path(env), collection="test_mcp_ext_iso") is True


def test_is_ingested_false_for_empty_dir():
    with tempfile.TemporaryDirectory() as tmp:
        assert _is_ingested(Path(tmp)) is False


# -- P10.6 T3 multimodal ------------------------------------------------------
def test_query_knowledge_hub_returns_image_content_for_image_refs(env, monkeypatch, tmp_path):
    """命中带 image_refs 的 chunk -> 工具返回 TextContent + ImageContent。"""
    from rag_mcp_server.application import rag_service
    from rag_mcp_server.ingestion.storage.image_storage import ImageStorage
    from mcp.types import ImageContent, TextContent

    # a stored image the assembler can resolve
    storage = ImageStorage(
        db_path=tmp_path / "db" / "image_index.db",
        images_root=tmp_path / "images",
    )
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16  # magic for _detect_mime
    storage.save_image("doc-pdf_p1_1", png, mime_type="image/png", document_id="doc-pdf")

    # make the server resolve build_image_storage to our temp storage
    import rag_mcp_server.mcp_server.server as server_mod

    monkeypatch.setattr(
        server_mod,
        "build_image_storage",
        lambda: ImageStorage(
            db_path=tmp_path / "db" / "image_index.db",
            images_root=tmp_path / "images",
        ),
    )

    def fake_query(text, index_dir=None, top_k=None, pipeline=None, strategy=None, filters=None, collection="default"):
        return [
            RetrievalResult(
                chunk_id="doc-pdf::0",
                source_path="doc.pdf",
                score=0.5,
                text="架构图",
                sources=["hybrid"],
                metadata={"image_refs": ["doc-pdf_p1_1"]},
            )
        ]

    monkeypatch.setattr(rag_service, "query", fake_query)
    blocks = _run(query_knowledge_hub("架构图", index_dir=str(env), collection="test_mcp_ext_iso"))
    types = [type(b) for b in blocks]
    assert TextContent in types
    assert ImageContent in types
    image_block = next(b for b in blocks if isinstance(b, ImageContent))
    assert image_block.type == "image"
    assert image_block.mime_type == "image/png"


def test_query_knowledge_hub_text_only_has_no_image_blocks(env, monkeypatch):
    """不带 image_refs 的结果 -> 只有 TextContent（现状兼容，US11）。"""
    from rag_mcp_server.application import rag_service
    from mcp.types import ImageContent, TextContent

    monkeypatch.setattr(
        rag_service,
        "query",
        lambda *a, **k: [
            RetrievalResult(
                chunk_id="doc::0", source_path="notes.md", score=0.5, text="内容", sources=["hybrid"]
            )
        ],
    )
    blocks = _run(query_knowledge_hub("检索", index_dir=str(env), collection="test_mcp_ext_iso"))
    assert blocks and all(isinstance(b, TextContent) for b in blocks)
    assert not any(isinstance(b, ImageContent) for b in blocks)


def test_query_knowledge_hub_text_only_does_not_build_image_storage(env, monkeypatch):
    """纯文本结果不应初始化 ImageStorage，避免查询时落盘副作用。"""
    from rag_mcp_server.application import rag_service
    import rag_mcp_server.mcp_server.server as server_mod

    monkeypatch.setattr(
        rag_service,
        "query",
        lambda *a, **k: [
            RetrievalResult(
                chunk_id="doc::0", source_path="notes.md", score=0.5, text="内容", sources=["hybrid"]
            )
        ],
    )
    monkeypatch.setattr(server_mod, "build_image_storage", lambda: pytest.fail("should not build image storage"))

    blocks = _run(query_knowledge_hub("检索", index_dir=str(env), collection="test_mcp_ext_iso"))

    # rich response: Markdown report + JSON references, no image storage built
    assert len(blocks) == 2
    assert "## 检索结果" in blocks[0].text


def test_query_knowledge_hub_multi_result_ranks_and_image_order(env, monkeypatch, tmp_path):
    """多结果编号连续，命中图片时 ImageContent 紧跟对应 TextContent。"""
    from rag_mcp_server.application import rag_service
    from rag_mcp_server.ingestion.storage.image_storage import ImageStorage
    from mcp.types import ImageContent, TextContent
    import rag_mcp_server.mcp_server.server as server_mod

    storage = ImageStorage(
        db_path=tmp_path / "db" / "image_index.db",
        images_root=tmp_path / "images",
    )
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
    storage.save_image("doc-a_p1_1", png, mime_type="image/png", document_id="doc-a")
    monkeypatch.setattr(
        server_mod,
        "build_image_storage",
        lambda: ImageStorage(
            db_path=tmp_path / "db" / "image_index.db",
            images_root=tmp_path / "images",
        ),
    )
    monkeypatch.setattr(
        rag_service,
        "query",
        lambda *a, **k: [
            RetrievalResult(
                chunk_id="doc-a::0",
                source_path="a.pdf",
                score=0.9,
                text="第一段",
                sources=["hybrid"],
                metadata={"image_refs": ["doc-a_p1_1"]},
            ),
            RetrievalResult(
                chunk_id="doc-b::0",
                source_path="b.md",
                score=0.8,
                text="第二段",
                sources=["hybrid"],
            ),
        ],
    )

    blocks = _run(query_knowledge_hub("检索", index_dir=str(env), collection="test_mcp_ext_iso"))

    # rich response: Markdown (with [1] and [2] markers) + JSON refs, then
    # the image block (image_refs on the first result only)
    block_types = [type(b) for b in blocks]
    assert TextContent in block_types
    assert ImageContent in block_types
    # Markdown first block carries both citation markers
    assert blocks[0].text.startswith("## 检索结果")
    assert "### [1]" in blocks[0].text
    assert "### [2]" in blocks[0].text


# -- P10.8 ticket 12 + ADR 0016: list_collections / get_document_summary -----
def _fake_stats():
    return {"documents": 2, "chunks": 3}


def test_list_collections_is_registered_as_a_tool():
    from rag_mcp_server.mcp_server.server import server

    tool_names = {t.name for t in _run(server.list_tools())}
    assert "list_collections" in tool_names
    assert "get_document_summary" in tool_names


def test_list_collections_returns_collection_names(env, monkeypatch):
    """ADR 0016 D23: list_collections 返回真正的集合清单（名 + 文档/chunk 数）。"""
    import rag_mcp_server.mcp_server.server as server_mod

    manager = type("M", (), {"stats": lambda self: _fake_stats()})()
    monkeypatch.setattr(server_mod, "_build_document_manager", lambda collection: manager)
    # env 的 evidence 在 index_dir/test_mcp_ext_iso/chunks.jsonl
    monkeypatch.setattr(
        server_mod,
        "_collection_names",
        lambda index_dir: ["test_mcp_ext_iso"],
    )
    text = _text(_run(list_collections(index_dir=str(env))))
    assert "共 1 个集合" in text
    assert "test_mcp_ext_iso" in text
    assert "documents=2 chunks=3" in text


def test_list_collections_not_ingested_is_friendly(env):
    with tempfile.TemporaryDirectory() as tmp:
        text = _text(_run(list_collections(index_dir=str(Path(tmp) / "empty"))))
        assert "尚未导入" in text


def test_get_document_summary_returns_rich_summary(env, monkeypatch):
    """存在 document_id -> 富摘要（source/chunks/images/title/summary）。"""
    import rag_mcp_server.mcp_server.server as server_mod

    def fake_stats_detailed():
        return [
            {"document_id": "doc-b", "source_path": "b.md", "chunk_count": 1,
             "title": "B 标题", "summary": "B 摘要"},
        ]

    manager = type("M", (), {"stats_detailed": lambda self: fake_stats_detailed()})()
    monkeypatch.setattr(server_mod, "_build_document_manager", lambda collection: manager)
    # get_document_summary uses the collection guard; the env evidence is under
    # test_mcp_ext_iso/ so pass that collection
    text = _text(_run(get_document_summary("doc-b", index_dir=str(env), collection="test_mcp_ext_iso")))
    assert "document_id=doc-b" in text
    assert "source=b.md" in text
    assert "chunks=1" in text
    assert "title=B 标题" in text
    assert "summary=B 摘要" in text


def test_get_document_summary_missing_doc_is_friendly(env, monkeypatch):
    """不存在 document_id -> 友好错误（非异常）。"""
    import rag_mcp_server.mcp_server.server as server_mod

    manager = type("M", (), {"stats_detailed": lambda self: []})()
    monkeypatch.setattr(server_mod, "_build_document_manager", lambda collection: manager)
    text = _text(_run(get_document_summary("does-not-exist", index_dir=str(env), collection="test_mcp_ext_iso")))
    assert "未找到文档 does-not-exist" in text


def test_get_document_summary_not_ingested_is_friendly(env):
    with tempfile.TemporaryDirectory() as tmp:
        text = _text(_run(get_document_summary("doc-a", index_dir=str(Path(tmp) / "empty"))))
        assert "尚未导入" in text
