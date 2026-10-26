"""Real MCP server built on the official mcp Python SDK (v2).

`query_knowledge_hub` and `semantic_search` reuse rag_service.query so the tool
boundary carries no business logic. query_knowledge_hub is config-driven
(default hybrid; strategy/top_k overridable); semantic_search forces dense
semantic retrieval. Both return a ContentBlock list (P10.6 T3): one TextContent
per result plus, when a result carries ``image_refs``, the matching
ImageContent blocks — so a hit on an image-bearing chunk shows the image to the
client while text-only results stay unchanged (US10/US11).

P10.8 ticket 12 adds `list_collections` and `get_document_summary`: knowledge-base
structure tools that aggregate `DocumentManager.stats_detailed()` (CONTEXT.md
"文档集合" / "文档摘要"). Their store is built from settings via the same
create_vector_store seam as the pipeline/Dashboard; `index_dir` only guards
"have you ingested yet" (same as the query tools — it is not a store selector).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from mcp.server import MCPServer
from mcp.types import ContentBlock, TextContent

from rag_mcp_server.application import rag_service
from rag_mcp_server.core.response.multimodal_assembler import (
    build_image_blocks,
    build_image_storage,
)
from rag_mcp_server.core.response.response_builder import (
    build_citations,
    build_markdown,
    build_references_json,
)
from rag_mcp_server.core.types import RetrievalResult
from rag_mcp_server.document_manager.manager import DocumentManager
from rag_mcp_server.libs.vector_store.factory import create_vector_store

logger = logging.getLogger(__name__)

server = MCPServer("rag_mcp_server")


def _ensure_utf8_stdio() -> None:
    """Force UTF-8 on the server's stdout/stderr.

    MCP stdio transport serializes JSON; on Windows the default ANSI
    code page (GBK) would corrupt non-ASCII content read by the client.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def _is_ingested(index_dir: Path, collection: str | None = None) -> bool:
    """True when this collection's index dir has been ingested at least once.

    The chunks.jsonl evidence snapshot is written by every ingest; it is not
    the query source (the vector store is) but gives users a clear "please run
    ingest first" message instead of a bare empty result. Evidence lives under
    ``index_dir/{collection}/`` (ADR 0016 D19, shared via rag_service.evidence_dir).
    """
    return (rag_service.evidence_dir(index_dir, collection or rag_service.DEFAULT_COLLECTION) / "chunks.jsonl").is_file()


def _ingest_guard(index_dir: Path, collection: str | None = None) -> list[ContentBlock] | None:
    """The "not ingested yet" friendly block, or None when ingested.

    Shared by every tool so the guard+message stays in one place (review FIX-1):
    query tools, list_collections, get_document_summary all return this single
    TextContent when the index has no chunks.jsonl, instead of a bare result.
    """
    if _is_ingested(index_dir, collection):
        return None
    chunks_file = rag_service.evidence_dir(index_dir, collection or rag_service.DEFAULT_COLLECTION) / "chunks.jsonl"
    return [TextContent(type="text", text=f"知识库尚未导入：{chunks_file}。请先运行 ingest。")]


def _result_blocks(
    query: str,
    results: list[RetrievalResult],
) -> list[ContentBlock]:
    """Rich Markdown report + JSON references + image blocks (ADR 0017).

    The Markdown body carries [1] markers and a reference list; the JSON
    references block is machine-readable for AI clients; image blocks
    (P10.6 T3) are assembled from each result's ``image_refs`` via the default
    ImageStorage. A missing image or storage never blocks the text result
    (build_image_blocks never raises).
    """
    citations = build_citations(results)
    blocks: list[ContentBlock] = [
        TextContent(type="text", text=build_markdown(query, results, citations)),
        TextContent(
            type="text",
            text=build_references_json(citations, query, len(results)),
        ),
    ]
    image_storage = (
        build_image_storage()
        if any((result.metadata or {}).get("image_refs") for result in results)
        else None
    )
    for result in results:
        blocks.extend(build_image_blocks(result, image_storage))
    return blocks


def _run_query(
    query: str,
    index_dir: str,
    strategy: str | None,
    top_k: int | None,
    collection: str | None = None,
) -> list[ContentBlock]:
    """Shared tool body: guard -> query -> friendly no-hit / content blocks.

    ``collection`` (ADR 0016) selects the Chroma collection; None uses the
    settings default. The query text may also carry ``collection:`` syntax —
    the explicit argument wins over the text selector (D18). The guard checks
    the *effective* collection (arg or text selector) so a text ``collection:``
    is never rejected against the wrong evidence dir.
    """
    from rag_mcp_server.retrieval.query_processor import parse_collection

    index_path = Path(index_dir)
    effective = collection or parse_collection(query)
    guard = _ingest_guard(index_path, effective)
    if guard is not None:
        return guard
    results = rag_service.query(
        query, index_dir=index_path, strategy=strategy, top_k=top_k,
        collection=effective,
    )
    if not results:
        return [TextContent(type="text", text="没有找到匹配 chunk")]
    return _result_blocks(query, results)


@server.tool()
async def query_knowledge_hub(
    query: str,
    index_dir: str = "data/index",
    strategy: str | None = None,
    top_k: int | None = None,
    collection: str | None = None,
) -> list[ContentBlock]:
    """对已导入的知识库执行配置驱动检索（dense/sparse/hybrid，默认读配置），返回 Markdown 富文本 + JSON 引用；命中带图 chunk 时附上图片。

    ``strategy``/``top_k`` 可覆盖配置（None = 用配置默认）。``collection`` 选择向量库集合（None = 用配置默认集合）；查询文本里也可写 ``collection:xxx``，显式参数优先（ADR 0016）。
    """
    return _run_query(query, index_dir, strategy=strategy, top_k=top_k, collection=collection)


@server.tool()
async def semantic_search(
    query: str,
    index_dir: str = "data/index",
    top_k: int | None = None,
    collection: str | None = None,
) -> list[ContentBlock]:
    """对已导入的知识库执行语义（dense）检索，返回 Markdown 富文本 + JSON 引用；命中带图 chunk 时附上图片。``collection`` 选择集合（显式参数优先）。"""
    return _run_query(query, index_dir, strategy="dense", top_k=top_k, collection=collection)


def _build_document_manager(collection: str | None = None) -> DocumentManager:
    """The DocumentManager for the knowledge-base structure tools.

    Built from settings via the same create_vector_store seam as the pipeline
    and Dashboard (data_service.knowledge_stats) so a separate MCP process reads
    exactly the store the ingest process wrote. ``collection`` (ADR 0016)
    selects the Chroma collection; None uses rag_service.DEFAULT_COLLECTION
    (the same default the query tools / ingest use — NOT the legacy
    settings.vector_store.collection, which predates multi-collection).
    ``image_storage`` follows the query path's build_image_storage seam
    (settings-lazy default) so ``stats_detailed`` can report image counts. A
    store that cannot be built propagates — a structure tool with a broken
    store is a real bug, unlike the Dashboard's degraded zero-cards (which
    exist for rendering).
    """
    from rag_mcp_server.config.settings import load_settings

    settings = load_settings()
    vstore = settings["vector_store"]
    store = create_vector_store(
        provider=vstore["provider"],
        persist_dir=vstore.get("persist_dir"),
        collection=collection or rag_service.DEFAULT_COLLECTION,
    )
    return DocumentManager(
        store=store,
        image_storage=build_image_storage(),
    )


def _collection_names(index_dir: Path) -> list[str]:
    """The ingested collection names, derived from the evidence directories.

    Evidence lives under ``index_dir/{collection}/chunks.jsonl`` (ADR 0016
    D19), so the set of collections with a chunk snapshot is exactly the set of
    ingested collections — no need to enumerate Chroma collections directly.
    Shared via rag_service.collection_names (C7: agentic list_directory and
    this tool use one source).
    """
    return rag_service.collection_names(index_dir)


@server.tool()
async def list_collections(index_dir: str = "data/index") -> list[ContentBlock]:
    """列出已摄入的集合（每个集合一行：文档数 / chunk 数），返回带来源结构。"""
    index_path = Path(index_dir)
    names = _collection_names(index_path)
    if not names:
        return [TextContent(type="text", text="知识库尚未导入。请先运行 ingest。")]
    lines = [f"共 {len(names)} 个集合"]
    for i, name in enumerate(names, start=1):
        try:
            stats = _build_document_manager(name).stats()
        except Exception as exc:  # noqa: BLE001 — one broken collection must not abort the list
            logger.warning("list_collections: 集合 %s stats 失败：%s", name, exc)
            lines.append(f"[{i}] {name} stats 不可用")
            continue
        lines.append(
            f"[{i}] {name} documents={stats.get('documents', 0)} chunks={stats.get('chunks', 0)}"
        )
    return [TextContent(type="text", text="\n".join(lines))]


@server.tool()
async def get_document_summary(
    document_id: str,
    index_dir: str = "data/index",
    collection: str | None = None,
) -> list[ContentBlock]:
    """返回文档摘要：source_path / chunk 数 / 关联图片数（metadata 有 title/summary 则带上）。``collection`` 选择集合（默认配置集合）。"""
    guard = _ingest_guard(Path(index_dir), collection)
    if guard is not None:
        return guard
    collections = _build_document_manager(collection).stats_detailed()
    match = next((c for c in collections if c["document_id"] == document_id), None)
    if match is None:
        return [TextContent(type="text", text=f"未找到文档 {document_id}")]
    parts = [
        f"document_id={match['document_id']}",
        f"source={match.get('source_path', '')}",
        f"chunks={match.get('chunk_count', 0)}",
    ]
    if "image_count" in match:
        parts.append(f"images={match['image_count']}")
    for key in ("title", "summary"):
        if key in match:
            parts.append(f"{key}={match[key]}")
    return [TextContent(type="text", text="\n".join(parts))]


# --- Agentic RAG tools (ADR 0019) ------------------------------------------
# Thin wrappers over the MCP-free pure functions in mcp_server/tools/agentic.py;
# server.py is the single registration point (Q8). Each returns a plain
# TextContent Markdown block (Q1).
from rag_mcp_server.mcp_server.tools.agentic import (  # noqa: E402
    list_directory as _agentic_list_directory,
    list_documents as _agentic_list_documents,
    preview_document as _agentic_preview_document,
    self_check as _agentic_self_check,
    verify_fact as _agentic_verify_fact,
)


@server.tool()
async def list_directory(index_dir: str = "data/index") -> list[ContentBlock]:
    """列出已摄入的集合（Agent 第一跳），返回 Markdown 文本。"""
    return [TextContent(type="text", text=_agentic_list_directory(index_dir))]


@server.tool()
async def list_documents(
    index_dir: str = "data/index", collection: str | None = None
) -> list[ContentBlock]:
    """列出某集合下的文档 + chunk 数，返回 Markdown 文本。``collection`` 选集合（默认配置）。"""
    return [TextContent(type="text", text=_agentic_list_documents(index_dir, collection))]


@server.tool()
async def preview_document(
    document_id: str,
    index_dir: str = "data/index",
    collection: str | None = None,
    max_chunks: int = 5,
) -> list[ContentBlock]:
    """预览某篇文档前 max_chunks 个 chunk（每 chunk 截断 300 字符），返回 Markdown 文本。"""
    return [TextContent(type="text", text=_agentic_preview_document(document_id, index_dir, collection, max_chunks))]


@server.tool()
async def verify_fact(
    claim: str,
    index_dir: str = "data/index",
    collection: str | None = None,
    top_k: int = 3,
) -> list[ContentBlock]:
    """检索证据并判读断言是否被支持（支持/不支持 + 理由 + 证据），返回 Markdown 文本。"""
    return [TextContent(type="text", text=_agentic_verify_fact(claim, index_dir, collection, top_k))]


@server.tool()
async def self_check(
    query: str,
    index_dir: str = "data/index",
    collection: str | None = None,
    top_k: int = 3,
) -> list[ContentBlock]:
    """自检检索证据是否足以回答查询（充分/不足 + gap + 证据），返回 Markdown 文本。"""
    return [TextContent(type="text", text=_agentic_self_check(query, index_dir, collection, top_k))]


if __name__ == "__main__":
    _ensure_utf8_stdio()
    server.run(transport="stdio")
