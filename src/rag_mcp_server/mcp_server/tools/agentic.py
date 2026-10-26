"""Agentic RAG tools — fine-grained primitives for multi-step Agent retrieval.

Each tool is a thin seam over existing services (DocumentManager / rag_service /
BaseLLM) — no new retrieval logic (ADR 0019). Together they let an Agent
decompose a query: ``list_directory`` (which collections) -> ``list_documents``
(which docs in a collection) -> ``preview_document`` (read a doc) ->
``verify_fact`` / ``self_check`` (judge evidence). Everything returns plain
Markdown text blocks (the Agent reads intermediate info, not a final answer),
and ``verify_fact`` / ``self_check`` degrade to "无法判读 + 证据" on LLM
failure instead of raising (D14, soft-degradation).

The module has NO MCP dependency: functions are plain, take injected seams
(llm / index dir / collection), and are unit-testable offline. server.py is the
single registration point (it wraps these with @server.tool).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rag_mcp_server.application import rag_service
from rag_mcp_server.core.types import Chunk
from rag_mcp_server.libs.llm.base import BaseLLM

# Preview truncation (aligned with response_builder.SNIPPET_MAX).
DEFAULT_PREVIEW_CHUNKS = 5
DEFAULT_SNIPPET_MAX = 300

# --- pure helpers -----------------------------------------------------------

def _snippet(text: str, max_chars: int = DEFAULT_SNIPPET_MAX) -> str:
    """Flatten and truncate a chunk text for display."""
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rsplit(" ", 1)[0] + "..."


def _resolve_collection(collection: str | None) -> str:
    """None -> the settings default collection (ADR 0016 D16)."""
    if collection:
        return collection
    from rag_mcp_server.config.settings import load_settings

    configured = load_settings().get("vector_store", {}).get("collection")
    return configured or rag_service.DEFAULT_COLLECTION


def _load_llm(llm: BaseLLM | None) -> BaseLLM:
    """Injected llm wins; None assembles from settings (ADR 0019 D2)."""
    if llm is not None:
        return llm
    from rag_mcp_server.application.rag_service import _llm_from_settings
    from rag_mcp_server.config.settings import load_settings

    return _llm_from_settings(load_settings())


# --- the five tools (plain functions, testable) -----------------------------

def list_directory(index_dir: str) -> str:
    """List the ingested collections (the Agent's first hop)."""
    index_path = Path(index_dir)
    names = rag_service.collection_names(index_path)
    if not names:
        return "知识库尚未导入。请先运行 ingest。"
    lines = [f"共 {len(names)} 个集合"]
    for i, name in enumerate(names, start=1):
        lines.append(f"[{i}] {name}")
    return "\n".join(lines)


def list_documents(index_dir: str, collection: str | None = None) -> str:
    """List the documents (with chunk counts) in a collection.

    Aggregates from the collection's chunks.jsonl evidence (ADR 0016 D19) via
    rag_service.load_all_chunks — the vector store is not touched, so the tool
    works on evidence alone and stays testable with a temp index.
    """
    index_path = Path(index_dir)
    col = _resolve_collection(collection)
    if not (rag_service.evidence_dir(index_path, col) / "chunks.jsonl").is_file():
        return f"集合 {col} 尚未导入。请先运行 ingest。"
    chunks = rag_service.load_all_chunks(index_path, col)
    if not chunks:
        return f"集合 {col} 暂无文档。"
    # aggregate chunk count per document_id, preserving first-seen order
    by_doc: dict[str, int] = {}
    for c in chunks:
        by_doc[c.document_id] = by_doc.get(c.document_id, 0) + 1
    lines = [f"集合 {col}：共 {len(by_doc)} 篇文档"]
    for i, (doc_id, count) in enumerate(by_doc.items(), start=1):
        lines.append(f"[{i}] {doc_id} chunks={count}")
    return "\n".join(lines)


def preview_document(
    document_id: str,
    index_dir: str,
    collection: str | None = None,
    max_chunks: int = DEFAULT_PREVIEW_CHUNKS,
) -> str:
    """Preview the first ``max_chunks`` chunks of a document (truncated)."""
    index_path = Path(index_dir)
    col = _resolve_collection(collection)
    if not (rag_service.evidence_dir(index_path, col) / "chunks.jsonl").is_file():
        return f"集合 {col} 尚未导入。请先运行 ingest。"
    chunks: list[Chunk] = [
        c for c in rag_service.load_all_chunks(index_path, col) if c.document_id == document_id
    ]
    if not chunks:
        return f"未找到文档 {document_id}"
    parts = []
    for c in chunks[:max_chunks]:
        parts.append(f"### {c.chunk_id}\n{_snippet(c.text)}")
    if len(chunks) > max_chunks:
        parts.append(f"\n...共 {len(chunks)} 个 chunk，显示前 {max_chunks} 个")
    return "\n".join(parts)


_VERIFY_PROMPT = """你是事实核查助手。给定一个断言和检索到的证据片段，判断该断言是否被证据支持。

输出严格 JSON（不要多余文字）：{{"supported": true 或 false, "reason": "一句话理由"}}

断言：{claim}
证据：
{evidence}
"""


def verify_fact(
    claim: str,
    index_dir: str,
    collection: str | None = None,
    top_k: int = 3,
    llm: BaseLLM | None = None,
    pipeline=None,
) -> str:
    """Retrieve evidence for a claim, then judge supported / not supported.

    ``pipeline`` is the injection seam for tests (a temp FakeEmbedding + Chroma
    pipeline); None assembles from settings. On retrieval miss or LLM/parse
    failure, degrades to "无法判读 + 原始证据" (D14 — the tool never raises).
    """
    index_path = Path(index_dir)
    col = _resolve_collection(collection)
    if not (rag_service.evidence_dir(index_path, col) / "chunks.jsonl").is_file():
        return f"集合 {col} 尚未导入。请先运行 ingest。"
    try:
        results = rag_service.query(
            claim, index_dir=index_path, top_k=top_k, collection=col, pipeline=pipeline,
        )
    except Exception:
        return f"检索失败，无法判读断言：{claim}"
    if not results:
        return f"没有检索到与断言相关的证据：{claim}"
    evidence = "\n".join(
        f"[{i+1}] {r.chunk_id} (score={r.score:.2f})\n{_snippet(r.text)}"
        for i, r in enumerate(results)
    )
    prompt = _VERIFY_PROMPT.format(claim=claim, evidence=evidence)
    try:
        backend = _load_llm(llm)
        raw = backend.chat([{"role": "user", "content": prompt}])
        verdict = _parse_verdict(raw)
    except Exception:
        # LLM failure / malformed JSON -> degrade, keep the evidence visible
        return f"**断言：**{claim}\n\n**判定：**无法判读（证据如下）\n\n**证据：**\n{evidence}"
    return (
        f"**断言：**{claim}\n\n"
        f"**判定：**{'支持' if verdict['supported'] else '不支持'} — {verdict['reason']}\n\n"
        f"**证据：**\n{evidence}"
    )


_SELF_CHECK_PROMPT = """你是检索质量评估助手。给定一个查询和检索到的证据片段，判断这些证据是否足以回答查询。

输出严格 JSON（不要多余文字）：{{"sufficient": true 或 false, "gap": "若不足，一句话说明缺什么信息；若足够则填空字符串"}}

查询：{query}
证据：
{evidence}
"""


def self_check(
    query: str,
    index_dir: str,
    collection: str | None = None,
    top_k: int = 3,
    llm: BaseLLM | None = None,
    pipeline=None,
) -> str:
    """Judge whether retrieved evidence is sufficient to answer the query."""
    index_path = Path(index_dir)
    col = _resolve_collection(collection)
    if not (rag_service.evidence_dir(index_path, col) / "chunks.jsonl").is_file():
        return f"集合 {col} 尚未导入。请先运行 ingest。"
    try:
        results = rag_service.query(
            query, index_dir=index_path, top_k=top_k, collection=col, pipeline=pipeline,
        )
    except Exception:
        return f"检索失败，无法自检查询：{query}"
    if not results:
        return f"没有检索到证据，不足以回答：{query}"
    evidence = "\n".join(
        f"[{i+1}] {r.chunk_id} (score={r.score:.2f})\n{_snippet(r.text)}"
        for i, r in enumerate(results)
    )
    prompt = _SELF_CHECK_PROMPT.format(query=query, evidence=evidence)
    try:
        backend = _load_llm(llm)
        raw = backend.chat([{"role": "user", "content": prompt}])
        verdict = _parse_sufficient(raw)
    except Exception:
        return f"**查询：**{query}\n\n**判定：**无法判读（证据如下）\n\n**证据：**\n{evidence}"
    gap = verdict.get("gap", "")
    verdict_line = f"证据{'充分' if verdict['sufficient'] else '不足'}"
    if gap:
        verdict_line += f" — {gap}"
    return f"**查询：**{query}\n\n**判定：**{verdict_line}\n\n**证据：**\n{evidence}"


def _strip_code_fence(text: str) -> str:
    """Tolerate a fenced ```json ... ``` wrapper around the LLM's JSON."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _parse_verdict(raw: str) -> dict[str, Any]:
    """Strictly parse {"supported": bool, "reason": str}; raise on any violation."""
    data = json.loads(_strip_code_fence(raw))
    if not isinstance(data, dict):
        raise ValueError("verdict 顶层必须是对象")
    supported = data.get("supported")
    if not isinstance(supported, bool):
        raise ValueError("verdict.supported 必须是 bool")
    reason = data.get("reason")
    if not isinstance(reason, str):
        raise ValueError("verdict.reason 必须是 str")
    return {"supported": supported, "reason": reason}


def _parse_sufficient(raw: str) -> dict[str, Any]:
    """Strictly parse {"sufficient": bool, "gap": str}; raise on any violation."""
    data = json.loads(_strip_code_fence(raw))
    if not isinstance(data, dict):
        raise ValueError("sufficient 顶层必须是对象")
    sufficient = data.get("sufficient")
    if not isinstance(sufficient, bool):
        raise ValueError("sufficient 必须是 bool")
    gap = data.get("gap", "")
    if not isinstance(gap, str):
        raise ValueError("gap 必须是 str")
    return {"sufficient": sufficient, "gap": gap}
