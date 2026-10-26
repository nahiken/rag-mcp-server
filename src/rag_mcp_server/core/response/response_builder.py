"""Build a Markdown answer with inline citations from retrieval results.

MCP query tools return this instead of bare text (ADR 0017): a human-readable
report with [1][2] markers plus a machine-readable JSON references block so an
AI client can cite/verify. CLI keeps its plain output — this is a presentation
upgrade for the MCP boundary only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from rag_mcp_server.core.types import RetrievalResult

SNIPPET_MAX = 300  # chars per result snippet in the Markdown body


@dataclass
class Citation:
    """One structured reference to a retrieved chunk."""

    index: int
    chunk_id: str
    source: str
    score: float
    snippet: str
    page: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "index": self.index,
            "chunk_id": self.chunk_id,
            "source": self.source,
            "score": round(self.score, 4),
            "snippet": self.snippet,
        }
        if self.page is not None:
            d["page"] = self.page
        return d


def build_citations(results: list[RetrievalResult]) -> list[Citation]:
    """1-based citations; source = source_path, page from metadata if present."""
    out: list[Citation] = []
    for i, r in enumerate(results, start=1):
        meta = r.metadata or {}
        page = meta.get("page")
        if page is not None:
            try:
                page = int(page)
            except (TypeError, ValueError):
                page = None
        out.append(
            Citation(
                index=i,
                chunk_id=r.chunk_id,
                source=r.source_path,
                score=r.score,
                snippet=" ".join(r.text.split())[:SNIPPET_MAX],
                page=page,
            )
        )
    return out


def build_markdown(
    query: str,
    results: list[RetrievalResult],
    citations: list[Citation],
) -> str:
    """Human-readable Markdown report with [n] markers and a reference list."""
    lines: list[str] = [
        "## 检索结果",
        f"针对查询 **{query}** 找到 {len(results)} 条相关结果：",
        "",
    ]
    for c, r in zip(citations, results):
        lines.append(f"### [{c.index}] {c.source}")
        relevance = f"**相关度:** {c.score:.2%}"
        if c.page is not None:
            relevance += f"  **页码:** {c.page}"
        lines.append(relevance)
        lines.append(f"\n> {c.snippet}\n")
    lines += ["---", "## 引用来源"]
    for c in citations:
        source_info = f"- [{c.index}] `{c.source}`"
        if c.page is not None:
            source_info += f" (p.{c.page})"
        lines.append(source_info)
    return "\n".join(lines)


def build_references_json(
    citations: list[Citation],
    query: str,
    result_count: int,
) -> str:
    """The machine-readable references block, embedded as a Markdown fenced JSON."""
    payload = {
        "query": query,
        "result_count": result_count,
        "citations": [c.to_dict() for c in citations],
    }
    body = json.dumps(payload, ensure_ascii=False, indent=2)
    return f"\n---\n**References (JSON):**\n```json\n{body}\n```"
