"""LLMReranker — rank candidates by an LLM's per-candidate 0-3 relevance scores.

The injected BaseLLM is asked to score each candidate as a JSON object array
[{passage_id, score (0-3), reasoning}] (strict JSON prompt contract). We parse
strictly (top-level list, dict items, passage_id + score present, numeric
score) and reorder by score descending. Unknown ids are silently skipped (a
missing candidate just loses its score); an empty result is success. On any
violation (bad JSON, bad shape, LLM failure) we raise RerankError so the
orchestration wrapper can fall back to the original order — never a bare
KeyError/TypeError/LLMError leaking to consumers. The reranked RetrievalResult
gets score = the LLM's 0-3 score; metadata stays clean (ADR 0009 pt.3).

See docs/spec-p10-rerank.md and ADR 0009.
"""

from __future__ import annotations

import json

from rag_mcp_server.core.errors import LLMError, RerankError
from rag_mcp_server.core.types import RetrievalResult, image_refs_metadata
from rag_mcp_server.core.validation import validate_positive_int
from rag_mcp_server.libs.llm.base import BaseLLM
from rag_mcp_server.libs.reranker.base import BaseReranker

DEFAULT_RERANK_PROMPT = """You are a relevance scorer for a RAG retriever.

Given a query and candidate passages, score each passage 0-3 for how well it
answers the query:
- 3 (Highly Relevant): directly answers or addresses the query
- 2 (Partially Relevant): related information but doesn't fully address it
- 1 (Marginally Relevant): some tangential connection
- 0 (Not Relevant): no meaningful connection

Consider semantic similarity, whether the passage contains the answer, and
specificity/completeness.

Output a JSON array of objects, one per passage, each with:
- passage_id: the passage's id (use it verbatim)
- score: an integer or float 0-3
- reasoning: one short sentence"""


class LLMReranker(BaseReranker):
    """Rerank by asking an injected LLM to score each candidate 0-3."""

    def __init__(
        self,
        llm: BaseLLM,
        prompt_template: str = DEFAULT_RERANK_PROMPT,
    ) -> None:
        self._llm = llm
        self._prompt_template = prompt_template

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        validate_positive_int("top_k", top_k)
        if not results:
            return []

        prompt = self._build_prompt(query, results)
        try:
            raw = self._llm.chat([{"role": "user", "content": prompt}])
        except LLMError as exc:
            raise RerankError(f"LLM 重排调用失败：{exc}") from exc

        scored = self._parse(raw)
        return self._apply(results, scored, top_k)

    # -- prompt construction ------------------------------------------------
    def _build_prompt(self, query: str, results: list[RetrievalResult]) -> str:
        passages = "\n".join(
            f"Passage ID: {r.chunk_id}\nText: {r.text}" for r in results
        )
        return (
            f"{self._prompt_template}\n\nQuery: {query}\n\nPassages:\n{passages}\n\n"
            "Output your response as a JSON array of objects, one per passage."
        )

    # -- strict parse -> RerankError (never a bare exception) ---------------
    def _parse(self, raw: str) -> list[dict]:
        text = raw.strip()
        if text.startswith("```"):
            # tolerate fenced code blocks; strip the ```json / ``` fences
            lines = text.splitlines()
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, ValueError) as exc:
            raise RerankError(f"LLM 重排输出不是有效 JSON：{exc}") from exc

        if not isinstance(parsed, list):
            raise RerankError(f"LLM 重排输出顶层必须是数组（实际 {type(parsed).__name__}）")
        result: list[dict] = []
        for i, item in enumerate(parsed):
            if not isinstance(item, dict):
                raise RerankError(f"LLM 重排输出第 {i} 个元素必须是对象（实际 {type(item).__name__}）")
            if "passage_id" not in item:
                raise RerankError(f"LLM 重排输出第 {i} 个元素缺 passage_id")
            if "score" not in item:
                raise RerankError(f"LLM 重排输出第 {i} 个元素缺 score")
            if not isinstance(item["score"], (int, float)) or isinstance(item["score"], bool):
                raise RerankError(f"LLM 重排输出第 {i} 个元素的 score 必须是数值（实际 {type(item['score']).__name__}）")
            result.append(item)
        return result

    # -- apply scores, keep only known ids, sort desc, truncate top_k -------
    def _apply(
        self,
        results: list[RetrievalResult],
        scored: list[dict],
        top_k: int,
    ) -> list[RetrievalResult]:
        by_id = {r.chunk_id: r for r in results}
        out: list[RetrievalResult] = []
        for item in scored:
            passage_id = item["passage_id"]
            if passage_id not in by_id:
                continue  # unknown id silently skipped (容错)
            original = by_id[passage_id]
            out.append(
                RetrievalResult(
                    chunk_id=original.chunk_id,
                    source_path=original.source_path,
                    score=float(item["score"]),
                    text=original.text,
                    sources=list(original.sources),
                    metadata=image_refs_metadata(original.metadata),
                )
            )
        out.sort(key=lambda r: r.score, reverse=True)
        return out[:top_k]
