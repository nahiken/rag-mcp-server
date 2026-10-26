"""Abstract contract for rerankers — the pluggable post-retrieval ranking layer.

BaseReranker.rerank(query, results, top_k) reorders (and re-scores) a
RetrievalResult list. Implementations must NOT change chunk_id/text/source_path
— only order and score (ADR 0009). NoneReranker is the default fallback that
keeps the original order. The orchestration wrapper (QueryReranker) catches
RerankError and falls back to the original order so a ranking backend failure
never blocks a query. See docs/spec-p10-rerank.md and ADR 0009.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag_mcp_server.core.types import RetrievalResult
from rag_mcp_server.core.validation import validate_positive_int

__all__ = ["BaseReranker", "NoneReranker"]


class BaseReranker(ABC):
    """Rerank candidates: query + results -> reordered results."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Return up to ``top_k`` results reordered by relevance to ``query``.

        Must not change chunk_id/text/source_path — only order and score.
        """


class NoneReranker(BaseReranker):
    """Keep the original order (default fallback). Only truncates to top_k."""

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        validate_positive_int("top_k", top_k)
        return list(results[:top_k])
