"""QueryReranker — orchestration wrapper: config + fallback around a backend.

Wraps a BaseReranker backend and guarantees a query never breaks on a ranking
failure: any RerankError is caught and the original order is returned (ADR 0009
"fail -> keep original order"). The fallback flag is exposed via ``last_fallback``
so the caller (rag_service.query) writes ``rerank.fallback=true`` into the trace
at its wiring point — it never touches the RetrievalResult contract.

Returns a plain list[RetrievalResult]; callers don't destructure a tuple.
See docs/spec-p10-rerank.md and ADR 0009.
"""

from __future__ import annotations

from rag_mcp_server.core.errors import RerankError
from rag_mcp_server.core.types import RetrievalResult
from rag_mcp_server.core.validation import validate_positive_int
from rag_mcp_server.libs.reranker.base import BaseReranker, NoneReranker


class QueryReranker:
    """Wrap a rerank backend; fall back to the original order on failure."""

    def __init__(self, backend: BaseReranker | None = None) -> None:
        # None backend -> NoneReranker (keep order; no ranking happens)
        self._backend = backend if backend is not None else NoneReranker()
        self.last_fallback: bool = False
        self.last_reason: str | None = None

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Return up to top_k results, reordered by the backend if it succeeds.

        On backend failure the original order (first top_k) is returned and
        ``last_fallback``/``last_reason`` are set so the caller can write the
        fallback marker into the trace.
        """
        validate_positive_int("top_k", top_k)
        self.last_fallback = False
        self.last_reason = None
        try:
            return self._backend.rerank(query, results, top_k)
        except RerankError as exc:
            self.last_fallback = True
            self.last_reason = str(exc)
            return list(results[:top_k])
