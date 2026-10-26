"""Hybrid retrieval: fuse dense (vector) + sparse (BM25) rankings with RRF.

Each route contributes its top `top_k * 3` candidates; RRF fuses the rankings
(score = Σ 1/(rrf_k + rank)), ties break by chunk_id ascending, and the fused
list is truncated back to top_k. The fused score goes into the result's `score`;
`sources` lists the single routes that actually produced the chunk.

There is no explicit fallback branch: if sparse has no hits, RRF degrades
naturally to the pure dense ordering; if both routes are empty, the result is
empty. See docs/spec-p9-hybrid.md (task 8) and ADR 0003.
"""

from __future__ import annotations

from rag_mcp_server.core.types import RetrievalResult, image_refs_metadata
from rag_mcp_server.retrieval.dense import DenseRetriever
from rag_mcp_server.retrieval.sparse_bm25 import SparseBM25

DEFAULT_RRF_K = 60
CANDIDATE_MULTIPLIER = 3


def rrf_fuse(
    ranked_ids: list[list[str]],
    top_k: int,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[tuple[str, float]]:
    """Fuse ranked id lists with Reciprocal Rank Fusion.

    Returns up to top_k (chunk_id, score) pairs, score descending; equal scores
    tie-break by chunk_id ascending (stable ordering).
    """
    scores: dict[str, float] = {}
    for ranked in ranked_ids:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
    ordered = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    return ordered[:top_k]


def fuse_results(
    dense_results: list[RetrievalResult],
    sparse_results: list[RetrievalResult],
    top_k: int,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[RetrievalResult]:
    """Fuse dense + sparse result lists with RRF into the domain shape.

    Shared by HybridRetriever.search and the orchestrator (rag_service), which
    times the two retrievals and the fusion separately; this function only does
    the fusion + source attribution, never the retrieval.
    """
    dense_by_id = {r.chunk_id: r for r in dense_results}
    sparse_by_id = {r.chunk_id: r for r in sparse_results}
    if not (set(dense_by_id) | set(sparse_by_id)):
        return []

    fused = rrf_fuse(
        [list(dense_by_id), list(sparse_by_id)],
        top_k=top_k,
        rrf_k=rrf_k,
    )

    results: list[RetrievalResult] = []
    for chunk_id, score in fused:
        source = dense_by_id.get(chunk_id) or sparse_by_id.get(chunk_id)
        sources: list[str] = []
        if chunk_id in dense_by_id:
            sources.append("dense")
        if chunk_id in sparse_by_id:
            sources.append("sparse")
        results.append(
            RetrievalResult(
                chunk_id=chunk_id,
                source_path=source.source_path if source else "",
                score=score,
                text=source.text if source else "",
                sources=sources,
                metadata=image_refs_metadata(source.metadata if source else {}),
            )
        )
    return results


class HybridRetriever:
    """Dense + sparse retrieval fused with RRF.

    ``dense`` and ``sparse`` are injected (spec: HybridRetriever(dense,
    sparse)); the orchestrator builds them — DenseRetriever(store) and
    SparseBM25(store.get_all()) — so the fusion logic never constructs
    retrievers itself.
    """

    def __init__(
        self,
        dense: DenseRetriever,
        sparse: SparseBM25,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        self._dense = dense
        self._sparse = sparse
        self._rrf_k = rrf_k

    def search(self, query_embedding: list[float], query_text: str, top_k: int = 3) -> list[RetrievalResult]:
        """Return up to top_k fused results, most relevant first.

        An empty query embedding short-circuits to [] (matching the dense
        contract): there is no dense contribution to fuse, so a query that was
        not embedded at all must not silently become a sparse-only search.
        """
        if not query_embedding:
            return []
        candidate_k = top_k * CANDIDATE_MULTIPLIER
        dense_results = self._dense.search(query_embedding, top_k=candidate_k)
        sparse_results = self._sparse.search(query_text, top_k=candidate_k)
        return fuse_results(dense_results, sparse_results, top_k=top_k, rrf_k=self._rrf_k)
