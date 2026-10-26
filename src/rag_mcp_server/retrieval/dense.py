"""Dense retrieval: a thin adapter over a BaseVectorStore.

The vector store already ranks by cosine similarity (score = 1/(1+distance),
descending). This adapter maps its VectorRecords to the domain RetrievalResult
shape the rest of the app consumes. The query text is embedded by the
orchestrator, not here (docs/spec-p9-hybrid.md, task 8).
"""

from __future__ import annotations

from typing import Any

from rag_mcp_server.core.types import RetrievalResult, image_refs_metadata
from rag_mcp_server.libs.vector_store.base import BaseVectorStore, VectorRecord


class DenseRetriever:
    """Maps vector-store nearest-neighbour results to RetrievalResults."""

    def __init__(self, store: BaseVectorStore) -> None:
        self.store = store

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 3,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Return the top_k nearest neighbours as RetrievalResults.

        ``where`` is forwarded to the store's query filter (ADR 0016 — an
        in-retrieval filter over record metadata, e.g. source_path/tags). The
        filter must happen at retrieval time, not post-hoc, because
        RetrievalResult.metadata only carries image_refs (P10.6 T3).
        """
        if not query_embedding:
            return []
        records = self.store.query(query_embedding, top_k, where=where)
        return [self._to_result(record) for record in records]

    @staticmethod
    def _to_result(record: VectorRecord) -> RetrievalResult:
        return RetrievalResult(
            chunk_id=record.chunk_id,
            source_path=str(record.metadata.get("source_path", "")),
            score=record.score if record.score is not None else 0.0,
            text=record.text,
            sources=["dense"],
            metadata=image_refs_metadata(record.metadata),
        )
