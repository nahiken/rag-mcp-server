"""Abstract contract for vector stores.

The single test seam for vector-store behaviour: everything above (ingestion
pipeline, retrieval, document lifecycle) talks to this interface, never to a
concrete backend. See docs/spec-p9-vstore.md and ADR 0004.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

# The four metadata keys every record must carry (see CONTEXT.md
# "Chroma Record Metadata").
REQUIRED_METADATA_KEYS = ("document_id", "source_path", "chunk_id", "content_hash")


@dataclass
class VectorRecord:
    """One record in the vector store: identity, metadata, text, and score.

    `chunk_id` is the record id (readable `<document_id>::<order>`, ADR 0002).
    `score` is the query similarity; it is None outside a query context (e.g.
    get_by_ids). The vector itself is an internal carrier and is never exposed
    as a field here (ADR 0004).
    """

    chunk_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    text: str = ""
    score: float | None = None


class BaseVectorStore(ABC):
    """Stores dense vectors plus their text and metadata.

    Input contract: all methods take lists; empty input short-circuits to an
    empty result, never an error. `query` returns records sorted by similarity
    descending (most relevant first). `delete` is idempotent.
    """

    @abstractmethod
    def add(self, records: list[VectorRecord], embeddings: list[list[float]]) -> None:
        """Insert records and vectors, row-aligned by position."""

    @abstractmethod
    def query(
        self,
        query_embedding: list[float],
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> list[VectorRecord]:
        """Return top_k nearest neighbours, most similar first."""

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """Delete records by id; missing ids are silent, empty is a no-op."""

    @abstractmethod
    def get_by_ids(self, ids: list[str]) -> list[VectorRecord]:
        """Return present records in request order; missing ids are skipped."""

    @abstractmethod
    def get_all(self, where: dict[str, Any] | None = None) -> list[VectorRecord]:
        """Return every record in the store (optionally metadata-filtered).

        Order is not guaranteed. Empty store returns []. ``where`` narrows to
        records whose metadata matches (ADR 0016 collection-scoped corpus for
        the sparse route). Used to rebuild the sparse (BM25) corpus per query
        (ADR 0003, rebuilt-from-vstore).
        """
