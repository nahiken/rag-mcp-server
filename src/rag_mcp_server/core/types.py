"""Core domain types: Document, Chunk, RetrievalResult, and id helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

# The default collection every ingest/query lands on when none is specified
# (ADR 0016 D22). Single source for the "default" literal across core/types,
# ingestion/loader, ingestion/pipeline and application/rag_service so the
# multi-collection default is never spelled in more than one place.
DEFAULT_COLLECTION = "default"


@dataclass
class Document:
    """A document imported into the store."""

    id: str
    text: str
    source_path: str
    file_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A retrieval unit produced by chunking a document."""

    document_id: str
    text: str
    source_path: str
    order: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        return f"{self.document_id}::{self.order}"


@dataclass
class RetrievalResult:
    """One chunk matched by a query.

    ``metadata`` carries a minimal slice of the source chunk/record metadata
    (P10.6 T3): only ``image_refs`` is copied at the retrieval seam so MCP can
    assemble ``ImageContent`` blocks. Other record keys never leak here; a
    record without image_refs leaves it an empty dict (behaviour unchanged).
    """

    chunk_id: str
    source_path: str
    score: float
    text: str
    sources: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def image_refs_metadata(record_metadata: dict[str, Any]) -> dict[str, Any]:
    """The minimal metadata slice a RetrievalResult may carry (P10.6 T3).

    Only ``image_refs`` is copied off the source record/chunk; every other key
    (document_id, content_hash, transform markers...) stays behind so the query
    result stays minimal and records without image_refs yield an empty dict
    (behaviour unchanged). Single source of truth shared by the dense / sparse /
    hybrid retrieval seams and the reranker.
    """
    refs = record_metadata.get("image_refs")
    if not refs:
        return {}
    return {"image_refs": list(refs)}


def make_document_id(path: PurePosixPath, collection: str = DEFAULT_COLLECTION) -> str:
    """Build a readable slug document id from a relative path and collection.

    The id must be readable for debugging and must not collide for the same
    filename living in different directories *or* different collections
    (ADR 0016 D17 — multi-collection). UUIDs and content hashes are
    intentionally avoided in the first version.

    Slug rule: non-word characters (including path separators, spaces, dots
    and underscores) collapse to a single ``-``; Unicode letters (CJK etc.)
    are preserved so Chinese filenames keep distinct ids. ``[^a-zA-Z0-9]+``
    would strip all CJK and collapse ``报告.md`` / ``总结.md`` into the same
    id ``md``, silently overwriting one document (regression fixed).

    ``collection`` prefixes the id as ``{collection}::{slug}`` so the same
    file in two collections gets distinct ids and cross-collection
    list/delete stays unambiguous (ADR 0016).
    """
    as_posix = PurePosixPath(path).as_posix()
    slug = re.sub(r"[\W_]+", "-", as_posix).strip("-").lower()
    if not collection:
        return slug
    return f"{collection}::{slug}"
