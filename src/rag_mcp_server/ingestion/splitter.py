"""Optional chunking variants: fixed-length windows and semantic grouping.

Two pure functions that mirror ``chunk_document``'s output contract (Chunk with
document_id/text/source_path/order/metadata; metadata carrying ``file_type`` +
optional ``image_refs``), so the pipeline can swap the chunker behind the same
seam (ADR 0020, T5).

- ``chunk_document_fixed``: uniform ``chunk_size`` sliding windows with
  ``overlap``, covering the whole text without losing characters — for dense
  text without natural paragraph boundaries.
- ``chunk_document_semantic``: paragraphs are embedded and adjacent paragraphs
  merge while their cosine similarity stays >= ``threshold``; a group is capped
  at ``max_group`` paragraphs; an over-long paragraph (or group) is continued by
  ``max_chars``. Embedding failure/unavailability degrades to paragraph chunking
  (reusing ``chunk_document``), so semantic chunking never breaks an ingest when
  Ollama is offline.

First-version heuristics: cosine similarity between paragraph vectors, no
tuning, no paragraph-merge sophistication (ADR 0020 D7/D20: 不追求精确).
"""

from __future__ import annotations

import logging
import math

from rag_mcp_server.core.types import Chunk, Document
from rag_mcp_server.ingestion.chunker import (
    DEFAULT_MAX_CHARS,
    _image_refs_of,
    chunk_document,
)

logger = logging.getLogger(__name__)


def _metadata_for(document: Document, text: str) -> dict:
    """Chunk metadata mirroring chunk_document (file_type + optional image_refs)."""
    metadata = {"file_type": document.file_type}
    refs = _image_refs_of(text)
    if refs:
        metadata["image_refs"] = refs
    return metadata


def chunk_document_fixed(
    document: Document,
    chunk_size: int = DEFAULT_MAX_CHARS,
    overlap: int = 100,
) -> list[Chunk]:
    """Uniform fixed-length windows over the full text, with ``overlap``.

    The window slides by ``chunk_size - overlap`` so consecutive windows share
    ``overlap`` characters; the first window starts at index 0 and the last
    reaches the end, so the whole text is covered with no character dropped.
    ``overlap >= chunk_size`` is clamped to a step of 1 (degenerate but never
    infinite). Empty text -> [].
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size 必须为正整数，得到 {chunk_size}")
    text = document.text
    if not text:
        return []
    step = max(1, chunk_size - overlap)
    chunks: list[Chunk] = []
    order = 0
    n = len(text)
    pos = 0
    while pos < n:
        piece = text[pos : pos + chunk_size]
        chunks.append(
            Chunk(
                document_id=document.id,
                text=piece,
                source_path=document.source_path,
                order=order,
                metadata=_metadata_for(document, piece),
            )
        )
        order += 1
        pos += step
    return chunks


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors (0.0 on degenerate)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def chunk_document_semantic(
    document: Document,
    embedding,
    max_chars: int = DEFAULT_MAX_CHARS,
    threshold: float = 0.6,
    max_group: int = 3,
) -> list[Chunk]:
    """Group paragraphs by embedding similarity, then emit chunks.

    Paragraphs use the same units as ``chunk_document`` (split on blank lines,
    stripped, empties dropped). Each paragraph is embedded; consecutive
    paragraphs whose cosine similarity is >= ``threshold`` merge into one group,
    and a group is capped at ``max_group`` paragraphs. Each group's joined text
    is emitted as one chunk, or continued by ``max_chars`` when over-long.

    Soft degradation (ADR 0020 D10): when the embedding raises or returns a
    wrong-shaped result, the whole document falls back to ``chunk_document``
    (paragraph chunking), so semantic chunking is an enhancement, never a break.
    Empty text -> [].
    """
    if max_chars < 1:
        raise ValueError(f"max_chars 必须为正整数，得到 {max_chars}")
    paragraphs = [p.strip() for p in document.text.split("\n\n")]
    paragraphs = [p for p in paragraphs if p]
    if not paragraphs:
        return []
    try:
        vectors = embedding.embed_texts(paragraphs)
    except Exception as exc:  # noqa: BLE001 — degrade, never break an ingest
        logger.warning("语义分块 embedding 失败，降级段落分块：%s", exc)
        return chunk_document(document, max_chars=max_chars)
    if not vectors or len(vectors) != len(paragraphs):
        logger.warning("语义分块 embedding 结果不可用（shape 不符），降级段落分块")
        return chunk_document(document, max_chars=max_chars)

    # group consecutive paragraphs by adjacent similarity + max_group cap
    groups: list[list[str]] = []
    current = [paragraphs[0]]
    prev_vec = vectors[0]
    for i in range(1, len(paragraphs)):
        sim = _cosine_similarity(prev_vec, vectors[i])
        if sim < threshold or len(current) >= max_group:
            groups.append(current)
            current = []
        current.append(paragraphs[i])
        prev_vec = vectors[i]
    if current:
        groups.append(current)

    chunks: list[Chunk] = []
    order = 0
    for group in groups:
        combined = "\n\n".join(group)
        if len(combined) > max_chars:
            pieces = [
                combined[i : i + max_chars] for i in range(0, len(combined), max_chars)
            ]
        else:
            pieces = [combined]
        for piece in pieces:
            chunks.append(
                Chunk(
                    document_id=document.id,
                    text=piece,
                    source_path=document.source_path,
                    order=order,
                    metadata=_metadata_for(document, piece),
                )
            )
            order += 1
    return chunks
