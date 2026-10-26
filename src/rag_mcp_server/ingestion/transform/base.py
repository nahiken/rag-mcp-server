"""Abstract contract for ingestion transforms — pluggable post-chunking steps.

BaseTransform.transform(chunks, trace=None) returns a new list of Chunks.
Implementations must degrade gracefully: a failure on one chunk never blocks the
others (the original chunk is kept). Transforms are wired into IngestPipeline
via ``transforms: list[BaseTransform]`` (default empty = behaviour unchanged).
Injected keys live on Chunk.metadata; the pipeline forwards whitelisted keys
into the vector-store record (ADR 0011). See docs/spec-p10-transform.md.

Shared plumbing (review P3): the ``transform()`` template-method skeleton and
``_with_metadata`` live here so subclasses only implement ``_process_one`` —
the per-chunk rule/LLM logic — and the module-level ``load_prompt_text`` gives
subclasses the prompt-file override seam (review P1).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from rag_mcp_server.core.types import Chunk

__all__ = ["BaseTransform", "load_prompt_text"]


def load_prompt_text(prompt: str | None, prompt_path: str | None, default: str) -> str:
    """Resolve the prompt for a transform.

    Priority (highest first): explicit ``prompt`` (direct pass, unit tests /
    runtime override) > ``prompt_path`` file override (review P1) > built-in
    ``default`` constant. A missing/invalid prompt_path falls back to the
    default (config error must not crash the transform); a *valid* file whose
    content is empty is treated as an explicit override (the caller chose it).
    """
    if prompt is not None:
        return prompt
    if prompt_path is not None:
        path = Path(prompt_path)
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            if text:
                return text
    return default


class BaseTransform(ABC):
    """Transform a list of chunks into a new list (same or per-chunk degraded)."""

    def transform(
        self,
        chunks: list[Chunk],
        trace=None,
    ) -> list[Chunk]:
        """Return the transformed chunks.

        Template-method skeleton (review P3): run ``_process_one`` per chunk so
        a failure on one chunk keeps that chunk's original content — never
        aborts the whole batch. Must not mutate the input list in place.
        """
        out: list[Chunk] = []
        for chunk in chunks:
            out.append(self._process_one(chunk))
        return out

    @abstractmethod
    def _process_one(self, chunk: Chunk) -> Chunk:
        """Per-chunk transform logic; must never raise for the batch (the
        skeleton does not wrap it — subclasses own their try/except)."""


def _with_metadata(
    chunk: Chunk,
    extra: dict,
    text: str | None = None,
) -> Chunk:
    """Rebuild a chunk keeping identity + original metadata, merged with ``extra``.

    ``text`` defaults to the chunk's current text (enrichers don't change it);
    pass a new text when the transform rewrites it (ChunkRefiner). Review P3:
    one shared helper in base instead of the previous two near-identical copies.
    """
    return Chunk(
        document_id=chunk.document_id,
        text=text if text is not None else chunk.text,
        source_path=chunk.source_path,
        order=chunk.order,
        metadata={**(chunk.metadata or {}), **extra},
    )
