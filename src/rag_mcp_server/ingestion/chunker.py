"""Split a Document into retrieval chunks.

First version strategy (DEV_SPEC section 7):
1. Split on blank lines into paragraphs.
2. Drop blank paragraphs.
3. A normal paragraph becomes one chunk.
4. An over-long paragraph is split further by max_chars.
"""

from __future__ import annotations

from rag_mcp_server.core.types import Chunk, Document
from rag_mcp_server.ingestion.pdf_images import IMAGE_PLACEHOLDER_RE

DEFAULT_MAX_CHARS = 800


def _image_refs_of(text: str) -> list[str]:
    """The image ids referenced in a chunk's text (P10.6 image_refs metadata)."""
    return [m.strip() for m in IMAGE_PLACEHOLDER_RE.findall(text)]


def chunk_document(document: Document, max_chars: int = DEFAULT_MAX_CHARS) -> list[Chunk]:
    paragraphs = [p.strip() for p in document.text.split("\n\n")]
    paragraphs = [p for p in paragraphs if p]

    chunks: list[Chunk] = []
    order = 0
    for paragraph in paragraphs:
        pieces = (
            [paragraph[i : i + max_chars] for i in range(0, len(paragraph), max_chars)]
            if len(paragraph) > max_chars
            else [paragraph]
        )
        for piece in pieces:
            metadata = {"file_type": document.file_type}
            refs = _image_refs_of(piece)
            if refs:
                # P10.6: chunks that contain [IMAGE: id] placeholders carry the
                # referenced ids so the record metadata lets MCP return images
                # (whitelisted via TRANSFORM_METADATA_KEYS, ADR 0011).
                metadata["image_refs"] = refs
            chunks.append(
                Chunk(
                    document_id=document.id,
                    text=piece,
                    source_path=document.source_path,
                    order=order,
                    metadata=metadata,
                )
            )
            order += 1
    return chunks
