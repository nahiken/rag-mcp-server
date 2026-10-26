"""Persist chunks as JSONL and write ingestion reports.

First version persistence format (DEV_SPEC): one chunk per line in
chunks.jsonl; the keyword index is deliberately not persisted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rag_mcp_server.core.errors import ChunkStoreError
from rag_mcp_server.core.types import Chunk


def _chunk_to_dict(chunk: Chunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "text": chunk.text,
        "source_path": chunk.source_path,
        "order": chunk.order,
        "metadata": chunk.metadata,
    }


def _chunk_from_dict(data: dict[str, Any]) -> Chunk:
    return Chunk(
        document_id=data["document_id"],
        text=data["text"],
        source_path=data["source_path"],
        order=data["order"],
        metadata=data.get("metadata", {}),
    )


def write_chunks(chunks: list[Chunk], path: Path) -> None:
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for chunk in chunks:
                handle.write(json.dumps(_chunk_to_dict(chunk), ensure_ascii=False) + "\n")
    except OSError as exc:
        raise ChunkStoreError(f"cannot write chunks to {path}: {exc}") from exc


def read_chunks(path: Path) -> list[Chunk]:
    path = Path(path)
    if not path.is_file():
        raise ChunkStoreError(f"chunks store not found: {path}")
    chunks: list[Chunk] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                chunks.append(_chunk_from_dict(data))
    except (OSError, json.JSONDecodeError) as exc:
        raise ChunkStoreError(f"cannot read chunks from {path}: {exc}") from exc
    return chunks


def write_ingestion_report(report: dict[str, Any], path: Path) -> None:
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
    except OSError as exc:
        raise ChunkStoreError(f"cannot write ingestion report to {path}: {exc}") from exc
