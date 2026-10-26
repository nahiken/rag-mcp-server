"""Resolve ingest targets and load importable documents.

Unsupported file types are not errors: they are collected as
SkippedFile records and reported, never raised.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rag_mcp_server.core.errors import DocumentLoadError
from rag_mcp_server.core.types import DEFAULT_COLLECTION, Document, make_document_id

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


@dataclass
class SkippedFile:
    path: Path
    reason: str


def resolve_targets(path: Path) -> tuple[list[Path], list[SkippedFile]]:
    """Resolve a file or directory path into supported files and skipped files."""
    path = Path(path)
    if path.is_file():
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            return [path], []
        return [], [SkippedFile(path=path, reason=f"unsupported file type: {path.suffix}")]

    files: list[Path] = []
    skipped: list[SkippedFile] = []
    if not path.is_dir():
        return [], [SkippedFile(path=path, reason="path does not exist")]

    for candidate in sorted(path.rglob("*")):
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(candidate)
        else:
            skipped.append(
                SkippedFile(
                    path=candidate,
                    reason=f"unsupported file type: {candidate.suffix}",
                )
            )
    return files, skipped


def load_document(file_path: Path, base: Path, collection: str = DEFAULT_COLLECTION) -> Document:
    """Load a supported file into a Document using a relative source path.

    txt/md are read as UTF-8 text; pdf is dispatched to the PDF loader and its
    extracted text used. A corrupt or textless PDF raises DocumentLoadError so
    the caller (ingestion pipeline) can collect it as skipped rather than
    failing the whole ingest.

    ``collection`` prefixes the document id (``{collection}::{slug}``, ADR
    0016) so the same file ingested into different collections stays distinct.
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        raise DocumentLoadError(f"file not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        from rag_mcp_server.ingestion.pdf_loader import load_pdf

        text = load_pdf(file_path)
        if text is None or text == "":
            reason = "pdf parse failed" if text is None else "no extractable text"
            raise DocumentLoadError(f"cannot extract text from {file_path}: {reason}")
    else:
        try:
            text = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            # non-UTF-8 txt/md must not crash the whole ingest: it is a load
            # failure like any other, collected as skipped by the caller
            raise DocumentLoadError(f"cannot read {file_path}: {exc}") from exc

    relative = file_path.relative_to(base)
    return Document(
        id=make_document_id(relative, collection=collection),
        text=text,
        source_path=relative.as_posix(),
        file_type=file_path.suffix.lstrip("."),
    )
