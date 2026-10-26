"""Image storage with SQLite indexing for multimodal RAG (P10.6, ticket 08).

Stores extracted PDF images on disk under ``data/images/{collection}/{image_id}.{ext}``
and keeps a document↔image mapping in a SQLite index (WAL). The vector store
records only carry ``image_refs`` id lists (ADR 0010 pt.9) — image bytes and
paths live here, decoupled from retrieval.

Contract (docs/completion-plan-p10.md P10.6 + .scratch/phase10/issues/08):
- ``save_image(image_id, data, mime_type, document_id, collection)`` writes the
  file and inserts/updates the index row in one call (single entry point, Q14).
  Idempotent: the same image_id overwrites file + row (INSERT OR REPLACE).
- ``lookup`` / ``list_by_document`` / ``delete_by_document`` round-trip the
  mapping; delete is best-effort (missing file unlink with missing_ok).
- mime_type is detected by magic bytes at save time (Q17) and drives the file
  extension (jpg/png); callers may override via the ``mime_type`` arg.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

MIME_TO_EXT = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/gif": "gif",
    "image/webp": "webp",
    "image/bmp": "bmp",
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS image_index (
    image_id    TEXT PRIMARY KEY,
    file_path   TEXT NOT NULL,
    collection  TEXT NOT NULL,
    document_id TEXT NOT NULL,
    mime_type   TEXT,
    page_num    INTEGER,
    created_at  TEXT
)
"""


class ImageStorage:
    """SQLite-backed store for extracted images (WAL, per-call connections)."""

    def __init__(
        self,
        db_path: Path | None = None,
        images_root: Path | None = None,
    ) -> None:
        self.db_path = Path(db_path) if db_path is not None else Path("data") / "db" / "image_index.db"
        self.images_root = Path(images_root) if images_root is not None else Path("data") / "images"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.images_root.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_schema(self) -> None:
        conn = self._connect()
        try:
            conn.execute(_SCHEMA)
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def detect_mime(data: bytes) -> str:
        """Detect image mime by magic bytes (Q17); default image/png.

        Public so the query/MCP side (multimodal_assembler) can re-detect the
        mime of a stored image with the same rule the save path used, without
        reaching into a private method across modules (S7 review fix).
        """
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if data.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if data.startswith((b"GIF87a", b"GIF89a")):
            return "image/gif"
        if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
            return "image/webp"
        if data.startswith(b"BM"):
            return "image/bmp"
        return "image/png"

    # Back-compat alias — the save path still calls the private name; kept so
    # callers don't need updating, but new code should use detect_mime.
    _detect_mime = detect_mime

    def save_image(
        self,
        image_id: str,
        data: bytes,
        mime_type: str | None = None,
        document_id: str = "",
        collection: str = "default",
        page_num: int | None = None,
    ) -> Path:
        """Write ``data`` to disk and index it; return the written file path.

        ``mime_type`` defaults to magic-byte detection (Q17); the file
        extension follows the detected/declared mime (png/jpg/...). Idempotent:
        a re-save of the same image_id overwrites file + row.
        """
        detected = self._detect_mime(data)
        mime = mime_type or detected
        ext = MIME_TO_EXT.get(mime, "png")
        image_dir = self.images_root / collection
        image_dir.mkdir(parents=True, exist_ok=True)
        target = image_dir / f"{image_id}.{ext}"
        target.write_bytes(data)

        conn = self._connect()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO image_index"
                " (image_id, file_path, collection, document_id, mime_type, page_num, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    image_id,
                    str(target),
                    collection,
                    document_id,
                    mime,
                    page_num,
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )
            conn.commit()
        finally:
            conn.close()
        return target

    def lookup(self, image_id: str) -> Path | None:
        """The stored file path for ``image_id`` (None when missing)."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT file_path FROM image_index WHERE image_id = ?",
                (image_id,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        path = Path(row[0])
        return path if path.is_file() else None

    def list_by_document(self, document_id: str) -> list[dict]:
        """All image rows belonging to ``document_id`` (id/path/mime/page)."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT image_id, file_path, mime_type, page_num, collection"
                " FROM image_index WHERE document_id = ? ORDER BY page_num, image_id",
                (document_id,),
            ).fetchall()
        finally:
            conn.close()
        return [
            {
                "image_id": image_id,
                "file_path": file_path,
                "mime_type": mime_type,
                "page_num": page_num,
                "collection": collection,
            }
            for image_id, file_path, mime_type, page_num, collection in rows
        ]

    def delete_by_document(self, document_id: str) -> int:
        """Delete a document's image files + index rows; return rows removed.

        Best-effort on the filesystem (missing files are not errors) — the
        index row is the source of truth and is always removed.
        """
        rows = self.list_by_document(document_id)
        conn = self._connect()
        try:
            deleted = conn.execute(
                "DELETE FROM image_index WHERE document_id = ?",
                (document_id,),
            ).rowcount
            conn.commit()
        finally:
            conn.close()
        for row in rows:
            path = Path(row["file_path"])
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        return deleted
