"""Tests for ImageStorage — save/lookup/list/delete round-trip (P10.6, ticket 08).

Contract (grill 2026-08-26, Q12/Q13/Q14/Q17):
- save_image writes data/images/{collection}/{image_id}.{ext} AND inserts the
  SQLite row in one call; idempotent (same id overwrites file + row)
- image_id = f"{document_id}_p{page}_{seq}" convention (tested at the wiring
  level, not here)
- document_id is a first-class column (list_by_document / delete_by_document)
- mime detected by magic bytes: PNG / JPEG → correct extension
- delete_by_document removes rows + files; missing file not an error
- SQLite persistence: a fresh ImageStorage on the same paths reopens rows
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from rag_mcp_server.ingestion.storage.image_storage import ImageStorage

# A real 1x1 PNG (magic bytes + minimal IHDR/IDAT/IEND).
PNG_BYTES = bytes.fromhex(
    "89504e470d0a1a0a"  # PNG signature
    "0000000d49484452"  # IHDR len + type
    "00000001000000010802000000"  # 1x1, 8-bit, RGB
    "90bb4f7b"  # IHDR CRC
    "0000000b49444154"  # IDAT len + type
    "789c6260000002ff00060100"  # deflate
    "c9e4ac05"  # IDAT CRC
    "0000000049454e44"  # IEND
    "ae426082"  # IEND CRC
)

# A minimal JPEG (magic bytes + SOI + EOI).
JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"\x00" * 8 + b"\xff\xd9"


@pytest.fixture
def storage(tmp_path) -> ImageStorage:
    return ImageStorage(
        db_path=tmp_path / "db" / "image_index.db",
        images_root=tmp_path / "images",
    )


def test_save_writes_file_and_index_row(storage, tmp_path):
    path = storage.save_image(
        "doc-a_p1_1", PNG_BYTES, document_id="doc-a", collection="c1", page_num=1
    )
    assert path.name == "doc-a_p1_1.png"  # PNG mime -> .png
    assert path.is_file()
    assert path.read_bytes() == PNG_BYTES
    # index row exists
    rows = storage.list_by_document("doc-a")
    assert len(rows) == 1
    assert rows[0]["image_id"] == "doc-a_p1_1"
    assert rows[0]["mime_type"] == "image/png"
    assert rows[0]["page_num"] == 1
    assert rows[0]["collection"] == "c1"


def test_save_jpeg_detects_mime_and_extension(storage):
    path = storage.save_image("img2", JPEG_BYTES, document_id="doc-a")
    assert path.name == "img2.jpg"  # JPEG magic -> .jpg
    assert path.read_bytes() == JPEG_BYTES
    row = storage.list_by_document("doc-a")[0]
    assert row["mime_type"] == "image/jpeg"


def test_lookup_returns_path_for_existing_and_none_for_missing(storage):
    storage.save_image("img1", PNG_BYTES, document_id="doc-a")
    found = storage.lookup("img1")
    assert found is not None and found.is_file()
    assert storage.lookup("nope") is None


def test_list_by_document_filters_by_document(storage):
    storage.save_image("a1", PNG_BYTES, document_id="doc-a", page_num=1)
    storage.save_image("a2", PNG_BYTES, document_id="doc-a", page_num=2)
    storage.save_image("b1", PNG_BYTES, document_id="doc-b", page_num=1)
    a_ids = [r["image_id"] for r in storage.list_by_document("doc-a")]
    b_ids = [r["image_id"] for r in storage.list_by_document("doc-b")]
    assert a_ids == ["a1", "a2"]  # ordered by page_num
    assert b_ids == ["b1"]


def test_delete_by_document_removes_rows_and_files(storage):
    storage.save_image("a1", PNG_BYTES, document_id="doc-a", page_num=1)
    storage.save_image("a2", PNG_BYTES, document_id="doc-a", page_num=2)
    storage.save_image("b1", PNG_BYTES, document_id="doc-b", page_num=1)
    removed = storage.delete_by_document("doc-a")
    assert removed == 2
    assert storage.list_by_document("doc-a") == []
    assert storage.lookup("a1") is None  # file gone
    assert storage.lookup("a2") is None
    # other document untouched
    assert storage.list_by_document("doc-b")[0]["image_id"] == "b1"
    assert storage.lookup("b1").is_file()


def test_delete_missing_document_is_idempotent(storage):
    assert storage.delete_by_document("nope") == 0


def test_save_same_image_id_is_idempotent_overwrite(storage):
    path1 = storage.save_image("dup", PNG_BYTES, document_id="doc-a")
    path2 = storage.save_image("dup", JPEG_BYTES, document_id="doc-a")
    # overwrites file + row; only one row
    assert path2.read_bytes() == JPEG_BYTES
    assert len(storage.list_by_document("doc-a")) == 1
    assert storage.lookup("dup").read_bytes() == JPEG_BYTES


def test_sqlite_persistence_across_instances(tmp_path):
    db = tmp_path / "db" / "image_index.db"
    root = tmp_path / "images"
    first = ImageStorage(db_path=db, images_root=root)
    first.save_image("persist1", PNG_BYTES, document_id="doc-a")
    # a fresh instance on the same paths reopens the rows (WAL persisted)
    second = ImageStorage(db_path=db, images_root=root)
    assert second.lookup("persist1") is not None
    assert second.list_by_document("doc-a")[0]["image_id"] == "persist1"


def test_unknown_bytes_default_to_png(storage):
    path = storage.save_image("unknown", b"\x00\x01\x02\x03", document_id="doc-a")
    assert path.name == "unknown.png"
    assert path.read_bytes() == b"\x00\x01\x02\x03"
