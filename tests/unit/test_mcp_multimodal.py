"""Tests for the multimodal MCP assembly seam (P10.6 T3).

Red/green contract (.scratch/phase10/subtickets/image-captioner/T3-retrieval-mcp.md):
- build_image_blocks(result, image_storage) -> list[ImageContent]: a result with
  image_refs emits one ImageContent per resolved image (base64 + mime).
- no image_refs / no storage -> [] (text-only, current behaviour unchanged).
- missing image file / oversized image / storage error -> that image skipped,
  the rest still emitted; never raises (US10).
- build_image_storage returns a default ImageStorage rooted at settings.

Prior art: test_image_refs_wiring.py (image_storage fixture), test_mcp_ext.py
(tool return shape).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from mcp.types import ImageContent

from rag_mcp_server.core.response.multimodal_assembler import (
    build_image_blocks,
    build_image_storage,
)
from rag_mcp_server.core.types import RetrievalResult
from rag_mcp_server.ingestion.storage.image_storage import ImageStorage


def _storage(tmp_path) -> ImageStorage:
    return ImageStorage(
        db_path=tmp_path / "db" / "image_index.db",
        images_root=tmp_path / "images",
    )


def _png_bytes() -> bytes:
    # minimal 1x1 PNG (magic 8950 4e47 0d0a 1a0a) — ImageStorage._detect_mime
    return (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
        b"\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n"
        b"\x2d\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )


def _result(refs=None) -> RetrievalResult:
    metadata = {"image_refs": refs} if refs else {}
    return RetrievalResult(
        chunk_id="d::0",
        source_path="p.pdf",
        score=0.8,
        text="带图 chunk",
        sources=["dense"],
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# build_image_blocks — happy path
# ---------------------------------------------------------------------------
def test_build_image_blocks_emits_image_content(tmp_path):
    storage = _storage(tmp_path)
    storage.save_image("d_p1_1", _png_bytes(), mime_type="image/png", document_id="d")
    blocks = build_image_blocks(_result(["d_p1_1"]), storage)
    assert len(blocks) == 1
    block = blocks[0]
    assert isinstance(block, ImageContent)
    assert block.type == "image"
    assert block.mime_type == "image/png"
    # base64 round-trips back to the stored bytes
    import base64

    assert base64.b64decode(block.data) == _png_bytes()


def test_build_image_blocks_multiple_refs_order_preserved(tmp_path):
    storage = _storage(tmp_path)
    storage.save_image("d_p1_1", _png_bytes(), mime_type="image/png", document_id="d")
    storage.save_image("d_p1_2", _png_bytes(), mime_type="image/png", document_id="d")
    blocks = build_image_blocks(_result(["d_p1_1", "d_p1_2"]), storage)
    assert [b.type for b in blocks] == ["image", "image"]


# ---------------------------------------------------------------------------
# no-image compatibility (US11) / no storage / errors never block (US10)
# ---------------------------------------------------------------------------
def test_build_image_blocks_no_refs_yields_empty(tmp_path):
    storage = _storage(tmp_path)
    assert build_image_blocks(_result(None), storage) == []
    assert build_image_blocks(_result([]), storage) == []


def test_build_image_blocks_none_storage_yields_empty(tmp_path):
    storage = _storage(tmp_path)
    storage.save_image("d_p1_1", _png_bytes(), mime_type="image/png", document_id="d")
    assert build_image_blocks(_result(["d_p1_1"]), None) == []


def test_build_image_blocks_missing_image_file_skipped(tmp_path):
    storage = _storage(tmp_path)
    # id referenced but never saved -> lookup returns None -> skip
    assert build_image_blocks(_result(["ghost"]), storage) == []


def test_build_image_blocks_missing_file_skips_that_image_keeps_others(tmp_path):
    storage = _storage(tmp_path)
    storage.save_image("d_p1_1", _png_bytes(), mime_type="image/png", document_id="d")
    blocks = build_image_blocks(_result(["d_p1_1", "ghost"]), storage)
    assert len(blocks) == 1
    assert blocks[0].data  # the resolvable one came back


def test_build_image_blocks_oversized_image_skipped(tmp_path):
    storage = _storage(tmp_path)
    storage.save_image("big", _png_bytes(), mime_type="image/png", document_id="d")
    # cap below the real size -> skipped, not fatal
    assert build_image_blocks(_result(["big"]), storage, max_bytes=1) == []


def test_build_image_blocks_read_error_skipped(tmp_path, monkeypatch):
    storage = _storage(tmp_path)
    storage.save_image("d_p1_1", _png_bytes(), mime_type="image/png", document_id="d")

    def _boom(*a, **k):
        raise OSError("io boom")

    monkeypatch.setattr(Path, "read_bytes", _boom)
    assert build_image_blocks(_result(["d_p1_1"]), storage) == []


# ---------------------------------------------------------------------------
# build_image_storage — defaults rooted at settings
# ---------------------------------------------------------------------------
def test_build_image_storage_uses_defaults(tmp_path, monkeypatch):
    import rag_mcp_server.config.settings as settings_mod

    monkeypatch.setattr(settings_mod, "load_settings", lambda: {})
    storage = build_image_storage()
    assert storage.images_root == Path("data") / "images"
    assert storage.db_path == Path("data") / "db" / "image_index.db"


def test_build_image_storage_uses_configured_root(tmp_path, monkeypatch):
    import rag_mcp_server.config.settings as settings_mod

    cfg = {
        "image_storage": {
            "db_path": str(tmp_path / "custom" / "idx.db"),
            "images_root": str(tmp_path / "custom" / "imgs"),
        }
    }
    monkeypatch.setattr(settings_mod, "load_settings", lambda: cfg)
    storage = build_image_storage()
    assert storage.images_root == tmp_path / "custom" / "imgs"
    assert storage.db_path == tmp_path / "custom" / "idx.db"


def test_build_image_storage_logs_settings_fallback(monkeypatch, caplog):
    import rag_mcp_server.config.settings as settings_mod

    def boom():
        raise RuntimeError("bad settings")

    monkeypatch.setattr(settings_mod, "load_settings", boom)
    with caplog.at_level("WARNING", logger="rag_mcp_server.core.response.multimodal_assembler"):
        storage = build_image_storage()

    assert storage.images_root == Path("data") / "images"
    assert storage.db_path == Path("data") / "db" / "image_index.db"
    assert "image_storage settings 读取失败" in caplog.text
    assert "bad settings" in caplog.text
