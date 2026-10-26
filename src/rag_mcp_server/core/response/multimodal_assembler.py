"""Multimodal assembly — turn image references on a RetrievalResult into content.

P10.6 T3 (issue 09): a query that hits an image-bearing chunk carries
``metadata["image_refs"]`` (ids only — bytes/paths live in ImageStorage, ADR
0010 pt.9). ``build_image_blocks`` resolves each id through an injected
ImageStorage, reads the stored bytes, detects the mime by magic bytes (the same
rule the save path used) and returns MCP ``ImageContent`` blocks. Missing
storage / missing image / oversized image are all skipped so a text result
always comes back — the image channel is an optional enhancement, never a
dependency (US10).

The MCP server holds no live ImageStorage reference (it runs in a separate
process from ingestion); ``build_image_storage`` lazily builds the default one
rooted at settings, so the query process reads exactly the images the ingest
process wrote to disk.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path

from mcp.types import ImageContent

from rag_mcp_server.core.types import RetrievalResult
from rag_mcp_server.ingestion.storage.image_storage import ImageStorage

logger = logging.getLogger(__name__)

# The default response-safety cap for a single image in an ImageContent block.
# The MCP SDK 2.x wire format is a JSON base64 payload; MCP spec allows large
# payloads but we keep a sane default so an oversized diagram cannot balloon a
# tool response. Missing/oversized images are skipped, never fatal (US10).
DEFAULT_MAX_IMAGE_BYTES = 2_000_000


def build_image_storage() -> ImageStorage:
    """The default ImageStorage for the query/MCP process.

    Reads the same defaults as the ingestion side (``data/db/image_index.db`` +
    ``data/images``) so a separate query process resolves the images the ingest
    process wrote. ``settings`` may override via an optional ``image`` /
    ``image_storage`` top-level section (forward-compatible with P10.8 wiring).
    """
    try:
        from rag_mcp_server.config.settings import load_settings

        cfg = load_settings().get("image_storage", {})
        db_path = Path(cfg["db_path"]) if cfg.get("db_path") else None
        images_root = Path(cfg["images_root"]) if cfg.get("images_root") else None
    except Exception as exc:  # noqa: BLE001 — settings must never break a query
        logger.warning("image_storage settings 读取失败，用默认根：%s", exc)
        db_path = None
        images_root = None
    return ImageStorage(db_path=db_path, images_root=images_root)


def build_image_blocks(
    result: RetrievalResult,
    image_storage: ImageStorage | None,
    max_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
) -> list[ImageContent]:
    """The MCP ImageContent blocks for a result's ``image_refs``.

    Each referenced image is looked up, read and emitted as an ``ImageContent``
    block (base64 data + mime detected from the file bytes). A missing storage,
    a missing/oversized image or a read failure skips that image (logged) and
    the remaining blocks are still returned; a result with no image_refs yields
    ``[]``. Never raises — the text result is always returned (US10).
    """
    refs = (result.metadata or {}).get("image_refs")
    if not refs or image_storage is None:
        return []

    blocks: list[ImageContent] = []
    for image_id in refs:
        block = _image_block(image_id, image_storage, max_bytes)
        if block is not None:
            blocks.append(block)
    return blocks


def _image_block(
    image_id: str,
    image_storage: ImageStorage,
    max_bytes: int,
) -> ImageContent | None:
    """One ImageContent for ``image_id``, or None when the image is unusable."""
    try:
        path = image_storage.lookup(image_id)
        if path is None:
            logger.warning("图片缺失，跳过出图（%s）", image_id)
            return None
        data = path.read_bytes()
        if len(data) > max_bytes:
            logger.warning("图片过大，跳过出图（%s，%d bytes > %d）", image_id, len(data), max_bytes)
            return None
        mime = ImageStorage.detect_mime(data)
        return ImageContent(
            type="image",
            data=base64.b64encode(data).decode("ascii"),
            mime_type=mime,
        )
    except Exception as exc:  # noqa: BLE001 — per-image best-effort, never block
        logger.warning("图片出图失败，跳过（%s）：%s", image_id, exc)
        return None
