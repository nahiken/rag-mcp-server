"""PDF image extraction — pdfplumber independent channel (P10.6, ticket 08).

Sibling to the markitdown text path (pdf_loader.load_pdf). ADR 0010: markitdown
only extracts text/table, never images; a separate pdfplumber pass pulls the
images from the SAME PDF (parsed twice — cheap, and keeps the text contract
frozen).

Returns ``(images, degraded)`` mirroring load_pdf's ``str|None`` triple:
- ``images == []``: the PDF scanned cleanly and has no images (like ``""``).
- ``images == [(image_id, bytes, mime, page), ...]``: successfully extracted.
- ``degraded == True``: extraction had a failure (open failed / an image could
  not be decoded) — the caller should mark ``has_unprocessed_images`` (Q16
  strict-degrade). Successfully extracted images are still returned.

Extraction rules (measured 2026-08-26):
- ``page.images[i]["stream"].get_data()`` returns the decoded bytes.
  DCTDecode → a complete JPEG file (magic \\xff\\xd8\\xff) usable as-is.
  FlateDecode → raw RGB(A) pixel rows (no filter bytes); reconstructed to a
  real PNG via PIL ``Image.frombytes`` + vertical flip (PDF y-up, PIL y-down).
- mime is decided by magic bytes on the extracted data (Q17), NOT by the
  /Filter alone, because a Flate image becomes a PNG after reconstruction.
"""

from __future__ import annotations

import io
import logging
import re

from rag_mcp_server.ingestion.storage.image_storage import ImageStorage

logger = logging.getLogger(__name__)

# Same slug rule as core.types.make_document_id: non-word chars collapse to a
# single dash, Unicode (CJK) letters are PRESERVED so Chinese document ids keep
# distinct, readable image ids (P9 regression — \W stripped CJK).
_SAFE_ID_RE = re.compile(r"[\W_]+")


def _safe_id(segment: str) -> str:
    return _SAFE_ID_RE.sub("-", segment).strip("-")


# image id shape: {document_id}_p{page}_{seq}. Single source of truth for the
# shape's tail — the image captioner (image_captioner.py) parses the same shape
# out of the placeholder (FIX-3: shared contract, no shotgun edit).
IMAGE_ID_SUFFIX_RE = re.compile(r"_p(\d+)_(\d+)$")

# [IMAGE: id] placeholder in the text stream. Single source of truth for the
# token: emitted by the pdfplumber channel, inherited by chunks (chunker.py)
# and replaced by the captioner (image_captioner.py) — S9 review: one regex,
# no divergent copies.
IMAGE_PLACEHOLDER_RE = re.compile(r"\[IMAGE:\s*([^\]]+)\]")


def image_id_for(document_id: str, page: int, seq: int) -> str:
    """Deterministic, readable image id (Q13): ``{document_id}_p{page}_{seq}``."""
    return f"{_safe_id(document_id)}_p{page}_{seq}"


def _flate_to_png(stream, width: int, height: int, colorspace: str, bits: int) -> bytes | None:
    """Reconstruct a PNG from raw FlateDecode pixel data via PIL.

    PDF stores pixel rows bottom-up (y=0 at the bottom); PIL expects top-down,
    so the image is vertically flipped after frombytes. Grayscale/RGB/RGBA by
    colorspace; a missing/broken decode returns None (best-effort).
    """
    from PIL import Image, ImageOps

    mode_map = {
        "DeviceRGB": "RGB",
        "DeviceGray": "L",
        "DeviceCMYK": "CMYK",
    }
    if colorspace not in mode_map or bits != 8:
        return None
    raw = stream.get_data()
    mode = mode_map[colorspace]
    expected = width * height * len(mode)
    if len(raw) < expected:
        return None
    try:
        im = Image.frombytes(mode, (width, height), raw[:expected])
        im = ImageOps.flip(im)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as exc:  # noqa: BLE001 — best-effort extraction
        logger.warning("flate 图片转 PNG 失败（宽=%s 高=%s）：%s", width, height, exc)
        return None


def extract_pdf_images(
    pdf_path,
    document_id: str,
) -> tuple[list[tuple[str, bytes, str, int]], bool]:
    """Extract images from ``pdf_path``; return ``(images, degraded)``.

    Images are ``(image_id, data, mime_type, page_num)`` in page order.
    DCTDecode images are returned as their embedded JPEG bytes; FlateDecode
    images are reconstructed to PNG. ``degraded`` is True when the channel had
    any failure (open error or an undecodable image) — the caller marks
    ``has_unprocessed_images`` (Q16 strict-degrade); extracted images are still
    returned so a partially-failed document keeps the ones we did get.
    """
    import pdfplumber

    results: list[tuple[str, bytes, str, int]] = []
    degraded = False
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_index, page in enumerate(pdf.pages, start=1):
                for seq, img_info in enumerate(page.images, start=1):
                    stream = img_info.get("stream")
                    if stream is None:
                        degraded = True
                        continue
                    width = int(img_info.get("srcsize", (0, 0))[0] or 0)
                    height = int(img_info.get("srcsize", (0, 0))[1] or 0)
                    colorspace = img_info.get("colorspace") or ["/DeviceRGB"]
                    # pdfminer yields PSLiteral objects ("/DeviceRGB"), not str
                    cs_name = str(colorspace[0]).lstrip("/") if colorspace else "DeviceRGB"
                    filters = stream.get_filters()
                    # /Filter entries are PSLiteral objects, not str
                    flate = any("FlateDecode" in str(f[0]) for f in filters)
                    data: bytes | None = None
                    try:
                        if flate:
                            data = _flate_to_png(
                                stream, width, height, cs_name, int(img_info.get("bits", 8))
                            )
                        else:
                            data = stream.get_data()
                    except Exception as exc:  # noqa: BLE001 — per-image best-effort
                        logger.warning("图片解码失败（%s %s）：%s", pdf_path, seq, exc)
                    if not data:
                        degraded = True
                        continue
                    mime = ImageStorage._detect_mime(data)
                    results.append(
                        (
                            image_id_for(document_id, page_index, seq),
                            data,
                            mime,
                            page_index,
                        )
                    )
    except Exception as exc:  # noqa: BLE001 — channel-level failure -> degraded
        logger.warning("pdfplumber 提图失败（%s）：%s", pdf_path, exc)
        return [], True
    return results, degraded
