"""Load PDF files into plain text.

Contract (docs/spec-p9-pdf.md, task 5): ``load_pdf(path) -> str | None``

- success: returns the extracted plain text (markitdown output)
- corrupt / fake PDF: returns ``None`` (skip signal, never raises); the real
  parse reason is kept in a warning log for debugging
- valid but textless PDF (scanned / image-only): returns ``""`` — distinct
  from the corrupt ``None`` case, so the caller can name the skip reason

No suffix guard and no text trimming: the loader takes a PDF path and returns
what was extracted, matching load_document's raw read behaviour.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_converter: object | None = None


def _markitdown_converter() -> object:
    global _converter
    if _converter is None:
        from markitdown import MarkItDown

        _converter = MarkItDown()
    return _converter


def load_pdf(path: Path) -> str | None:
    """Parse ``path`` into plain text, or return a skip signal.

    Returns ``None`` for a corrupt/fake PDF and ``""`` for a valid but textless
    PDF; neither raises. A warning log keeps the real parse reason when a PDF
    cannot be read.
    """
    path = Path(path)
    try:
        result = _markitdown_converter().convert(path)
    except Exception as exc:  # noqa: BLE001 - any parse failure becomes a skip
        logger.warning("pdf parse failed for %s: %s", path, exc)
        return None
    return result.text_content
