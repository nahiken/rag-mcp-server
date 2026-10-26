"""Tests for pdf_images — pdfplumber extraction channel (P10.6, ticket 08).

Red/green contract (ADR 0010 + grill 2026-08-26):
- extract_pdf_images opens the SAME PDF independently of markitdown text path
- a PDF with a DCTDecode image yields one (image_id, jpeg_bytes, image/jpeg, 1)
- image_id follows {document_id}_p{page}_{seq} (Q13)
- a text-only / no-image PDF yields ([], degraded=False) (never raises)
- a corrupt PDF yields ([], degraded=True) (Q16 strict-degrade)

Uses PIL to generate a real image-bearing PDF (PIL always embeds DCTDecode/JPEG,
measured 2026-08-26), so the JPEG magic-byte path is covered without hand-built
PDFs. The markitdown text loader stays untouched — only this module is tested.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from rag_mcp_server.ingestion.pdf_images import extract_pdf_images, image_id_for
from tests.fixtures.make_pdf import minimal_pdf


@pytest.fixture
def image_pdf(tmp_path) -> Path:
    """A single-page PDF with one embedded (DCTDecode) image."""
    img = Image.new("RGB", (8, 8), "red")
    buf = io.BytesIO()
    img.save(buf, format="PDF", resolution=72)
    path = tmp_path / "with-image.pdf"
    path.write_bytes(buf.getvalue())
    return path


def test_image_id_for_readable_and_safe():
    assert image_id_for("report", 1, 2) == "report_p1_2"
    # document_id may contain separators; they collapse to safe id segments
    assert image_id_for("my dir/报告", 3, 1) == "my-dir-报告_p3_1"


def test_extract_dctdecode_image(image_pdf):
    images, degraded = extract_pdf_images(image_pdf, "with-image")
    assert len(images) == 1
    assert degraded is False
    image_id, data, mime, page = images[0]
    assert image_id == "with-image_p1_1"
    assert mime == "image/jpeg"
    assert data.startswith(b"\xff\xd8\xff")  # JPEG magic
    assert page == 1


def test_extract_text_only_pdf_returns_empty(tmp_path):
    path = tmp_path / "plain.pdf"
    path.write_bytes(minimal_pdf("Hello RAG PDF"))
    images, degraded = extract_pdf_images(path, "plain")
    assert images == []
    assert degraded is False


def test_extract_corrupt_pdf_returns_degraded(tmp_path):
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"%PDF-1.4 fake\n")
    images, degraded = extract_pdf_images(path, "broken")
    assert images == []
    assert degraded is True
