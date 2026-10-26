"""Tests for the PDF loader seam — load_pdf external behaviour.

Red/green contract (docs/spec-p9-pdf.md, task 5, tickets 05-T1/T2/T3):
- a valid PDF parses to its text via the real pdfminer path (hand-built fixture,
  not mocked)
- a corrupt / fake PDF returns None (skipped signal), never raises
- a valid but textless PDF (empty content stream) returns "" — distinct from
  the corrupt None case

The only seam under test is the public load_pdf contract; the markitdown
converter is not mocked.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_mcp_server.ingestion.pdf_loader import load_pdf

from tests.fixtures.make_pdf import empty_content_pdf, minimal_pdf


def test_load_pdf_returns_extracted_text(tmp_path):
    path = tmp_path / "hello.pdf"
    path.write_bytes(minimal_pdf("Hello RAG PDF"))
    text = load_pdf(path)
    assert text is not None
    assert "Hello RAG PDF" in text


@pytest.mark.parametrize(
    "payload",
    [
        # "%PDF-1.4 fake\n" (trailing newline) makes pdfminer reject the file and
        # markitdown raise; a bare "%PDF-1.4 fake" without newline falls back to
        # returning the bytes as text instead of raising.
        b"%PDF-1.4 fake\n",
        b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\n",  # no /Root -> PDFSyntaxError
        b"",  # empty file is not a parseable PDF
    ],
)
def test_load_pdf_corrupt_pdf_returns_none(tmp_path, payload):
    path = tmp_path / "broken.pdf"
    path.write_bytes(payload)
    text = load_pdf(path)
    # corrupt PDFs must be a skip signal (None) and must never leak their raw
    # bytes as retrievable text
    assert text is None


def test_load_pdf_textless_pdf_returns_empty_string(tmp_path):
    path = tmp_path / "scanned.pdf"
    path.write_bytes(empty_content_pdf())
    assert load_pdf(path) == ""
