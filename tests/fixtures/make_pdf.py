"""Generate minimal valid PDFs for tests.

The minimal PDFs are hand-built (correct xref offsets) so pdfminer / markitdown
can parse them without adding a PDF-generation dependency (reportlab/pypdf).
Shared by test_pdf_loader, the ingestion pipeline tests, and the e2e tests.
"""

from __future__ import annotations


def _assemble(bodies: list[bytes]) -> bytes:
    """Wrap PDF body objects with a correct xref table and trailer."""
    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for body in bodies:
        offsets.append(len(out))
        out += body
    xref_start = len(out)
    out += f"xref\n0 {len(bodies) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer << /Size {len(bodies) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_start}\n%%EOF\n".encode()
    )
    return bytes(out)


def minimal_pdf(text: str = "Hello RAG PDF") -> bytes:
    """A single-page, single-text-line PDF containing ``text``."""
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    content = f"BT /F1 24 Tf 72 720 Td ({escaped}) Tj ET"
    bodies = [
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n",
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n",
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj\n",
        f"4 0 obj << /Length {len(content)} >> stream\n{content}\nendstream endobj\n".encode(),
        b"5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n",
    ]
    return _assemble(bodies)


def empty_content_pdf() -> bytes:
    """A valid single-page PDF whose content stream is empty (no text).

    Parsing succeeds (the file is not corrupt) but there is no extractable
    text — the "no extractable text" case, distinct from a broken PDF.
    """
    bodies = [
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n",
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n",
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << >> >> endobj\n",
        b"4 0 obj << /Length 0 >> stream\n\nendstream endobj\n",
    ]
    return _assemble(bodies)
