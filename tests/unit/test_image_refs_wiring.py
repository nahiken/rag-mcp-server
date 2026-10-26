"""Tests for P10.6 ticket 08 wiring — image channel through the pipeline.

Contract (grill 2026-08-26):
- IngestPipeline with image_storage injected: a PDF with images extracts them
  to disk + index, appends [IMAGE: id] placeholders, chunks carry image_refs,
  and the vector record metadata carries image_refs (whitelist).
- A PDF without images: no image_refs, no degradation.
- A corrupt PDF (fails text load) is skipped as before — image channel never
  turns a text-skip into an error.
- DocumentManager.delete with image_storage injected clears the document's
  images (best-effort, same isolation contract as integrity cleanup).
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from rag_mcp_server.document_manager.manager import DocumentManager
from rag_mcp_server.ingestion.pipeline import IngestPipeline
from rag_mcp_server.ingestion.storage.image_storage import ImageStorage
from rag_mcp_server.libs.embedding.fake import FakeEmbedding
from rag_mcp_server.libs.vector_store.chroma_store import ChromaVectorStore
from tests.fixtures.make_pdf import minimal_pdf


@pytest.fixture
def image_pdf_bytes() -> bytes:
    """A PDF with BOTH text (so load_document passes) and a DCTDecode image.

    PIL-generated PDFs are image-only (textless -> skipped by load_document),
    so this hand-built PDF pairs a text line with a JPEG XObject: markitdown
    extracts the text, pdfplumber extracts the image.
    """
    img = Image.new("RGB", (8, 8), "red")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    jpeg = buf.getvalue()

    content = b"BT /F1 24 Tf 72 720 Td (Hello image PDF) Tj ET\nq 100 0 0 100 72 500 cm /Im1 Do Q"
    hdr = (
        b"6 0 obj << /Type /XObject /Subtype /Image /Width 8 /Height 8 "
        b"/ColorSpace /DeviceRGB /Filter /DCTDecode /Length "
    )
    body6 = hdr + str(len(jpeg)).encode() + b" >> stream\n" + jpeg + b"\nendstream endobj\n"
    bodies = [
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n",
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n",
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> "
        b"/XObject << /Im1 6 0 R >> >> >> endobj\n",
        b"4 0 obj << /Length " + str(len(content)).encode() + b" >> stream\n" + content + b"\nendstream endobj\n",
        b"5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n",
        body6,
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for body in bodies:
        offsets.append(len(out))
        out += body
    xref = len(out)
    out += b"xref\n0 " + str(len(bodies) + 1).encode() + b"\n"
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        b"trailer << /Size " + str(len(bodies) + 1).encode()
        + b" /Root 1 0 R >>\nstartxref\n" + str(xref).encode() + b"\n%%EOF\n"
    )
    return bytes(out)


def _pipe(tmp_path, with_images: bool = True) -> tuple[IngestPipeline, ImageStorage, Path]:
    embedding = FakeEmbedding(dim=4)
    store = ChromaVectorStore(
        persist_dir=str(tmp_path / "storage" / "chroma"), collection="test_p106_wiring"
    )
    image_storage = ImageStorage(
        db_path=tmp_path / "db" / "image_index.db",
        images_root=tmp_path / "images",
    )
    pipe = IngestPipeline(
        embedding=embedding,
        store=store,
        index_dir=tmp_path / "storage" / "index",
        image_storage=image_storage,
        collection="test_p106_wiring",  # ADR 0016: pipeline collection == store
    )
    src = tmp_path / "source"
    src.mkdir(exist_ok=True)
    return pipe, image_storage, src


def test_ingest_pdf_with_images_extracts_and_refs(tmp_path, image_pdf_bytes):
    pipe, image_storage, src = _pipe(tmp_path)
    (src / "doc.pdf").write_bytes(image_pdf_bytes)

    summary = pipe.ingest(src, base=src)
    assert summary.total_files == 1
    assert summary.total_chunks >= 1

    # image saved to disk + indexed under the document
    # (document_id is the collection-prefixed slug of "doc.pdf", ADR 0016;
    # image_id flattens the :: prefix to -, see pdf_images._safe_id)
    images = image_storage.list_by_document("test_p106_wiring::doc-pdf")
    assert len(images) == 1
    assert images[0]["image_id"] == "test-p106-wiring-doc-pdf_p1_1"
    assert images[0]["mime_type"] == "image/jpeg"
    assert image_storage.lookup("test-p106-wiring-doc-pdf_p1_1") is not None

    # vector records carry image_refs (whitelisted via TRANSFORM_METADATA_KEYS)
    records = pipe.store.get_all()
    assert records
    refs = [r.metadata.get("image_refs") for r in records if r.metadata.get("image_refs")]
    assert refs, "至少一个 chunk 应携带 image_refs"
    assert "test-p106-wiring-doc-pdf_p1_1" in refs[0]


def test_ingest_text_pdf_has_no_image_refs(tmp_path):
    pipe, image_storage, src = _pipe(tmp_path)
    (src / "plain.pdf").write_bytes(minimal_pdf("Hello RAG PDF"))

    summary = pipe.ingest(src, base=src)
    assert summary.total_files == 1
    assert image_storage.list_by_document("plain-pdf") == []
    records = pipe.store.get_all()
    assert all("image_refs" not in r.metadata for r in records)


def test_ingest_pdf_with_images_degrade_flag_only_on_failure(tmp_path):
    """Q16 strict-degrade: a channel failure marks the document degraded but
    the text still ingests (images are optional). The flag never lands on the
    vector record (observation, not retrieval metadata)."""
    pipe, image_storage, src = _pipe(tmp_path)
    (src / "doc.pdf").write_bytes(minimal_pdf("text with no real image"))

    # Force the extraction channel to report degraded with no images. The
    # pipeline imports it lazily, so patch at the module that defines it.
    import rag_mcp_server.ingestion.pdf_images as pdf_images_mod

    monkeypatch = pytest.MonkeyPatch()

    def _fake_extract(_pdf_path, _document_id):
        return [], True

    monkeypatch.setattr(pdf_images_mod, "extract_pdf_images", _fake_extract)
    try:
        summary = pipe.ingest(src, base=src)
    finally:
        monkeypatch.undo()
    assert summary.total_files == 1
    # text still ingested (images are optional), no image_refs on records
    records = pipe.store.get_all()
    assert all("image_refs" not in r.metadata for r in records)


def test_ingest_without_image_storage_is_unchanged(tmp_path, image_pdf_bytes):
    """image_storage=None keeps existing behaviour: no extraction, no refs."""
    embedding = FakeEmbedding(dim=4)
    store = ChromaVectorStore(
        persist_dir=str(tmp_path / "storage" / "chroma"), collection="test_p106_noimg"
    )
    pipe = IngestPipeline(
        embedding=embedding,
        store=store,
        index_dir=tmp_path / "storage" / "index",
        # no image_storage
    )
    src = tmp_path / "source"
    src.mkdir()
    (src / "doc.pdf").write_bytes(image_pdf_bytes)

    summary = pipe.ingest(src, base=src)
    assert summary.total_files == 1
    records = store.get_all()
    assert all("image_refs" not in r.metadata for r in records)


def test_delete_with_image_storage_clears_document_images(tmp_path, image_pdf_bytes):
    """Q11: DocumentManager.delete with image_storage injected clears images."""
    embedding = FakeEmbedding(dim=4)
    store = ChromaVectorStore(
        persist_dir=str(tmp_path / "storage" / "chroma"), collection="test_p106_del"
    )
    image_storage = ImageStorage(
        db_path=tmp_path / "db" / "image_index.db",
        images_root=tmp_path / "images",
    )
    pipe = IngestPipeline(
        embedding=embedding,
        store=store,
        index_dir=tmp_path / "storage" / "index",
        image_storage=image_storage,
        collection="test_p106_del",  # ADR 0016
    )
    src = tmp_path / "source"
    src.mkdir()
    (src / "doc.pdf").write_bytes(image_pdf_bytes)

    pipe.ingest(src, base=src)
    # ADR 0016: document_id carries the collection prefix
    doc_id = "test_p106_del::doc-pdf"
    assert image_storage.list_by_document(doc_id)

    manager = DocumentManager(store=store, image_storage=image_storage)
    removed = manager.delete(doc_id)
    assert removed >= 1
    assert image_storage.list_by_document(doc_id) == []
    assert image_storage.lookup("test_p106_del-doc-pdf_p1_1") is None  # file gone too
