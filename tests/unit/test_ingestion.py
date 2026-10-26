from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rag_mcp_server.core.errors import DocumentLoadError
from rag_mcp_server.ingestion.chunker import DEFAULT_MAX_CHARS, chunk_document
from rag_mcp_server.ingestion.jsonl_store import (
    read_chunks,
    write_chunks,
    write_ingestion_report,
)
from rag_mcp_server.ingestion.loader import (
    SkippedFile,
    load_document,
    resolve_targets,
)
from rag_mcp_server.core.types import Chunk, Document
from tests.fixtures.make_pdf import empty_content_pdf, minimal_pdf


def write_sample_tree(root: Path) -> None:
    (root / "notes.txt").write_text(
        "RAG 是 Retrieval Augmented Generation。\n\n这是第二段，介绍 MCP。",
        encoding="utf-8",
    )
    (root / "docs").mkdir()
    (root / "docs" / "rag-notes.md").write_text(
        "## RAG Loop\n\n最小闭环使用关键词检索 baseline。",
        encoding="utf-8",
    )
    (root / "docs" / "ignore.pdf").write_bytes(b"%PDF-1.4 fake")
    (root / "docs" / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n fake")


class LoaderTests(unittest.TestCase):
    def test_resolve_targets_single_txt_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "notes.txt"
            target.write_text("hello", encoding="utf-8")
            files, skipped = resolve_targets(target)
            self.assertEqual(files, [target])
            self.assertEqual(skipped, [])

    def test_resolve_targets_directory_recurses_and_skips_unsupported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_sample_tree(root)
            files, skipped = resolve_targets(root)
            names = sorted(p.name for p in files)
            # .pdf is now a supported format, so ignore.pdf joins the files
            self.assertEqual(names, ["ignore.pdf", "notes.txt", "rag-notes.md"])
            self.assertEqual(len(skipped), 1)
            self.assertTrue(all(isinstance(s, SkippedFile) for s in skipped))
            self.assertEqual({s.path.suffix for s in skipped}, {".png"})
            self.assertTrue(all("unsupported" in s.reason for s in skipped))

    def test_resolve_targets_unsupported_single_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "image.png"
            target.write_bytes(b"\x89PNG\r\n\x1a\n fake")
            files, skipped = resolve_targets(target)
            self.assertEqual(files, [])
            self.assertEqual(len(skipped), 1)
            self.assertEqual(skipped[0].path, target)

    def test_load_document_uses_relative_source_path_and_slug_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "docs").mkdir()
            doc_file = root / "docs" / "rag-notes.md"
            doc_file.write_text("# title\nbody", encoding="utf-8")
            doc = load_document(doc_file, base=root)
            self.assertEqual(doc.source_path, "docs/rag-notes.md")
            # ADR 0016: document_id carries the default collection prefix
            self.assertEqual(doc.id, "default::docs-rag-notes-md")
            self.assertEqual(doc.file_type, "md")
            self.assertIn("title", doc.text)

    def test_load_document_single_file_base_is_its_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            doc_file = root / "notes.txt"
            doc_file.write_text("hello", encoding="utf-8")
            doc = load_document(doc_file, base=doc_file.parent)
            self.assertEqual(doc.source_path, "notes.txt")
            # ADR 0016: document_id carries the default collection prefix
            self.assertEqual(doc.id, "default::notes-txt")

    def test_resolve_targets_recognizes_pdf_as_supported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf_file = root / "doc.pdf"
            pdf_file.write_bytes(minimal_pdf("Hello PDF"))
            files, skipped = resolve_targets(pdf_file)
            self.assertEqual(files, [pdf_file])
            self.assertEqual(skipped, [])

    def test_load_document_pdf_dispatch_returns_document(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf_file = root / "docs" / "guide.pdf"
            pdf_file.parent.mkdir()
            pdf_file.write_bytes(minimal_pdf("Hello RAG PDF"))
            doc = load_document(pdf_file, base=root)
            self.assertEqual(doc.file_type, "pdf")
            self.assertEqual(doc.source_path, "docs/guide.pdf")
            self.assertIn("Hello RAG PDF", doc.text)

    def test_load_document_corrupt_pdf_raises_document_load_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf_file = root / "broken.pdf"
            pdf_file.write_bytes(b"%PDF-1.4 fake\n")
            with self.assertRaises(DocumentLoadError):
                load_document(pdf_file, base=root)

    def test_load_document_non_utf8_txt_raises_document_load_error(self) -> None:
        # a non-UTF-8 txt/md must be a load failure (skipped by the caller),
        # never a raw UnicodeDecodeError escaping and crashing the whole ingest
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            txt_file = root / "latin1.txt"
            txt_file.write_bytes("héllo wörld".encode("latin-1"))
            with self.assertRaises(DocumentLoadError):
                load_document(txt_file, base=root)

    def test_load_document_non_utf8_md_raises_document_load_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            md_file = root / "latin1.md"
            md_file.write_bytes("# títle\nbódy".encode("latin-1"))
            with self.assertRaises(DocumentLoadError):
                load_document(md_file, base=root)

    def test_load_document_textless_pdf_raises_document_load_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf_file = root / "scanned.pdf"
            pdf_file.write_bytes(empty_content_pdf())
            with self.assertRaises(DocumentLoadError):
                load_document(pdf_file, base=root)


class ChunkerTests(unittest.TestCase):
    def test_splits_document_by_blank_line(self) -> None:
        doc = Document(
            id="d",
            text="第一段。\n\n第二段。\n\n第三段。",
            source_path="d.md",
            file_type="md",
        )
        chunks = chunk_document(doc)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].text, "第一段。")
        self.assertEqual(chunks[2].text, "第三段。")

    def test_drops_blank_paragraphs(self) -> None:
        doc = Document(id="d", text="a\n\n\n\nb", source_path="d.txt", file_type="txt")
        chunks = chunk_document(doc)
        self.assertEqual(len(chunks), 2)
        self.assertEqual([c.text for c in chunks], ["a", "b"])

    def test_chunk_order_starts_at_zero_and_increments(self) -> None:
        doc = Document(id="d", text="a\n\nb", source_path="d.txt", file_type="txt")
        chunks = chunk_document(doc)
        self.assertEqual([c.order for c in chunks], [0, 1])
        self.assertEqual([c.chunk_id for c in chunks], ["d::0", "d::1"])

    def test_long_paragraph_split_by_max_chars(self) -> None:
        doc = Document(id="d", text="x" * 2000, source_path="d.txt", file_type="txt")
        chunks = chunk_document(doc, max_chars=800)
        self.assertEqual(len(chunks), 3)
        self.assertTrue(all(len(c.text) <= 800 for c in chunks))
        self.assertEqual("".join(c.text for c in chunks), "x" * 2000)
        self.assertEqual([c.order for c in chunks], [0, 1, 2])

    def test_chunk_carries_source_path_and_file_type_metadata(self) -> None:
        doc = Document(
            id="d",
            text="段落内容",
            source_path="notes.txt",
            file_type="txt",
        )
        chunks = chunk_document(doc)
        self.assertEqual(chunks[0].source_path, "notes.txt")
        self.assertEqual(chunks[0].metadata, {"file_type": "txt"})

    def test_default_max_chars_is_800(self) -> None:
        self.assertEqual(DEFAULT_MAX_CHARS, 800)


class JsonlStoreTests(unittest.TestCase):
    def test_write_and_read_chunks_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chunks.jsonl"
            chunks = [
                Chunk(
                    document_id="d",
                    text="第一段",
                    source_path="d.txt",
                    order=0,
                    metadata={"file_type": "txt"},
                ),
                Chunk(
                    document_id="d",
                    text="第二段",
                    source_path="d.txt",
                    order=1,
                    metadata={"file_type": "txt"},
                ),
            ]
            write_chunks(chunks, path)
            lines = path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 2)
            loaded = read_chunks(path)
            self.assertEqual([c.chunk_id for c in loaded], ["d::0", "d::1"])
            self.assertEqual(loaded[0].text, "第一段")
            self.assertEqual(loaded[0].metadata, {"file_type": "txt"})

    def test_write_ingestion_report_records_imported_skipped_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ingestion_report.json"
            report = {
                "imported": [
                    {"path": "docs/rag-notes.md", "document_id": "docs-rag-notes-md", "chunk_count": 2},
                ],
                "skipped": [{"path": "docs/ignore.pdf", "reason": "unsupported file type: .pdf"}],
                "total_files": 1,
                "total_skipped": 1,
                "total_chunks": 2,
            }
            write_ingestion_report(report, path)
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["total_chunks"], 2)
            self.assertEqual(data["total_skipped"], 1)
            self.assertEqual(data["imported"][0]["document_id"], "docs-rag-notes-md")
            self.assertIn("unsupported", data["skipped"][0]["reason"])


if __name__ == "__main__":
    unittest.main()
