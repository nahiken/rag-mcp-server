from __future__ import annotations

import re
import unittest
from pathlib import Path

from rag_mcp_server.core.errors import (
    ChunkStoreError,
    DocumentLoadError,
    QueryError,
    RagMcpError,
)
from rag_mcp_server.core.types import (
    Chunk,
    Document,
    RetrievalResult,
    make_document_id,
)


class DocumentTests(unittest.TestCase):
    def test_document_fields(self) -> None:
        doc = Document(
            id="docs-rag-notes-md",
            text="hello",
            source_path="docs/rag-notes.md",
            file_type="md",
            metadata={"title": "RAG Notes"},
        )
        self.assertEqual(doc.id, "docs-rag-notes-md")
        self.assertEqual(doc.text, "hello")
        self.assertEqual(doc.source_path, "docs/rag-notes.md")
        self.assertEqual(doc.file_type, "md")
        self.assertEqual(doc.metadata, {"title": "RAG Notes"})

    def test_document_metadata_defaults_to_empty_dict(self) -> None:
        doc = Document(
            id="x",
            text="t",
            source_path="x.md",
            file_type="md",
        )
        self.assertEqual(doc.metadata, {})


class ChunkTests(unittest.TestCase):
    def test_chunk_fields(self) -> None:
        chunk = Chunk(
            document_id="docs-rag-notes-md",
            text="paragraph",
            source_path="docs/rag-notes.md",
            order=2,
            metadata={"file_type": "md"},
        )
        self.assertEqual(chunk.document_id, "docs-rag-notes-md")
        self.assertEqual(chunk.text, "paragraph")
        self.assertEqual(chunk.source_path, "docs/rag-notes.md")
        self.assertEqual(chunk.order, 2)
        self.assertEqual(chunk.metadata, {"file_type": "md"})

    def test_chunk_id_combines_document_id_and_order(self) -> None:
        chunk = Chunk(
            document_id="docs-rag-notes-md",
            text="paragraph",
            source_path="docs/rag-notes.md",
            order=2,
        )
        self.assertEqual(chunk.chunk_id, "docs-rag-notes-md::2")

    def test_chunk_metadata_defaults_to_empty_dict(self) -> None:
        chunk = Chunk(
            document_id="x",
            text="t",
            source_path="x.md",
            order=0,
        )
        self.assertEqual(chunk.metadata, {})


class RetrievalResultTests(unittest.TestCase):
    def test_retrieval_result_fields(self) -> None:
        result = RetrievalResult(
            chunk_id="docs-rag-notes-md::0",
            source_path="docs/rag-notes.md",
            score=2.0,
            text="paragraph",
        )
        self.assertEqual(result.chunk_id, "docs-rag-notes-md::0")
        self.assertEqual(result.source_path, "docs/rag-notes.md")
        self.assertEqual(result.score, 2.0)
        self.assertEqual(result.text, "paragraph")


class DocumentIdTests(unittest.TestCase):
    def test_make_document_id_slugifies_relative_path(self) -> None:
        self.assertEqual(make_document_id(Path("docs/rag-notes.md")), "default::docs-rag-notes-md")

    def test_make_document_id_handles_windows_separator(self) -> None:
        self.assertEqual(
            make_document_id(Path("docs") / "rag-notes.md"),
            "default::docs-rag-notes-md",
        )

    def test_same_filename_in_different_dirs_do_not_conflict(self) -> None:
        self.assertEqual(make_document_id(Path("a/notes.md")), "default::a-notes-md")
        self.assertEqual(make_document_id(Path("b/notes.md")), "default::b-notes-md")
        self.assertNotEqual(
            make_document_id(Path("a/notes.md")),
            make_document_id(Path("b/notes.md")),
        )

    def test_chinese_filenames_keep_distinct_ids(self) -> None:
        r"""回归（修复）：中文/非 ASCII 文件名必须保留字面字符，不能塌缩成扩展名。

        旧的 ``[^a-zA-Z0-9]+`` slug 会把「报告.md」和「总结.md」都压成 ``md``，
        chunk_id 碰撞导致后写入的文档覆盖先写入的（数据丢失）。
        ``[\W_]+`` 保留 Unicode 单词字符（中文），只折叠分隔符。
        """
        self.assertEqual(make_document_id(Path("报告.md")), "default::报告-md")
        self.assertEqual(make_document_id(Path("总结.md")), "default::总结-md")
        self.assertNotEqual(make_document_id(Path("报告.md")), make_document_id(Path("总结.md")))
        self.assertEqual(make_document_id(Path("中文文档.txt")), "default::中文文档-txt")
        self.assertEqual(make_document_id(Path("data/中文文档.md")), "default::data-中文文档-md")
        # 不同目录下的同名中文文件仍不冲突
        self.assertNotEqual(
            make_document_id(Path("a/报告.md")),
            make_document_id(Path("b/报告.md")),
        )

    def test_document_id_is_readable_and_not_uuid_or_hash(self) -> None:
        value = make_document_id(Path("docs/rag-notes.md"))
        # readable: lowercase ascii slug of path words joined by "-"
        self.assertTrue(value.isascii())
        # ADR 0016: document_id carries the default collection prefix
        self.assertEqual(value, "default::docs-rag-notes-md")
        self.assertNotIn("_", value)
        self.assertNotIn(" ", value)
        # not a uuid: no 8-4-4-4-12 hex pattern
        self.assertFalse(bool(re.match(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", value)))


class ErrorHierarchyTests(unittest.TestCase):
    def test_all_custom_errors_are_rag_mcp_errors(self) -> None:
        self.assertTrue(issubclass(DocumentLoadError, RagMcpError))
        self.assertTrue(issubclass(ChunkStoreError, RagMcpError))
        self.assertTrue(issubclass(QueryError, RagMcpError))

    def test_rag_mcp_error_is_exception(self) -> None:
        self.assertTrue(issubclass(RagMcpError, Exception))


if __name__ == "__main__":
    unittest.main()
