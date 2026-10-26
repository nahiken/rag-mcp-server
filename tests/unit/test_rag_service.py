from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from rag_mcp_server.application.rag_service import ingest, query
from rag_mcp_server.ingestion.pipeline import IngestPipeline
from rag_mcp_server.libs.embedding.fake import FakeEmbedding
from rag_mcp_server.libs.vector_store.chroma_store import ChromaVectorStore


def _fake_pipeline(index_dir: Path, collection: str = "test_chunks") -> IngestPipeline:
    """Offline pipeline: FakeEmbedding + temp Chroma, for unit tests.

    ``collection`` defaults to "test_chunks" but is injectable so tests that
    pre-write shared content into the in-process EphemeralClient (Chroma's
    per-collection shared memory across instances) get an isolated collection
    and don't trip the known idempotency/pollution flake in sibling tests.
    The same collection flows to the store and to the pipeline so the evidence
    directory (``index_dir/{collection}/``, ADR 0016) matches the store.
    """
    embedding = FakeEmbedding(dim=4)
    store = ChromaVectorStore(persist_dir=None, collection=collection)
    return IngestPipeline(embedding=embedding, store=store, index_dir=index_dir, collection=collection)


class RagServiceIngestTests(unittest.TestCase):
    def test_ingest_writes_chunks_jsonl_and_skipped_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "docs").mkdir()
            (root / "notes.txt").write_text("RAG 关键词检索。\n\nMCP 工具。", encoding="utf-8")
            (root / "docs" / "guide.md").write_text("# Guide\n正文。", encoding="utf-8")
            (root / "docs" / "broken.pdf").write_bytes(b"%PDF-1.4 fake\n")
            index_dir = root / "out"

            summary = ingest(root, base=root, index_dir=index_dir, pipeline=_fake_pipeline(index_dir), trace_dir=root)

            # txt + md are imported; the corrupt pdf is skipped (not unsupported)
            self.assertEqual(summary.total_files, 2)
            self.assertEqual(summary.total_skipped, 1)
            self.assertEqual(summary.total_chunks, 3)

            # ADR 0016: evidence lives under index_dir/{collection}/
            chunks_file = index_dir / "test_chunks" / "chunks.jsonl"
            report_file = index_dir / "test_chunks" / "ingestion_report.json"
            self.assertTrue(chunks_file.is_file())
            self.assertTrue(report_file.is_file())
            lines = chunks_file.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 3)
            self.assertIn('"source_path": "docs/guide.md"', lines[0])

            report = report_file.read_text(encoding="utf-8")
            self.assertIn('"total_skipped": 1', report)
            self.assertIn('"reason": "cannot extract text', report)
            self.assertIn('"path": "docs/broken.pdf"', report)

    def test_ingest_single_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            notes = root / "notes.txt"
            notes.write_text("单文件导入。", encoding="utf-8")
            index_dir = root / "out"
            summary = ingest(notes, index_dir=index_dir, pipeline=_fake_pipeline(index_dir), trace_dir=root)
            self.assertEqual(summary.total_files, 1)
            self.assertEqual(summary.total_chunks, 1)
            # ADR 0016: document_id is prefixed with the collection
            self.assertEqual(summary.imported[0]["document_id"], "test_chunks::notes-txt")

    def test_ingest_forwards_on_progress_to_pipeline(self) -> None:
        # P10.5 ticket 06: rag_service.ingest must forward on_progress to the
        # pipeline so the Dashboard can consume file-level progress from the
        # service seam. Stage names mirror trace; total = files to process.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "notes.txt").write_text("第一段。\n\n第二段。", encoding="utf-8")
            index_dir = root / "out"
            events: list[tuple[str, int, int]] = []

            ingest(
                root,
                base=root,
                index_dir=index_dir,
                pipeline=_fake_pipeline(index_dir),
                trace_dir=root,
                on_progress=lambda s, c, t: events.append((s, c, t)),
            )

            self.assertEqual(
                [s for s, _, _ in events],
                ["load", "split", "embed", "upsert"],
            )
            self.assertEqual({c for _, c, _ in events}, {1})
            self.assertEqual({t for _, _, t in events}, {1})

    def test_ingest_forwards_integrity_for_incremental_skip(self) -> None:
        # P10.5 T1: rag_service.ingest forwards the FileIntegrityChecker so the
        # service seam can opt into file-level incremental skip. A second ingest
        # of unchanged files is skipped, not re-embedded.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docs = root / "docs"
            docs.mkdir()
            (docs / "notes.txt").write_text("第一段。\n\n第二段。", encoding="utf-8")
            index_dir = root / "out"
            from rag_mcp_server.libs.loader.file_integrity import FileIntegrityChecker

            integrity = FileIntegrityChecker(db_path=root / "db" / "history.db")
            pipeline = _fake_pipeline(index_dir)

            first = ingest(docs, base=docs, index_dir=index_dir, pipeline=pipeline, trace_dir=root, integrity=integrity)
            self.assertEqual(first.total_files, 1)
            self.assertEqual(first.total_chunks, 2)

            second = ingest(docs, base=docs, index_dir=index_dir, pipeline=pipeline, trace_dir=root, integrity=integrity)
            self.assertEqual(second.total_files, 0)
            self.assertEqual(second.total_skipped, 1)
            self.assertEqual(second.skipped[0]["reason"], "unchanged (integrity)")

    def test_ingest_force_bypasses_integrity_skip(self) -> None:
        # P10.5 T2: rag_service.ingest forwards `force` — a forced re-ingest
        # of an unchanged file re-runs the full chain (bypassing should_skip)
        # yet does not duplicate chunks (content_hash idempotency still holds).
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docs = root / "docs"
            docs.mkdir()
            (docs / "notes.txt").write_text("第一段。\n\n第二段。", encoding="utf-8")
            index_dir = root / "out"
            from rag_mcp_server.libs.loader.file_integrity import FileIntegrityChecker

            integrity = FileIntegrityChecker(db_path=root / "db" / "history.db")
            # isolated collection: this test pre-writes identical content into
            # the shared EphemeralClient, which would trip the idempotency /
            # pollution flake in test_ingest_forwards_integrity_for_incremental_skip
            pipeline = _fake_pipeline(index_dir, collection="test_chunks_force")

            first = ingest(docs, base=docs, index_dir=index_dir, pipeline=pipeline, trace_dir=root, integrity=integrity)
            self.assertEqual(first.total_files, 1)
            self.assertEqual(first.total_chunks, 2)

            # without force: unchanged file is integrity-skipped
            skipped = ingest(docs, base=docs, index_dir=index_dir, pipeline=pipeline, trace_dir=root, integrity=integrity)
            self.assertEqual(skipped.total_files, 0)
            self.assertEqual(skipped.total_skipped, 1)

            # with force: full chain re-runs, but content_hash prevents dups
            forced = ingest(docs, base=docs, index_dir=index_dir, pipeline=pipeline, trace_dir=root, integrity=integrity, force=True)
            self.assertEqual(forced.total_files, 1)
            self.assertEqual(forced.total_chunks, 0)  # no dup chunks
            records = pipeline.store.get_by_ids(["test_chunks_force::notes-txt::0", "test_chunks_force::notes-txt::1"])
            self.assertEqual(len(records), 2)

            # force must still refresh the integrity record (spec F1): after a
            # forced re-ingest, a non-force ingest of the unchanged file is
            # integrity-skipped again — external behaviour proves mark_success
            # re-recorded the file (if it had been left stale/failed, the file
            # would be re-ingested, total_files == 1)
            after_force = ingest(docs, base=docs, index_dir=index_dir, pipeline=pipeline, trace_dir=root, integrity=integrity)
            self.assertEqual(after_force.total_files, 0)
            self.assertEqual(after_force.total_skipped, 1)
            self.assertEqual(after_force.skipped[0]["reason"], "unchanged (integrity)")


class RagServiceQueryTests(unittest.TestCase):
    def test_query_results_trace_to_real_chunks_and_disk(self) -> None:
        """引用来源存在且相关：query 命中必须能追溯回 store 中的 chunk，且源文件在磁盘上。

        查询与摄取共享同一个 pipeline（注入缝），store 即检索源。
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "docs").mkdir()
            (root / "docs" / "rag-notes.md").write_text(
                "RAG 最小闭环使用关键词检索，检索结果按 score 排序。",
                encoding="utf-8",
            )
            (root / "notes.txt").write_text(
                "MCP 是 Model Context Protocol，用于连接工具。",
                encoding="utf-8",
            )
            index_dir = root / "out"
            pipeline = _fake_pipeline(index_dir)
            ingest(root, base=root, index_dir=index_dir, pipeline=pipeline, trace_dir=root)

            results = query("检索", index_dir=index_dir, top_k=3, pipeline=pipeline, trace_dir=root)
            self.assertGreaterEqual(len(results), 1)
            result = results[0]
            # ADR 0016: chunk_id carries the collection prefix
            self.assertEqual(result.chunk_id, "test_chunks::docs-rag-notes-md::0")
            # source_path 必须对应磁盘上真实存在的文件
            self.assertTrue((root / result.source_path).is_file())
            # 命中文本必须与查询相关
            self.assertIn("检索", result.text)


class DefaultPipelineAssemblyTests(unittest.TestCase):
    def test_default_pipeline_builds_file_integrity_checker(self) -> None:
        """装配 seam（P10.8 收口 spec 决策 2）：真实装配路径默认注入 FileIntegrityChecker。

        从 settings 装配的 pipeline 必须自带 checker（无 settings 开关，--force 即逃生口），
        使重复摄取未变更文件被跳过。桩掉 embedding/vector_store factory 以隔离真实 IO，
        只断言装配决策本身。
        """
        from unittest.mock import patch

        from rag_mcp_server.application.rag_service import _default_pipeline
        from rag_mcp_server.libs.loader.file_integrity import FileIntegrityChecker

        dummy_embedding = object()
        dummy_store = object()
        with tempfile.TemporaryDirectory() as tmp:
            # FileIntegrityChecker() 用默认相对路径 data/db/，须在临时 cwd 下运行
            # 以免在仓库根写库；embedding/vector_store 桩掉隔离真实 IO。
            old_cwd = os.getcwd()
            os.chdir(Path(tmp))
            try:
                with patch(
                    "rag_mcp_server.libs.embedding.factory.create_embedding",
                    return_value=dummy_embedding,
                ), patch(
                    "rag_mcp_server.libs.vector_store.factory.create_vector_store",
                    return_value=dummy_store,
                ):
                    pipe = _default_pipeline(Path("index"))
            finally:
                os.chdir(old_cwd)
        self.assertIsInstance(pipe.integrity, FileIntegrityChecker)
        self.assertIs(pipe.store, dummy_store)
        self.assertIs(pipe.embedding, dummy_embedding)


if __name__ == "__main__":
    unittest.main()
