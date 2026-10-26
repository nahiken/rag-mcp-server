r"""E2E regression for Bug 1: Chinese filenames must keep distinct document ids.

Dual-process CLI chain (same style as test_minimum_rag_loop.py): a temp config
+ fake embedding + sparse store. The bug: ``make_document_id`` collapsed all
CJK to ``-`` so ``报告.md`` and ``总结.md`` both became ``document_id=md``;
their chunk ids collided (``md::0``) and the second file's chunk silently
overwrote the first in Chroma — one document was lost from the index.

After the fix (``[\W_]+`` keeps Unicode word chars), the two files must be
indexed side by side, their chunks must not overwrite each other, and a query
must be able to distinguish them.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


def write_offline_config(tmp: Path) -> Path:
    """Temp config: sparse retrieval (no embedding needed), temp Chroma dir."""
    config_dir = tmp / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "settings.yaml").write_text(
        f"vector_store:\n"
        f"  persist_dir: {tmp / 'chroma'}\n"
        f"  collection: default\n"
        f"retrieval:\n"
        f"  strategy: sparse\n",
        encoding="utf-8",
    )
    return config_dir


def run_cli(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(SRC) + (os.pathsep + existing if existing else "")
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


def _make_chinese_fixture(root: Path) -> None:
    """Two different Chinese filenames whose old slug collapsed to the same id."""
    (root / "docs").mkdir(parents=True, exist_ok=True)
    (root / "docs" / "报告.md").write_text(
        "报告是关于季度销售数据的分析，包含收入与支出。",
        encoding="utf-8",
    )
    (root / "docs" / "总结.md").write_text(
        "总结是本季度项目进度的回顾，包含里程碑与风险。",
        encoding="utf-8",
    )


class ChineseFilenamesE2ETests(unittest.TestCase):
    def test_two_chinese_files_stay_distinct_and_both_retrievable(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            _make_chinese_fixture(root)
            index = root / "index"
            write_offline_config(root)

            # 1. ingest: both Chinese files must import, none lost to id collision
            ingest_run = run_cli(
                "-m", "rag_mcp_server", "ingest", str(root / "docs"),
                "--index-dir", str(index), "--embedding-provider", "fake", "--json",
                cwd=root,
            )
            self.assertEqual(ingest_run.returncode, 0, ingest_run.stderr)
            ingest_data = json.loads(ingest_run.stdout)
            self.assertEqual(ingest_data["total_files"], 2, "两个中文文件都应导入")
            self.assertEqual(ingest_data["total_skipped"], 0)
            self.assertEqual(ingest_data["total_chunks"], 2)

            # document ids must be distinct (the actual bug: both were 'md')
            doc_ids = {item["document_id"] for item in ingest_data["imported"]}
            self.assertEqual(len(doc_ids), 2, f"两个中文文件 document_id 必须不同，实际 {doc_ids}")
            self.assertIn("default::报告-md", doc_ids)
            self.assertIn("default::总结-md", doc_ids)

            # chunks.jsonl evidence: both chunks present, no overwrite
            chunks_file = index / "default" / "chunks.jsonl"
            self.assertTrue(chunks_file.is_file())
            chunk_ids = [
                json.loads(line)["chunk_id"]
                for line in chunks_file.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(chunk_ids), 2)
            self.assertEqual(len(set(chunk_ids)), 2, f"chunk_id 不应碰撞，实际 {chunk_ids}")
            self.assertIn("default::报告-md::0", chunk_ids)
            self.assertIn("default::总结-md::0", chunk_ids)

            # 2. a query for one Chinese term must hit that document, not the other
            query_run = run_cli(
                "-m", "rag_mcp_server", "query", "销售",
                "--index-dir", str(index), "--json",
                cwd=root,
            )
            self.assertEqual(query_run.returncode, 0, query_run.stderr)
            query_data = json.loads(query_run.stdout)
            self.assertGreaterEqual(len(query_data["results"]), 1, "查询应命中报告")
            # 销售 only appears in 报告.md -> the hit must be the 报告 chunk
            hit_ids = [r["chunk_id"] for r in query_data["results"]]
            self.assertIn("default::报告-md::0", hit_ids, f"查询「销售」应命中报告 chunk，实际 {hit_ids}")
            # and it must not have swallowed the other document's text
            self.assertIn("销售", query_data["results"][0]["text"])

    def test_chinese_filename_source_path_roundtrips_utf8(self) -> None:
        # the CLI JSON output must carry the Chinese relative path un-corrupted
        # (subprocess UTF-8 pipes + _ensure_utf8_stdio) so citations trace back
        # to the real file on disk
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            _make_chinese_fixture(root)
            index = root / "index"
            write_offline_config(root)

            ingest_run = run_cli(
                "-m", "rag_mcp_server", "ingest", str(root / "docs"),
                "--index-dir", str(index), "--embedding-provider", "fake", "--json",
                cwd=root,
            )
            self.assertEqual(ingest_run.returncode, 0, ingest_run.stderr)
            imported = json.loads(ingest_run.stdout)["imported"]
            for item in imported:
                self.assertTrue((root / "docs" / item["path"]).is_file(), item["path"])

    def test_partial_update_reingest_integrity_and_chunk_count(self) -> None:
        """Bug 1 + Bug 2 combined at the real-storage level: a Chinese file that
        grows between ingests must (a) keep its distinct document id, (b) not be
        integrity-skipped (content changed), (c) record the *added* chunk count
        in the integrity DB, and (d) keep both chunks in the store."""
        import hashlib
        import sqlite3

        from rag_mcp_server.ingestion.pipeline import IngestPipeline
        from rag_mcp_server.libs.embedding.fake import FakeEmbedding
        from rag_mcp_server.libs.loader.file_integrity import FileIntegrityChecker
        from rag_mcp_server.libs.vector_store.chroma_store import ChromaVectorStore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            docs = root / "docs"
            docs.mkdir()
            source = docs / "报告.md"
            source.write_text("报告是关于季度销售数据的分析。", encoding="utf-8")

            embedding = FakeEmbedding(dim=4)
            store = ChromaVectorStore(persist_dir=str(root / "chroma"), collection="default")
            checker = FileIntegrityChecker(db_path=root / "db" / "ingestion_history.db")
            pipe = IngestPipeline(
                embedding=embedding, store=store, index_dir=root / "index",
                integrity=checker, collection="default",
            )

            first = pipe.ingest(docs, base=docs)
            self.assertEqual(first.total_files, 1)
            self.assertEqual(first.total_chunks, 1)
            self.assertEqual(first.imported[0]["document_id"], "default::报告-md")  # Bug 1: distinct id

            # partial update: the file grows by one paragraph -> new chunk only
            source.write_text("报告是关于季度销售数据的分析。\n\n支出同比上升百分之二十。", encoding="utf-8")
            second = pipe.ingest(docs, base=docs)
            self.assertEqual(second.total_files, 1)
            self.assertEqual(second.total_chunks, 1)  # only the new paragraph added
            # Bug 1: id stays distinct even after re-ingest; store keeps both chunks
            records = store.get_by_ids(["default::报告-md::0", "default::报告-md::1"])
            self.assertEqual(len(records), 2)
            self.assertEqual(records[1].text, "支出同比上升百分之二十。")

            # Bug 2: integrity record must hold the *added* count (1), not the
            # chunker's doc_chunks count (2)
            # file_hash 含文件身份（relative path）+ 内容（F1 修复后的语义）
            relative = "报告.md"
            file_hash = hashlib.sha256(
                relative.encode("utf-8") + b"\0" + source.read_bytes()
            ).hexdigest()
            conn = sqlite3.connect(str(root / "db" / "ingestion_history.db"))
            try:
                row = conn.execute(
                    "SELECT chunk_count, status FROM file_integrity WHERE file_hash = ?",
                    (file_hash,),
                ).fetchone()
            finally:
                conn.close()
            self.assertIsNotNone(row)
            self.assertEqual(row[0], 1, "integrity chunk_count 应记录实际新增数")
            self.assertEqual(row[1], "success")

            # unchanged re-ingest -> integrity-skipped (performance layer)
            third = pipe.ingest(docs, base=docs)
            self.assertEqual(third.total_files, 0)
            self.assertEqual(third.total_skipped, 1)
            self.assertIn("unchanged (integrity)", third.skipped[0]["reason"])


if __name__ == "__main__":
    unittest.main()
