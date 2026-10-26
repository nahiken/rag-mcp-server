"""Phase 9 end-to-end verification (task 13): PDF ingest -> sparse query -> delete.

Offline, dual-process (CLI subprocesses share a temp persist_dir Chroma via a
temp config, mirroring test_minimum_rag_loop.py). Chain:
1. ingest a real fixture PDF (minimal_pdf, ASCII text) via the CLI
2. sparse query hits the PDF chunk (citations traceable)
3. DocumentManager.delete removes the doc (only Chroma; sparse auto-consistent)
4. sparse query no longer returns the deleted document

verify.py already runs pytest -m not integration on tests/e2e, so this module
is included automatically. No changes to verify.py needed.
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

from rag_mcp_server.document_manager.manager import DocumentManager  # noqa: E402
from rag_mcp_server.libs.vector_store.chroma_store import ChromaVectorStore  # noqa: E402
from tests.fixtures.make_pdf import minimal_pdf  # noqa: E402


def write_offline_config(tmp: Path) -> Path:
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


def _make_pdf_fixture(root: Path) -> None:
    """A real fixture PDF (minimal_pdf) under docs/ so ingest scans it.

    minimal_pdf renders with a Helvetica Type1 font which cannot encode CJK,
    so the PDF text is ASCII; the sparse query uses an English term (the
    bilingual CN+EN path is covered by test_minimum_rag_loop).
    """
    (root / "docs").mkdir(parents=True, exist_ok=True)
    (root / "docs" / "guide.pdf").write_bytes(minimal_pdf("RAG PDF semantic search test"))


class Phase9E2ETests(unittest.TestCase):
    def test_pdf_ingest_sparse_query_delete_full_chain(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            _make_pdf_fixture(root)
            index = root / "index"
            write_offline_config(root)

            # 1. ingest the PDF via CLI (offline: fake embedding, sparse store)
            ingest_run = run_cli(
                "-m", "rag_mcp_server", "ingest", str(root / "docs"),
                "--index-dir", str(index), "--embedding-provider", "fake", "--json",
                cwd=root,
            )
            self.assertEqual(ingest_run.returncode, 0, ingest_run.stderr)
            ingest_data = json.loads(ingest_run.stdout)
            self.assertEqual(ingest_data["total_files"], 1, "PDF 应被摄取")
            self.assertGreater(ingest_data["total_chunks"], 0)

            chunks_file = index / "default" / "chunks.jsonl"
            report_file = index / "default" / "ingestion_report.json"
            self.assertTrue(chunks_file.is_file())
            self.assertTrue(report_file.is_file())

            # 2. sparse query hits the PDF chunk, citation traceable
            query_run = run_cli(
                "-m", "rag_mcp_server", "query", "search",
                "--index-dir", str(index), "--json",
                cwd=root,
            )
            self.assertEqual(query_run.returncode, 0, query_run.stderr)
            query_data = json.loads(query_run.stdout)
            self.assertGreaterEqual(len(query_data["results"]), 1, "sparse 查询应命中 PDF")
            guide_hits = [r for r in query_data["results"] if "guide" in r["chunk_id"]]
            self.assertTrue(guide_hits, f"应命中 PDF chunk，实际 {[r['chunk_id'] for r in query_data['results']]}")

            # 3. DocumentManager.delete removes the doc (only Chroma)
            # CLI ingest uses the settings default collection (default, from
            # this test's temp config); the store read must match.
            store = ChromaVectorStore(persist_dir=str(root / "chroma"), collection="default")
            manager = DocumentManager(store=store)
            docs = manager.list()
            self.assertEqual(len(docs), 1)
            doc_id = docs[0]["document_id"]
            removed = manager.delete(doc_id)
            self.assertGreater(removed, 0)

            # 4. sparse query no longer returns the deleted document
            after = run_cli(
                "-m", "rag_mcp_server", "query", "search",
                "--index-dir", str(index), "--json",
                cwd=root,
            )
            self.assertEqual(after.returncode, 0, after.stderr)
            after_data = json.loads(after.stdout)
            # chunk_id is "<document_id>::<order>"; assert no result belongs to the doc
            self.assertFalse(
                any(doc_id in r["chunk_id"] for r in after_data["results"]),
                f"删除后该文档不应再出现，实际 {[r['chunk_id'] for r in after_data['results']]}",
            )


if __name__ == "__main__":
    unittest.main()
