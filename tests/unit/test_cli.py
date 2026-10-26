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


def write_offline_config(tmp: Path, index: Path) -> Path:
    """Write a temp config/settings.yaml that keeps the offline run isolated.

    Retrieval strategy is forced to ``sparse`` (pure BM25 over the vector
    store, no embedding needed) and the Chroma persist_dir points under the
    temp directory so a separate query process reads the same store without
    touching the repo. Returns the temp config dir.
    """
    config_dir = tmp / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "settings.yaml").write_text(
        f"vector_store:\n"
        f"  persist_dir: {tmp / 'chroma'}\n"
        f"retrieval:\n"
        f"  strategy: sparse\n",
        encoding="utf-8",
    )
    return config_dir


def run_cli(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(SRC) + (os.pathsep + existing if existing else "")
    # Windows 管道默认用 GBK，这里显式强制子进程用 UTF-8，保证中文往返一致
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


def make_fixture(root: Path) -> None:
    # All importable docs live under docs/ so ingest scans only that subdir —
    # config/, chroma/ and the index live at root and must never be scanned.
    (root / "docs").mkdir()
    (root / "docs" / "notes.txt").write_text(
        "RAG 是检索增强生成，最小闭环使用关键词检索。\n\nMCP 是模型上下文协议，用于连接工具。",
        encoding="utf-8",
    )
    (root / "docs" / "guide.md").write_text(
        "## 检索\n\n关键词检索返回 top 3 结果并附带引用来源。",
        encoding="utf-8",
    )
    (root / "docs" / "broken.pdf").write_bytes(b"%PDF-1.4 fake\n")


class CliModuleEntryTests(unittest.TestCase):
    def test_python_m_rag_mcp_server_prints_help(self) -> None:
        completed = run_cli("-m", "rag_mcp_server", "--help", cwd=ROOT)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("ingest", completed.stdout)
        self.assertIn("query", completed.stdout)


class CliIngestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        make_fixture(self.root)
        write_offline_config(self.root, self.root / "out")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_module_ingest_writes_artifacts_and_human_readable_output(self) -> None:
        index = self.root / "out"
        completed = run_cli(
            "-m", "rag_mcp_server", "ingest", str(self.root / "docs"),
            "--index-dir", str(index), "--embedding-provider", "fake",
            cwd=self.root,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("已导入", completed.stdout)
        self.assertIn("跳过", completed.stdout)
        # ADR 0016: evidence lives under index_dir/{collection}/ (default default)
        self.assertTrue((index / "default" / "chunks.jsonl").is_file())
        self.assertTrue((index / "default" / "ingestion_report.json").is_file())

    def test_module_ingest_json_output(self) -> None:
        index = self.root / "out"
        completed = run_cli(
            "-m", "rag_mcp_server", "ingest", str(self.root / "docs"),
            "--index-dir", str(index), "--embedding-provider", "fake", "--json",
            cwd=self.root,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        data = json.loads(completed.stdout)
        self.assertEqual(data["total_files"], 2)
        self.assertEqual(data["total_skipped"], 1)
        self.assertGreater(data["total_chunks"], 0)

    def test_module_ingest_force_flag_parses_and_runs(self) -> None:
        # P10.5 T2: `ingest --force` must parse (no argparse error) and run a
        # full ingest. Force bypasses the file-level integrity skip; here there
        # is no checker so it behaves like a normal ingest but the flag must
        # not break the command line.
        index = self.root / "out"
        completed = run_cli(
            "-m", "rag_mcp_server", "ingest", str(self.root / "docs"),
            "--index-dir", str(index), "--embedding-provider", "fake", "--json",
            "--force",
            cwd=self.root,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        data = json.loads(completed.stdout)
        self.assertEqual(data["total_files"], 2)
        self.assertEqual(data["total_skipped"], 1)
        self.assertGreater(data["total_chunks"], 0)

    def test_module_ingest_unknown_provider_is_user_error_not_traceback(self) -> None:
        # S1 review fix: an unknown --embedding-provider fails fast with a
        # ValueError in the factory; the CLI must surface it as a readable
        # user error (CONTEXT 用户错误), never a Python traceback.
        index = self.root / "out"
        completed = run_cli(
            "-m", "rag_mcp_server", "ingest", str(self.root / "docs"),
            "--index-dir", str(index), "--embedding-provider", "bogus",
            cwd=self.root,
        )
        self.assertNotEqual(completed.returncode, 0, completed.stdout)
        self.assertIn("错误：", completed.stderr)
        self.assertIn("bogus", completed.stderr)
        self.assertNotIn("Traceback", completed.stderr)
        self.assertNotIn("Traceback", completed.stdout)


class CliQueryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        make_fixture(self.root)
        write_offline_config(self.root, self.root / "out")
        self.index = self.root / "out"
        run_cli("-m", "rag_mcp_server", "ingest", str(self.root / "docs"), "--index-dir", str(self.index), "--embedding-provider", "fake", cwd=self.root)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_module_query_human_readable(self) -> None:
        completed = run_cli(
            "-m", "rag_mcp_server", "query", "检索",
            "--index-dir", str(self.index),
            cwd=self.root,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("检索", completed.stdout)
        self.assertIn("guide.md", completed.stdout)

    def test_script_query_json_output_traces_citations(self) -> None:
        completed = run_cli(
            str(ROOT / "scripts" / "query.py"), "检索",
            "--index-dir", str(self.index), "--json",
            cwd=self.root,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        data = json.loads(completed.stdout)
        self.assertIn("results", data)
        self.assertGreaterEqual(len(data["results"]), 1)
        for result in data["results"]:
            self.assertIn("chunk_id", result)
            self.assertIn("source_path", result)
            self.assertIn("score", result)
            self.assertIn("sources", result)
            # 引用来源相关：命中文本包含查询词
            self.assertIn("检索", result["text"])

    def test_query_no_result_is_not_an_error(self) -> None:
        completed = run_cli(
            "-m", "rag_mcp_server", "query", "qxzvw",
            "--index-dir", str(self.index),
            cwd=self.root,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("没有找到匹配", completed.stdout)

    def test_query_no_result_json_is_empty_results(self) -> None:
        completed = run_cli(
            "-m", "rag_mcp_server", "query", "qxzvw",
            "--index-dir", str(self.index), "--json",
            cwd=self.root,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        data = json.loads(completed.stdout)
        self.assertEqual(data["results"], [])

    def test_query_invalid_top_k_is_user_error_not_traceback(self) -> None:
        # P1 review fix: negative/zero top_k must be a friendly error, not a
        # bare Python traceback
        completed = run_cli(
            "-m", "rag_mcp_server", "query", "检索",
            "--index-dir", str(self.index), "--top-k", "-1",
            cwd=self.root,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertNotIn("Traceback", completed.stderr)
        self.assertIn("错误", completed.stderr)

    def test_query_default_top_k_reads_config(self) -> None:
        """CLI 不传 --top-k 时，结果数由配置 retrieval.top_k 决定（死配置接线）。"""
        # 现有 fixture 有 2 个含"检索"的 chunk；配置 retrieval.top_k=1 应只回 1 条
        config_dir = self.root / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "settings.yaml").write_text(
            f"vector_store:\n  persist_dir: {self.root / 'chroma'}\n"
            f"retrieval:\n  strategy: sparse\n  top_k: 1\n",
            encoding="utf-8",
        )
        completed = run_cli(
            "-m", "rag_mcp_server", "query", "检索",
            "--index-dir", str(self.index), "--json",
            cwd=self.root,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        data = json.loads(completed.stdout)
        self.assertEqual(len(data["results"]), 1, f"配置 top_k=1 应截断，实际 {len(data['results'])} 条")

    def test_top_k_flag_limits_results(self) -> None:
        completed = run_cli(
            "-m", "rag_mcp_server", "query", "检索",
            "--index-dir", str(self.index), "--json", "--top-k", "1",
            cwd=self.root,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        data = json.loads(completed.stdout)
        self.assertEqual(len(data["results"]), 1)


class CliScriptEntryTests(unittest.TestCase):
    def test_script_ingest_and_query_share_application_service(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_fixture(root)
            write_offline_config(root, root / "out")
            index = root / "out"
            ingest_run = run_cli(
                str(ROOT / "scripts" / "ingest.py"), str(root / "docs"),
                "--index-dir", str(index), "--embedding-provider", "fake", "--json",
                cwd=root,
            )
            self.assertEqual(ingest_run.returncode, 0, ingest_run.stderr)
            data = json.loads(ingest_run.stdout)
            self.assertEqual(data["total_files"], 2)

            query_run = run_cli(
                str(ROOT / "scripts" / "query.py"), "MCP",
                "--index-dir", str(index), "--json",
                cwd=root,
            )
            self.assertEqual(query_run.returncode, 0, query_run.stderr)
            qdata = json.loads(query_run.stdout)
            self.assertGreaterEqual(len(qdata["results"]), 1)
            for result in qdata["results"]:
                self.assertIn("MCP", result["text"])


class CliErrorTests(unittest.TestCase):
    def test_ingest_nonexistent_path_prints_user_error_and_nonzero_exit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_offline_config(root, root / "out")
            completed = run_cli(
                "-m", "rag_mcp_server", "ingest", str(root / "missing"),
                "--embedding-provider", "fake",  # offline: never touch data/
                cwd=root,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertNotIn("Traceback", completed.stderr)

    def test_query_missing_store_is_not_an_error(self) -> None:
        """Empty store (nothing ingested) returns no results, not an exception.

        The vector store is the source of truth; an unpopulated store is a
        legitimate empty result (spec: no-result returns empty list).
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = root / "out"
            write_offline_config(root, index)
            completed = run_cli(
                "-m", "rag_mcp_server", "query", "检索",
                "--index-dir", str(index),
                cwd=root,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("没有找到匹配", completed.stdout)


if __name__ == "__main__":
    unittest.main()
