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
    """Write a temp config/settings.yaml for the offline dual-process run.

    Retrieval strategy is forced to ``sparse`` (pure BM25 over the vector
    store, no embedding needed) and Chroma persist_dir points under the temp
    directory so the query subprocess reads the same store the ingest
    subprocess wrote. Both run with cwd=tmp so config resolves here.
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


def make_bilingual_fixture(root: Path) -> None:
    """双语 fixture：中文学习内容 + 英文 RAG/MCP 术语。

    所有可导入文档都在 docs/ 下，ingest 只扫描该子目录——config/、chroma/
    与 index 都在 root 下，绝不能被当作源文件扫描。
    """
    (root / "docs").mkdir()
    (root / "docs" / "notes.txt").write_text(
        "RAG 是 Retrieval Augmented Generation 的缩写，最小闭环先用关键词检索。\n\n"
        "MCP 是 Model Context Protocol，用于连接工具与数据。",
        encoding="utf-8",
    )
    (root / "docs" / "rag-notes.md").write_text(
        "## RAG Loop\n\n"
        "最小闭环：txt/md 导入，段落切 chunk，词频检索返回 top 3 结果，并附带引用来源。\n\n"
        "## MCP\n\n后续通过 MCP 测试客户端调用 query_knowledge_hub。",
        encoding="utf-8",
    )
    (root / "docs" / "broken.pdf").write_bytes(b"%PDF-1.4 fake\n")


class MinimumRagLoopE2ETests(unittest.TestCase):
    def test_full_loop_ingest_report_and_queried_citations_trace_to_disk(self) -> None:
        """从双语 fixture 完整跑通 ingest -> report -> query。

        关键验证点：query 命中的 chunk 必须能追溯回磁盘上真实存在的源文件，
        且命中文本与查询相关（引用来源存在且相关），而不是只看是否生成了输出。
        双进程共享同一个持久 Chroma（temp config persist_dir），策略 sparse 离线。
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_bilingual_fixture(root)
            index = root / "index"
            write_offline_config(root)

            # 1. ingest via module entry (offline: fake embedding, sparse store)
            ingest_run = run_cli(
                "-m", "rag_mcp_server", "ingest", str(root / "docs"),
                "--index-dir", str(index), "--embedding-provider", "fake", "--json",
                cwd=root,
            )
            self.assertEqual(ingest_run.returncode, 0, ingest_run.stderr)
            ingest_data = json.loads(ingest_run.stdout)
            self.assertEqual(ingest_data["total_files"], 2)
            self.assertEqual(ingest_data["total_skipped"], 1)

            # 2. chunks.jsonl 与 ingestion_report.json 存在
            chunks_file = index / "default" / "chunks.jsonl"
            report_file = index / "default" / "ingestion_report.json"
            self.assertTrue(chunks_file.is_file(), "chunks.jsonl 必须存在")
            self.assertTrue(report_file.is_file(), "ingestion_report.json 必须存在")
            chunk_lines = [json.loads(line) for line in chunks_file.read_text(encoding="utf-8").splitlines()]
            self.assertGreater(len(chunk_lines), 0)

            # 3. 损坏 PDF 写入 skipped report（reason 含 pdf parse failed）
            # base = root/docs，所以相对路径不含 docs/ 前缀
            report = json.loads(report_file.read_text(encoding="utf-8"))
            skipped_paths = [item["path"] for item in report["skipped"]]
            self.assertIn("broken.pdf", skipped_paths)
            self.assertIn("pdf parse failed", report["skipped"][0]["reason"])

            # 4. query 命中双语关键词，且每条引用都能追溯回磁盘真实文件
            query_run = run_cli(
                "-m", "rag_mcp_server", "query", "检索",
                "--index-dir", str(index), "--json",
                cwd=root,
            )
            self.assertEqual(query_run.returncode, 0, query_run.stderr)
            query_data = json.loads(query_run.stdout)
            self.assertGreaterEqual(len(query_data["results"]), 1, "query 必须命中指定 chunk")

            chunks_by_id = {c["chunk_id"]: c for c in chunk_lines}
            for result in query_data["results"]:
                # 引用来源存在：source_path（相对 docs/）对应磁盘上的真实文件
                source_file = root / "docs" / result["source_path"]
                self.assertTrue(source_file.is_file(), f"引用源文件不存在：{result['source_path']}")
                # 引用来源相关：命中 chunk 在 store 中可追溯
                self.assertIn(result["chunk_id"], chunks_by_id, f"chunk 不可追溯：{result['chunk_id']}")
                self.assertEqual(chunks_by_id[result["chunk_id"]]["text"], result["text"])
                # 命中文本与查询相关
                self.assertIn("检索", result["text"], "命中文本必须与查询词相关")

            # 5. 人类可读输出包含来源与命中文本
            human_run = run_cli(
                "-m", "rag_mcp_server", "query", "MCP",
                "--index-dir", str(index),
                cwd=root,
            )
            self.assertEqual(human_run.returncode, 0, human_run.stderr)
            self.assertIn("MCP", human_run.stdout)
            self.assertIn("::", human_run.stdout)  # chunk_id 格式
            self.assertIn(".md", human_run.stdout)  # source 出现在人类可读输出
            self.assertIn("来源", human_run.stdout)  # sources 元信息

            # 6. 无结果的人类可读与 JSON 输出
            no_hit = run_cli(
                "-m", "rag_mcp_server", "query", "qxzvw",
                "--index-dir", str(index),
                cwd=root,
            )
            self.assertEqual(no_hit.returncode, 0, no_hit.stderr)
            self.assertIn("没有找到匹配", no_hit.stdout)
            no_hit_json = run_cli(
                "-m", "rag_mcp_server", "query", "qxzvw",
                "--index-dir", str(index), "--json",
                cwd=root,
            )
            self.assertEqual(no_hit_json.returncode, 0, no_hit_json.stderr)
            self.assertEqual(json.loads(no_hit_json.stdout)["results"], [])


if __name__ == "__main__":
    unittest.main()
