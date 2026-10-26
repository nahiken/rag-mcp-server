"""Golden fixture + e2e recall gate (P10.7 T2-03).

离线门禁：先摄取 ``tests/fixtures/eval_docs/`` 的 3 个单 chunk golden fixture，
再 subprocess 运行 ``scripts/evaluate.py``（真实进程、真实 rag_service.query），
用 ``tests/fixtures/golden_test_set.json`` 的硬编码 ``<document_id>::0`` 验证
通过路径与阈值失败路径。全程临时 settings（强制 sparse、trace 关闭）、临时
collection/index，不依赖本地历史 Chroma 状态。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
EVAL_DOCS = ROOT / "tests" / "fixtures" / "eval_docs"
GOLDEN_TEST_SET = ROOT / "tests" / "fixtures" / "golden_test_set.json"
INGEST_SCRIPT = ROOT / "scripts" / "ingest.py"
EVALUATE_SCRIPT = ROOT / "scripts" / "evaluate.py"

EXPECTED_CHUNK_IDS = {
    # ADR 0016: chunk ids carry the default:: collection prefix
    "知识库检索返回文档片段": "default::zh-knowledge-md::0",
    "MCP 工具 semantic_search": "default::en-rag-terms-md::0",
    "hit_rate mrr 门禁": "default::project-terms-md::0",
}


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


def write_offline_config(root: Path) -> None:
    """临时 settings：sparse 策略 + 临时 persist_dir + trace 关闭（离线、不污染 logs/）。"""
    config_dir = root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "settings.yaml").write_text(
        f"vector_store:\n"
        f"  persist_dir: {root / 'chroma'}\n"
        f"retrieval:\n"
        f"  strategy: sparse\n"
        f"trace:\n"
        f"  enabled: false\n",
        encoding="utf-8",
    )


def ingest_fixtures(root: Path, index_dir: Path) -> None:
    docs = root / "docs"
    shutil.copytree(EVAL_DOCS, docs)
    completed = run_cli(
        str(INGEST_SCRIPT),
        str(docs),
        "--index-dir",
        str(index_dir),
        "--embedding-provider",
        "fake",
        "--json",
        cwd=root,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["total_files"] == 3, report
    assert report["total_skipped"] == 0, report


def evaluate(root: Path, index_dir: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return run_cli(
        str(EVALUATE_SCRIPT),
        "--test-set",
        str(GOLDEN_TEST_SET),
        "--index-dir",
        str(index_dir),
        "--top-k",
        "3",
        "--strategy",
        "sparse",
        *extra,
        cwd=root,
    )


class RecallGateE2ETests(unittest.TestCase):
    def test_recall_gate_passes_offline(self) -> None:
        """真实摄取 golden fixture 后，e2e 门禁通过：hit_rate/mrr 双 1.0。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = root / "index"
            write_offline_config(root)
            ingest_fixtures(root, index)

            human = evaluate(root, index)
            self.assertEqual(human.returncode, 0, human.stderr)
            self.assertIn("hit_rate", human.stdout)
            self.assertIn("mrr", human.stdout)
            for query in EXPECTED_CHUNK_IDS:
                self.assertIn(query, human.stdout)

            json_run = evaluate(root, index, "--json")
            self.assertEqual(json_run.returncode, 0, json_run.stderr)
            report = json.loads(json_run.stdout)
            self.assertEqual(report["hit_rate"], 1.0)
            self.assertEqual(report["mrr"], 1.0)
            self.assertEqual(len(report["per_query"]), 3)
            for item in report["per_query"]:
                self.assertIn(EXPECTED_CHUNK_IDS[item["query"]], item["retrieved_chunk_ids"])

    def test_recall_gate_fails_when_threshold_raised(self) -> None:
        """同一 index + 同一 golden，但把 min_mrr 抬到 1.1：报告保留在 stdout、摘要进 stderr、退出码非 0。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = root / "index"
            write_offline_config(root)
            ingest_fixtures(root, index)

            failed = evaluate(root, index, "--min-mrr", "1.1", "--json")
            self.assertNotEqual(failed.returncode, 0)
            report = json.loads(failed.stdout)
            self.assertEqual(report["mrr"], 1.0)
            self.assertIn("评估门禁失败", failed.stderr)
            self.assertNotIn("Traceback", failed.stderr)


if __name__ == "__main__":
    unittest.main()
