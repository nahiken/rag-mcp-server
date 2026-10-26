"""One-shot launcher for the RAG-MCP-SERVER offline demo (no Ollama needed).

Cross-platform (PowerShell / cmd / bash): just `python scripts/demo.py`.
It creates a demo doc + an isolated config (sparse strategy, persist_dir under
the temp dir), ingests it, and runs a query — all offline with FakeEmbedding.

Usage:
    python scripts/demo.py            # full flow: create -> ingest -> query
    python scripts/demo.py ingest     # recreate + ingest only
    python scripts/demo.py query 词    # query the ingested data (default term: 检索)
    python scripts/demo.py query 词 --json
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Default demo data lives in the system temp dir so the repo is never polluted.
DEMO_DIR = Path(os.environ.get("RAG_DEMO_DIR", tempfile.gettempdir())) / "rag-demo"

DEMO_DOC = (
    "RAG 混合检索演示，支持中文关键词检索。\n\n"
    "MCP 是 Model Context Protocol，用于连接工具与数据。\n\n"
    "语义检索用向量，关键词检索用 BM25。\n"
)


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONIOENCODING"] = "utf-8"
    print("==>", " ".join(cmd))
    return subprocess.run(
        [sys.executable, *cmd],
        cwd=cwd,
        env=env,
        text=True,
        encoding="utf-8",
        capture_output=True,
    )


def create_data() -> None:
    shutil.rmtree(DEMO_DIR, ignore_errors=True)
    docs = DEMO_DIR / "docs"
    config = DEMO_DIR / "config"
    docs.mkdir(parents=True)
    config.mkdir(parents=True)
    (docs / "notes.txt").write_text(DEMO_DOC, encoding="utf-8")
    (config / "settings.yaml").write_text(
        f"vector_store:\n"
        f"  persist_dir: {(DEMO_DIR / 'chroma').as_posix()}\n"
        f"retrieval:\n"
        f"  strategy: sparse\n"
        f"  top_k: 3\n",
        encoding="utf-8",
    )
    print(f"==> demo 数据已写入 {DEMO_DIR}")


def ingest() -> int:
    create_data()
    proc = _run(
        ["-m", "rag_mcp_server", "ingest", str(DEMO_DIR / "docs"),
         "--embedding-provider", "fake", "--index-dir", str(DEMO_DIR / "out"), "--json"],
        cwd=DEMO_DIR,
    )
    if proc.returncode != 0:
        print(proc.stderr)
        return proc.returncode
    if not proc.stdout:
        print("==> ingest 无输出，未知异常（stdout 为空）", file=sys.stderr)
        return 1
    if proc.stdout:
        print(proc.stdout)
    try:
        summary = json.loads(proc.stdout)
    except json.JSONDecodeError:
        print("==> ingest 输出非 JSON：", file=sys.stderr)
        print(proc.stdout, file=sys.stderr)
        return 1
    print(f"==> 摄取完成：{summary['total_files']} 文件，{summary['total_chunks']} chunk")
    print("    查询示例： python scripts/demo.py query 检索")
    return 0


def query(term: str, *extra: str) -> int:
    proc = _run(
        ["-m", "rag_mcp_server", "query", term,
         "--index-dir", str(DEMO_DIR / "out"), *extra],
        cwd=DEMO_DIR,
    )
    if proc.stdout:
        print(proc.stdout)
    if proc.returncode != 0 and proc.stderr:
        print(proc.stderr)
    return proc.returncode


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "run":
        code = ingest()
        if code != 0:
            return code
        return query("检索")
    if cmd == "ingest":
        return ingest()
    if cmd == "query":
        rest = sys.argv[2:]
        term = rest[0] if rest else "检索"
        return query(term, *rest[1:])
    print("用法: python scripts/demo.py [run|ingest|query <词> [--json]]", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
