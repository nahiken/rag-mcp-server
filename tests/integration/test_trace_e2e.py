"""Trace 端到端集成 — P10.1（p10-trace-001）真实链路验证。

覆盖 spec-p10-trace.md 的端到端面（unit 测试不覆盖的部分）：
  E1. 默认配置下（trace.enabled=true, trace.dir=logs）真实 query/ingest 落盘
      {cwd}/logs/traces.jsonl —— 验证相对目录解析与默认行为（spike 已确认：
      路径相对当前工作目录解析）。
  E2. CLI 双进程（ingest 子进程 + query 子进程）各落一条 trace，JSONL 逐行
      累积、单行合法 JSON、UTF-8 中文不转义、可被 jq 直接消费。
  E3. 默认配置混合调用：先 ingest 再 query，同一 traces.jsonl 内既有
      ingestion 又有 query trace，每行字段契约完整（trace_id/trace_type/
      started_at/finished_at/total_elapsed_ms/stages/各阶段 elapsed_ms）。
  E4. （真实链路，需本地 Ollama nomic-embed-text @127.0.0.1:11451）真实
      Ollama embedding 摄取 + dense 查询，trace 阶段与离线路径一致，
      并证明真实 embedding 的耗时确实被计入 query_processing 阶段。

默认全量（pytest -m "not integration"）不运行本模块；显式运行：
    PYTHONPATH=src python -m pytest -m integration tests/integration/test_trace_e2e.py -v
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


def _run_cli(*args: str, cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    base_env = os.environ.copy()
    existing = base_env.get("PYTHONPATH")
    base_env["PYTHONPATH"] = str(SRC) + (os.pathsep + existing if existing else "")
    base_env["PYTHONIOENCODING"] = "utf-8"
    if env:
        base_env.update(env)
    return subprocess.run(
        [sys.executable, *args],
        cwd=cwd,
        env=base_env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


def _read_trace_lines(trace_file: Path) -> list[dict]:
    if not trace_file.is_file():
        return []
    return [
        json.loads(line)
        for line in trace_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _assert_trace_contract(trace: dict) -> None:
    """trace 字段契约（user stories 3-4）：必需字段 + stages 形状。"""
    for field in ("trace_id", "trace_type", "started_at", "finished_at", "total_elapsed_ms", "stages"):
        assert field in trace, f"trace 缺字段 {field}"
    assert isinstance(trace["stages"], list)
    for stage in trace["stages"]:
        assert isinstance(stage, dict)
        assert "stage" in stage and "data" in stage and "elapsed_ms" in stage
        assert isinstance(stage["elapsed_ms"], (int, float))
    assert isinstance(trace["total_elapsed_ms"], (int, float))


# -- E1: 默认配置真实落盘 {cwd}/logs/traces.jsonl -----------------------------

def test_default_config_writes_trace_to_cwd_logs():
    """默认配置（trace.enabled=true, dir=logs）下，真实调用落盘到 cwd/logs。"""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        root = Path(tmp)
        (root / "docs").mkdir()
        (root / "docs" / "note.md").write_text("RAG 端到端 trace 验证。", encoding="utf-8")

        # 使用真实默认配置（无 config/settings.yaml 覆盖）离线 fake embedding
        ingest_run = _run_cli(
            "-m", "rag_mcp_server", "ingest", str(root / "docs"),
            "--index-dir", str(root / "index"),
            "--embedding-provider", "fake", "--json",
            cwd=root,
        )
        assert ingest_run.returncode == 0, ingest_run.stderr

        trace_file = root / "logs" / "traces.jsonl"
        assert trace_file.is_file(), f"默认 trace.dir=logs 应在 cwd/logs 落盘：{trace_file}"
        traces = _read_trace_lines(trace_file)
        assert len(traces) == 1
        assert traces[0]["trace_type"] == "ingestion"
        _assert_trace_contract(traces[0])


# -- E2: CLI 双进程逐行累积 + 单行 JSON + UTF-8 --------------------------------

def test_cli_ingest_then_query_appends_lines_and_stays_valid_jsonl():
    """ingest 子进程 + query 子进程先后运行：traces.jsonl 两行、单行合法 JSON、
    中文不转义，可被逐行 json.loads 消费（等价 jq 可解析）。"""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        root = Path(tmp)
        (root / "docs").mkdir()
        (root / "docs" / "笔记.md").write_text(
            "RAG 关键词检索 baseline，结果按 score 排序。", encoding="utf-8"
        )

        # 离线配置：fake embedding + sparse（不依赖外部服务）+ 显式 trace.dir
        (root / "config").mkdir()
        (root / "config" / "settings.yaml").write_text(
            "retrieval:\n  strategy: sparse\n  top_k: 3\n"
            "trace:\n  enabled: true\n  dir: traces\n",
            encoding="utf-8",
        )

        ingest_run = _run_cli(
            "-m", "rag_mcp_server", "ingest", str(root / "docs"),
            "--index-dir", str(root / "index"),
            "--embedding-provider", "fake", "--json",
            cwd=root,
        )
        assert ingest_run.returncode == 0, ingest_run.stderr

        query_run = _run_cli(
            "-m", "rag_mcp_server", "query", "检索",
            "--index-dir", str(root / "index"), "--json",
            cwd=root,
        )
        assert query_run.returncode == 0, query_run.stderr

        trace_file = root / "traces" / "traces.jsonl"
        assert trace_file.is_file()
        traces = _read_trace_lines(trace_file)
        assert len(traces) == 2, f"ingest+query 应累积 2 行，实际 {len(traces)}"
        assert traces[0]["trace_type"] == "ingestion"
        assert traces[1]["trace_type"] == "query"
        for trace in traces:
            _assert_trace_contract(trace)

        # 中文 stage 数据不应被 \uXXXX 转义（ensure_ascii=False）。
        # 注：query trace 只记 stage/data，不含查询文本；中文可读性以
        # ingest trace 的文件路径（笔记.md）验证。
        raw = trace_file.read_text(encoding="utf-8")
        assert "笔记.md" in raw
        assert "\\u" not in raw


# -- E3: 混合调用 + 字段契约 --------------------------------------------------

def test_mixed_calls_same_trace_file_with_full_contract(tmp_path):
    """同一 trace 文件内累积 ingestion 与 query 两条 trace，字段契约完整。"""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("第一份文档内容。", encoding="utf-8")
    (docs / "b.md").write_text("第二份文档内容。", encoding="utf-8")

    from rag_mcp_server.application.rag_service import ingest, query
    from rag_mcp_server.ingestion.pipeline import IngestPipeline
    from rag_mcp_server.libs.embedding.fake import FakeEmbedding
    from rag_mcp_server.libs.vector_store.chroma_store import ChromaVectorStore

    trace_dir = tmp_path / "traces"
    embedding = FakeEmbedding(dim=4)
    store = ChromaVectorStore(persist_dir=None, collection="e2e_trace_mixed")
    pipeline = IngestPipeline(embedding=embedding, store=store, index_dir=tmp_path / "out")

    ingest(docs, base=docs, index_dir=tmp_path / "out", pipeline=pipeline, trace_dir=trace_dir)
    query("第一份", index_dir=tmp_path / "out", pipeline=pipeline, trace_dir=trace_dir)

    traces = _read_trace_lines(trace_dir / "traces.jsonl")
    assert len(traces) == 2
    types = [t["trace_type"] for t in traces]
    assert types == ["ingestion", "query"]
    for trace in traces:
        _assert_trace_contract(trace)
        assert trace["total_elapsed_ms"] >= sum(
            s["elapsed_ms"] for s in trace["stages"]
        ) - 0.01, "总耗时应 ≥ 各阶段耗时之和（阶段计时为分段计时）"


# -- E4: 真实 Ollama 链路（需本地 nomic-embed-text）---------------------------

def test_real_ollama_trace_stages_match_offline_and_embedding_timed():
    """真实 Ollama embedding 摄取 + dense 查询：trace 阶段形状与离线一致，
    query_processing 包含 embedding 耗时（真实网络往返 > 0）。

    CLI query 无 --strategy flag（由配置驱动），故写临时配置强制 dense。
    """
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        root = Path(tmp)
        (root / "docs").mkdir()
        (root / "docs" / "rag-notes.md").write_text(
            "RAG 使用向量检索，返回带引用的结果。", encoding="utf-8"
        )
        (root / "config").mkdir()
        (root / "config" / "settings.yaml").write_text(
            "retrieval:\n  strategy: dense\n  top_k: 2\n",
            encoding="utf-8",
        )

        ingest_run = _run_cli(
            "-m", "rag_mcp_server", "ingest", str(root / "docs"),
            "--index-dir", str(root / "index"),
            "--embedding-provider", "ollama",  # 真实 Ollama
            "--json",
            cwd=root,
        )
        assert ingest_run.returncode == 0, ingest_run.stderr

        query_run = _run_cli(
            "-m", "rag_mcp_server", "query", "向量检索",
            "--index-dir", str(root / "index"),
            "--json", "--top-k", "2",
            cwd=root,
        )
        assert query_run.returncode == 0, query_run.stderr
        query_data = json.loads(query_run.stdout)
        assert len(query_data["results"]) >= 1, "dense 查询应命中"

        trace_file = root / "logs" / "traces.jsonl"
        assert trace_file.is_file()
        traces = _read_trace_lines(trace_file)
        assert len(traces) == 2
        types = [t["trace_type"] for t in traces]
        assert types == ["ingestion", "query"]

        ingest_trace = traces[0]
        # per-file 四阶段形状与离线一致
        assert [s["stage"] for s in ingest_trace["stages"]] == [
            "load", "split", "embed", "upsert"
        ]
        embed_stage = next(s for s in ingest_trace["stages"] if s["stage"] == "embed")
        assert embed_stage["elapsed_ms"] > 1, "真实 Ollama embedding 应产生可测耗时"

        query_trace = traces[1]
        stage_names = [s["stage"] for s in query_trace["stages"]]
        assert stage_names == ["query_processing", "dense_retrieval"]
        # query_processing 含 embedding 时间：真实网络往返应 > 1ms
        qp = next(s for s in query_trace["stages"] if s["stage"] == "query_processing")
        assert qp["elapsed_ms"] > 1, "query_processing 应包含真实 embedding 耗时"
        _assert_trace_contract(query_trace)
