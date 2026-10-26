"""Component tests for write_trace — the JSONL trace sink.

Red/green contract (docs/spec-p10-trace.md, ticket 01):
- write_trace appends one trace dict as a single-line JSON document
  (UTF-8, ensure_ascii=False) to {dir}/traces.jsonl
- appends across calls (never truncates)
- a write failure is swallowed (warning), never raised to the caller — the
  observation path must never block the business path
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_mcp_server.observability.logger import write_trace


def test_write_trace_writes_single_line_json(tmp_path):
    write_trace({"trace_type": "query", "trace_id": "abc"}, tmp_path)

    target = tmp_path / "traces.jsonl"
    assert target.is_file()
    lines = target.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["trace_type"] == "query"
    assert parsed["trace_id"] == "abc"


def test_write_trace_appends_across_calls(tmp_path):
    write_trace({"trace_type": "query"}, tmp_path)
    write_trace({"trace_type": "ingestion"}, tmp_path)

    lines = (tmp_path / "traces.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["trace_type"] == "query"
    assert json.loads(lines[1])["trace_type"] == "ingestion"


def test_write_trace_preserves_utf8_without_escaping(tmp_path):
    write_trace({"trace_type": "query", "note": "中文耗时 12ms"}, tmp_path)

    line = (tmp_path / "traces.jsonl").read_text(encoding="utf-8")
    assert "中文耗时" in line
    assert "\\u4e2d" not in line  # ensure_ascii=False


def test_write_trace_creates_missing_dir(tmp_path):
    target = tmp_path / "deep" / "logs"
    write_trace({"trace_type": "ingestion"}, target)
    assert (target / "traces.jsonl").is_file()


def test_write_trace_failure_is_swallowed_not_raised(tmp_path):
    # a non-writable target must not raise — observation never blocks business
    write_trace({"trace_type": "query"}, tmp_path / "missing" / "parent")
    # no exception raised; nothing to assert beyond that
