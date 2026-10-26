"""Trace file reading and per-file stage aggregation for the Dashboard.

The ingestion pipeline writes one JSON document per line to
``logs/traces.jsonl`` (ADR 0007). This module turns that file into rows the
ingestion_traces page can render as a stacked bar: one row per *file*, with the
load/split/transform/embed/upsert stage timings summed. ``skip`` /
``integrity_skip`` files are surfaced as skipped, not fabricated as timings.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# The ingestion stages we aggregate, in pipeline order — the SINGLE source of
# the stage list. Traces record them per file; transform only appears when
# transforms are configured. StageTiming.timings keys and the pages' chart
# columns are derived from here, so adding a pipeline stage touches one place
# (S3/S6: no parallel arrays to keep in sync).
_STAGE_KEYS = ("load", "split", "transform", "embed", "upsert")
# Public column keys ("<stage>_ms") — the pages melt on these for the chart.
STAGE_MS_KEYS = tuple(f"{stage}_ms" for stage in _STAGE_KEYS)
# Stage events that mark a file as not-loaded (surfaced as skipped).
_SKIP_STAGES = ("skip", "integrity_skip")


@dataclass(frozen=True)
class StageTiming:
    """Per-file timing row for the ingestion waterfall.

    ``timings`` maps ``STAGE_MS_KEYS`` (e.g. ``"load_ms"``) to summed elapsed
    milliseconds; ``total_ms`` is the sum of the stages present. A skipped file
    carries all-zero timings: the pipeline did spend time in ``load`` before
    failing, but the Dashboard semantic is "skipped = untimed", so the row
    reads honestly as skipped (T4).
    """

    file: str
    timings: dict[str, float]
    total_ms: float
    skipped: bool

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"file": self.file}
        data.update(self.timings)
        data["total_ms"] = self.total_ms
        data["skipped"] = self.skipped
        return data


def load_traces(path: Path | str) -> list[dict[str, Any]]:
    """Read traces.jsonl into a list of trace dicts.

    A missing file or empty file returns ``[]``; a line that is not valid JSON
    is skipped so a dirty file never crashes the Dashboard (US 13).
    """
    path = Path(path)
    if not path.is_file():
        return []
    traces: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                traces.append(data)
    return traces


def _stage_ms(record: dict[str, Any]) -> float:
    value = record.get("elapsed_ms")
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def stage_data_by_name(trace: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Map a trace's stages to their ``data`` dicts, keyed by stage name.

    Stages keep pipeline order in ``trace["stages"]``; this is a lookup helper
    for pages that want one stage's data (e.g. ``split.data.chunks``) without
    re-walking the list. Records without a data dict are skipped. A stage that
    never ran (e.g. no ``transform`` configured) is simply absent.

    NOTE: an ingestion trace is *run*-scoped — one ``ingest()`` with N files
    emits N load/split/embed/upsert groups, so a stage-name key would collapse
    them (last file wins). Prefer :func:`group_files_by_stage_data` for traces
    that may contain more than one file; this helper is correct only for
    single-file runs (HTTP uploads) or query traces.
    """
    out: dict[str, dict[str, Any]] = {}
    for record in trace.get("stages"):
        if not isinstance(record, dict):
            continue
        stage = record.get("stage")
        data = record.get("data")
        if isinstance(stage, str) and isinstance(data, dict):
            out[stage] = data
    return out


def group_files_by_stage_data(trace: dict[str, Any]) -> list[dict[str, Any]]:
    """Group one ingestion trace's stage records per FILE, preserving order.

    One ``ingest()`` run can cover many files; each file contributes its own
    load/split(/transform)/embed/upsert records. The pipeline records a
    ``file`` name on the ``load`` (and ``skip``/``integrity_skip``) records and
    the stages that follow belong to that file — so we keep a "current file"
    cursor as we walk the sequential stage records (same rule as
    ``aggregate_by_file``). Each returned entry is
    ``{"file": <name or None>, "stages": [{stage, data, elapsed_ms}, ...]}``,
    in file order. Used by the 摄取追踪 drilldown so a multi-file run shows
    every file instead of collapsing to the last one.
    """
    files: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for record in trace.get("stages"):
        if not isinstance(record, dict):
            continue
        data = record.get("data") if isinstance(record.get("data"), dict) else {}
        file = data.get("file")
        if isinstance(file, str) and file:
            current = {"file": file, "stages": []}
            files.append(current)
        if current is not None and isinstance(record.get("stage"), str):
            current["stages"].append(
                {
                    "stage": record.get("stage"),
                    "data": data,
                    "elapsed_ms": record.get("elapsed_ms"),
                }
            )
    return files


def aggregate_by_file(traces: list[dict[str, Any]]) -> list[StageTiming]:
    """Aggregate ingestion stage timings per file, sorted by total desc.

    Only ``trace_type == "ingestion"`` traces contribute. A file's per-stage
    times are summed across all its records; the total is the sum of the stages
    present. A ``skip``/``integrity_skip`` file is marked ``skipped=True`` and
    its timings are zeroed (no fabricated timings, and a skip never crowds a
    real file out of a chart). Returns the FULL set of files — top-N charting
    is the page's job, so no file ever disappears from the detail/skip views
    (T2).
    """
    sums: dict[str, list[float]] = defaultdict(lambda: [0.0] * len(STAGE_MS_KEYS))
    skipped: set[str] = set()
    # Only load/skip/integrity_skip records carry {"file": ...}; the stages
    # that follow (split/embed/upsert) belong to the file the load started, so
    # we keep a "current file" cursor as we walk the sequential stage records.
    current_file = ""

    for trace in traces:
        if trace.get("trace_type") != "ingestion":
            continue
        stages = trace.get("stages")
        if not isinstance(stages, list):
            continue
        for record in stages:
            if not isinstance(record, dict):
                continue
            stage = record.get("stage")
            data = record.get("data") if isinstance(record.get("data"), dict) else {}
            file = data.get("file")
            if isinstance(file, str) and file:
                current_file = file
            else:
                file = current_file
            if not file:
                continue
            if stage in _SKIP_STAGES:
                skipped.add(file)
                sums[file]  # register the file so a skip-only file still shows up
                continue
            if stage in _STAGE_KEYS:
                idx = _STAGE_KEYS.index(stage)
                sums[file][idx] += _stage_ms(record)

    rows: list[StageTiming] = []
    for file, ms in sums.items():
        is_skipped = file in skipped
        timings = (
            {ms_key: value for ms_key, value in zip(STAGE_MS_KEYS, ms)}
            if not is_skipped
            else {ms_key: 0.0 for ms_key in STAGE_MS_KEYS}
        )
        rows.append(
            StageTiming(
                file=file,
                timings=timings,
                total_ms=0.0 if is_skipped else sum(timings.values()),
                skipped=is_skipped,
            )
        )

    rows.sort(key=lambda r: r.total_ms, reverse=True)
    return rows


# --- query traces (ADR 0014 extension point) --------------------------------

# The query stages recorded by rag_service.query, in order — the single source
# for the query_traces page (query_processing / dense / sparse / fusion / rerank).
_QUERY_STAGE_KEYS = (
    "query_processing",
    "dense_retrieval",
    "sparse_retrieval",
    "fusion",
    "rerank",
)
QUERY_STAGE_MS_KEYS = tuple(f"{stage}_ms" for stage in _QUERY_STAGE_KEYS)


@dataclass(frozen=True)
class QueryTraceRow:
    """One query trace: the query text, its stage timings and total.

    ``result_count`` is the actual number of results returned (recorded in the
    trace metadata by rag_service at finish — C8), not the requested top_k.
    """

    query: str
    timings: dict[str, float]
    total_ms: float
    result_count: int

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"query": self.query, "result_count": self.result_count}
        data.update(self.timings)
        data["total_ms"] = self.total_ms
        return data


def aggregate_queries(traces: list[dict[str, Any]]) -> list[QueryTraceRow]:
    """Aggregate query stage timings per query, sorted by total desc.

    Only ``trace_type == "query"`` traces contribute. A query's per-stage times
    are summed across its records; ``result_count`` comes from the trace
    metadata (the actual returned count, written by rag_service at finish —
    C8), falling back to ``top_k`` for traces written before that. Returns the
    FULL set — the page decides top-N.
    """
    rows: list[QueryTraceRow] = []
    for trace in traces:
        if trace.get("trace_type") != "query":
            continue
        stages = trace.get("stages")
        if not isinstance(stages, list):
            continue
        metadata = trace.get("metadata") if isinstance(trace.get("metadata"), dict) else {}
        query = metadata.get("query", "")
        if not isinstance(query, str) or not query:
            continue
        timings = {f"{stage}_ms": 0.0 for stage in _QUERY_STAGE_KEYS}
        total = 0.0
        for record in stages:
            if not isinstance(record, dict):
                continue
            stage = record.get("stage")
            if stage in _QUERY_STAGE_KEYS:
                ms = _stage_ms(record)
                timings[f"{stage}_ms"] += ms
                total += ms
        # C8: prefer the real result count recorded at finish; old traces fall
        # back to top_k (which was the historical meaning of the field).
        result_count = metadata.get("result_count", metadata.get("top_k", 0))
        rows.append(
            QueryTraceRow(
                query=query,
                timings=timings,
                total_ms=total,
                result_count=int(result_count) if isinstance(result_count, int) else 0,
            )
        )
    rows.sort(key=lambda r: r.total_ms, reverse=True)
    return rows
