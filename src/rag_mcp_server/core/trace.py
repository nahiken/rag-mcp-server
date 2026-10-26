"""Trace recording primitive.

The single source of stage timing for query / ingestion traces. Callers only
declare *what* they completed; TraceContext measures *how long* it took
(monotonic wall clock between record_stage calls). finish() finalizes the
trace; to_dict() produces a pure, JSON-serializable dict that write_trace
(observability/logger.py) persists as one JSONL line.

Contract (docs/spec-p10-trace.md, ADR 0007):
- trace_type is Literal["query", "ingestion"]
- record_stage auto-measures elapsed_ms since the previous record_stage
  (elapsed_ms is an override, for externally-timed stages)
- to_dict() is JSON-serializable (no datetimes — RFC3339 string timestamps)

The enabled/dir resolution helpers here (trace_is_enabled / resolve_trace_dir)
are shared by every call site so the settings shape is resolved in one place.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Literal

_TraceType = Literal["query", "ingestion"]

DEFAULT_TRACE_DIR = Path("logs")


def _now_iso() -> str:
    """RFC3339-ish timestamp string (second precision) for trace fields."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def trace_is_enabled(settings: dict) -> bool:
    """True when tracing is on (settings trace.enabled, default enabled)."""
    return (settings.get("trace") or {}).get("enabled", True) is not False


def resolve_trace_dir(settings: dict, trace_dir: Path | str | None) -> Path:
    """Resolve where traces are written: explicit override, else settings, else logs/."""
    if trace_dir is not None:
        return Path(trace_dir)
    return Path((settings.get("trace") or {}).get("dir", DEFAULT_TRACE_DIR))


class TraceContext:
    """Accumulates stage records for one query or ingestion trace."""

    def __init__(self, trace_type: _TraceType, metadata: dict | None = None) -> None:
        self._trace_type = trace_type
        self._trace_id = uuid.uuid4().hex
        self._started_at = _now_iso()
        self._finished_at: str | None = None
        self._stages: list[dict] = []
        # Optional top-level metadata (JSON-serializable). The query trace
        # carries {"query":..., "result_count":..., "top_k":...} so the
        # Dashboard's query_traces page can label rows without a per-stage
        # field (C1/C8). Empty dict -> absent from to_dict (trace shape for
        # ingestion stays unchanged).
        self._metadata: dict = dict(metadata) if metadata else {}
        self._start_monotonic = time.monotonic()
        self._last_ts = self._start_monotonic

    def record_stage(
        self,
        stage: str,
        data: dict,
        elapsed_ms: float | None = None,
    ) -> None:
        """Record one completed stage.

        ``elapsed_ms`` overrides the auto-measured duration; when None the
        duration since the previous record_stage (or construction) is used.
        """
        now = time.monotonic()
        if elapsed_ms is None:
            elapsed_ms = (now - self._last_ts) * 1000.0
        self._last_ts = now
        self._stages.append({"stage": stage, "data": data, "elapsed_ms": elapsed_ms})

    def update_metadata(self, **kwargs) -> None:
        """Merge extra top-level metadata fields (e.g. result_count at finish)."""
        self._metadata.update(kwargs)

    def finish(self) -> None:
        """Finalize the trace (records the finish timestamp)."""
        self._finished_at = _now_iso()

    def to_dict(self) -> dict:
        """Return the trace as a pure, JSON-serializable dict."""
        total = self._total_elapsed_ms()
        data = {
            "trace_id": self._trace_id,
            "trace_type": self._trace_type,
            "started_at": self._started_at,
            "finished_at": self._finished_at or self._started_at,
            "total_elapsed_ms": total,
            "stages": self._stages,
        }
        if self._metadata:
            data["metadata"] = self._metadata
        return data

    def _total_elapsed_ms(self) -> float:
        # whole trace duration: construction (start) to finish; rounded to keep
        # the JSONL output tidy
        return round((time.monotonic() - self._start_monotonic) * 1000.0, 3)


class DisabledTrace:
    """A trace stub used when tracing is disabled (ADR 0007: enabled: false).

    record_stage/finish/to_dict are no-ops so call sites don't branch on the
    enabled flag — they just call the same API and nothing is persisted.
    """

    def record_stage(self, stage: str, data: dict, elapsed_ms: float | None = None) -> None:
        return None

    def update_metadata(self, **kwargs) -> None:
        return None

    def finish(self) -> None:
        return None

    def to_dict(self) -> dict:
        return {}
