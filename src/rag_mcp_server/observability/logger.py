"""Observability sinks: trace JSONL writer.

write_trace appends one trace dict as a single-line JSON document to
`{dir}/traces.jsonl` (UTF-8, ensure_ascii=False so Chinese stage payloads stay
readable). The write is best-effort: a failure is logged as a warning and
swallowed, because the observation path must never block the business path
(query/ingest return normally even if logging is broken).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

TRACES_FILENAME = "traces.jsonl"


def write_trace(trace_dict: dict, trace_dir: Path | str) -> None:
    """Append one trace as a JSONL line under ``trace_dir``.

    Never raises: observation failures are warnings, not business errors.
    """
    target = Path(trace_dir) / TRACES_FILENAME
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(trace_dict, ensure_ascii=False)
        with open(target, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except (OSError, TypeError, ValueError) as exc:  # noqa: BLE001 - best-effort
        logger.warning("trace 写盘失败（%s）: %s", target, exc)
