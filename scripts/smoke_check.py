from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

REQUIRED_FILES = [
    "AGENTS.md",
    "init.sh",
    "claude-progress.md",
    "feature_list.json",
    "task_plan.md",
    "MISSION.md",
    "CONTEXT.md",
    "DEV_SPEC.md",
]

REQUIRED_DIRS = [
    "src/rag_mcp_server",
    "tests/unit",
    "tests/e2e",
    "scripts",
]


def main() -> int:
    missing = [name for name in REQUIRED_FILES if not (ROOT / name).exists()]
    if missing:
        print(f"missing required files: {missing}", file=sys.stderr)
        return 1

    missing_dirs = [name for name in REQUIRED_DIRS if not (ROOT / name).is_dir()]
    if missing_dirs:
        print(f"missing required directories: {missing_dirs}", file=sys.stderr)
        return 1

    data = json.loads((ROOT / "feature_list.json").read_text(encoding="utf-8"))
    in_progress = [f["id"] for f in data["features"] if f["status"] == "in_progress"]
    if len(in_progress) > 1:
        print(f"more than one feature in progress: {in_progress}", file=sys.stderr)
        return 1

    feature_ids = {f["id"] for f in data["features"]}
    required_features = {"core-001", "ingestion-001", "retrieval-001", "cli-001", "e2e-001"}
    missing_features = sorted(required_features - feature_ids)
    if missing_features:
        print(f"missing planned features: {missing_features}", file=sys.stderr)
        return 1

    dev_spec = (ROOT / "DEV_SPEC.md").read_text(encoding="utf-8")
    for marker in ["Minimum RAG Loop", "unittest", "data/index/chunks.jsonl"]:
        if marker not in dev_spec:
            print(f"DEV_SPEC.md missing marker: {marker}", file=sys.stderr)
            return 1

    package = importlib.import_module("rag_mcp_server")
    if package.__version__ != "0.1.0":
        print("unexpected package version", file=sys.stderr)
        return 1

    print("smoke_check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())