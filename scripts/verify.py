from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def env_with_src() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(SRC) + (os.pathsep + existing if existing else "")
    return env


def run(command: list[str]) -> int:
    print("==>", " ".join(command))
    completed = subprocess.run(command, cwd=ROOT, env=env_with_src())
    return completed.returncode


def main() -> int:
    commands = [
        [sys.executable, "scripts/smoke_check.py"],
        # --ignore dashboard smoke: it imports streamlit at module top level
        # (dev-only extra), so verify must stay green offline without it
        # (spec-p10-8-closeout: verify 不依赖 streamlit). AppTest coverage runs
        # separately when streamlit is installed.
        [sys.executable, "-m", "pytest", "-m", "not integration", "tests/unit", "tests/e2e", "--ignore=tests/e2e/test_dashboard_smoke.py"],
    ]
    for command in commands:
        code = run(command)
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())