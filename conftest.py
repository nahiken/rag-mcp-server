"""Pytest bootstrap: make src/ importable from the repo root.

The package lives under src/rag_mcp_server; without an editable install this
conftest is what lets `python -m pytest` import it.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
