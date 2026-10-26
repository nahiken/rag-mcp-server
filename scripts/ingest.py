"""Script entry for ingest. Delegates to the same application service as the module entry."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rag_mcp_server.cli.main import main

if __name__ == "__main__":
    argv = ["ingest", *sys.argv[1:]]
    raise SystemExit(main(argv))
