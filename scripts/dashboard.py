"""Launch the Streamlit Dashboard from the repo.

``streamlit run`` does not reliably inherit PYTHONPATH on Windows, so this
launcher adds ``src/`` to sys.path (same convention as scripts/verify.py etc.)
then hands off to streamlit's CLI entrypoint.

streamlit is a dev-only optional dependency: the import is lazy so that when it
is missing the user gets the same friendly stderr hint as ``app.py`` (US 9)
instead of a raw ImportError traceback.

Usage:
    python scripts/dashboard.py
    # optional: --server.port 8502  etc. (any streamlit run flag)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
APP = SRC / "rag_mcp_server" / "observability" / "dashboard" / "app.py"


def main() -> int:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

    try:
        from streamlit.web import cli as stcli
    except ModuleNotFoundError:
        sys.stderr.write(
            "Dashboard 需要 streamlit（dev 依赖）。请安装：\n"
            "    pip install \"streamlit>=1.36\"\n"
            "或：pip install -e \".[dev]\"\n"
        )
        return 1

    sys.argv = ["streamlit", "run", str(APP), *sys.argv[1:]]
    raise SystemExit(stcli.main())


if __name__ == "__main__":
    raise SystemExit(main())
