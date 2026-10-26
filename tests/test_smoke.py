import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rag_mcp_server import __version__


def test_package_imports() -> None:
    assert __version__ == "0.1.0"