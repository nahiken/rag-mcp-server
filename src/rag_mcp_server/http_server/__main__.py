"""Run the RAG HTTP server.

Usage:
    python -m rag_mcp_server.http_server [--host 127.0.0.1] [--port 8123] [--reload]

Default host is 127.0.0.1 (localhost only). Pass --host 0.0.0.0 explicitly if
you need LAN access.
"""

from __future__ import annotations

import argparse
import logging


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG knowledge base HTTP server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
    )

    import uvicorn

    uvicorn.run(
        "rag_mcp_server.http_server.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
