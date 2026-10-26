"""MCP test client for query_knowledge_hub using the official mcp SDK.

Starts the real server over stdio, lists tools, calls
query_knowledge_hub, and verifies that every returned citation traces
back to a real chunk in the store and a real source file on disk, with
hit text relevant to the query.

Exit code 0 means all checks passed; non-zero means a verification
failed.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mcp import ClientSession, StdioServerParameters, stdio_client  # noqa: E402


async def run(query: str, index_dir: str | None, base: str | None) -> int:
    index_path = Path(index_dir) if index_dir else Path("data/index")
    base_path = Path(base) if base else index_path.parent

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "rag_mcp_server.mcp_server.server"],
        cwd=str(ROOT),
        env={"PYTHONPATH": str(SRC), "PYTHONIOENCODING": "utf-8"},
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            print("== initialize ==")
            tools = await session.list_tools()
            names = [t.name for t in tools.tools]
            print("== tools/list ==")
            print("   tools:", names)
            if "query_knowledge_hub" not in names:
                print("缺少 query_knowledge_hub", file=sys.stderr)
                return 1

            arguments: dict = {"query": query}
            if index_dir:
                arguments["index_dir"] = index_dir
            result = await session.call_tool("query_knowledge_hub", arguments)

            print("== tools/call query_knowledge_hub ==")
            print("   query:", query)
            for block in result.content:
                text = getattr(block, "text", str(block))
                print("   content:")
                for line in text.splitlines():
                    print("     " + line)

            citations = _parse_citations(result)
            print("   citations:", len(citations))

    # 验证引用来源存在且相关
    from rag_mcp_server.application.rag_service import load_all_chunks

    store_chunks = {c.chunk_id: c for c in load_all_chunks(index_path)}
    ok = True
    for citation in citations:
        traceable = citation["chunk_id"] in store_chunks
        on_disk = (base_path / citation["source_path"]).is_file()
        related = query.lower() in citation["text"].lower()
        status = "OK" if (traceable and on_disk and related) else "FAIL"
        print(f"   [{status}] {citation['chunk_id']} source={citation['source_path']} "
              f"traceable={traceable} on_disk={on_disk} related={related}")
        if status == "FAIL":
            ok = False

    if not ok:
        print("引用来源验证失败", file=sys.stderr)
        return 1
    return 0


def _parse_citations(result) -> list[dict]:
    """Extract per-chunk lines from the tool's text content.

    The tool returns a human-readable list like:
        [1] docs-rag-notes-md::1 source=docs/rag-notes.md
            <text>
    """
    citations: list[dict] = []
    for block in result.content:
        text = getattr(block, "text", None)
        if not text:
            continue
        current: dict | None = None
        for line in text.splitlines():
            if line.startswith("["):
                import re

                match = re.match(r"^\[\d+\] (\S+) source=(\S+)", line)
                if match:
                    current = {"chunk_id": match.group(1), "source_path": match.group(2), "text": ""}
                    citations.append(current)
            elif current is not None:
                current["text"] += line.strip()
    return citations


def main() -> int:
    parser = argparse.ArgumentParser(description="MCP 测试客户端：通过官方 SDK 调用 query_knowledge_hub")
    parser.add_argument("query", help="查询关键词")
    parser.add_argument("--index-dir", default=None, help="chunks 存储目录，默认 data/index")
    parser.add_argument("--base", default=None, help="导入源路径的 base 目录，用于验证引用文件在磁盘上存在")
    args = parser.parse_args()
    return asyncio.run(run(args.query, args.index_dir, args.base))


if __name__ == "__main__":
    raise SystemExit(main())
