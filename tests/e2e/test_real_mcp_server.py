from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mcp import ClientSession, StdioServerParameters, stdio_client  # noqa: E402

from rag_mcp_server.application.rag_service import ingest  # noqa: E402


def make_fixture(root: Path) -> None:
    # docs/ only — config/, chroma/ and index live at root, never scanned
    (root / "docs").mkdir()
    (root / "docs" / "notes.txt").write_text(
        "RAG 是检索增强生成，最小闭环使用关键词检索。\n\nMCP 是模型上下文协议。",
        encoding="utf-8",
    )
    (root / "docs" / "guide.md").write_text(
        "## 检索\n\n关键词检索返回 top 3 结果并附带引用来源。",
        encoding="utf-8",
    )


def _server_command(index_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m", "rag_mcp_server.mcp_server.server",
    ]


class RealMcpServerE2ETests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        root = Path(self.tmp.name)
        make_fixture(root)
        self.index_dir = root / "index"
        # offline, but persisted to a temp dir the server subprocess can read
        config_dir = root / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "settings.yaml").write_text(
            f"vector_store:\n"
            f"  persist_dir: {root / 'chroma'}\n"
            f"retrieval:\n"
            f"  strategy: sparse\n"
            f"trace:\n"
            f"  enabled: false\n",  # isolate: MCP e2e is not a trace test (US 14)
            encoding="utf-8",
        )
        # ingest must read the temp config, so chdir to root during setup
        self._prev_cwd = os.getcwd()
        os.chdir(root)
        ingest(
            root / "docs",
            base=root / "docs",
            index_dir=self.index_dir,
            embedding_provider="fake",  # offline; store persisted under tmp
        )
        os.chdir(self._prev_cwd)
        self.cwd = root  # server subprocess runs here so config resolves

    def tearDown(self) -> None:
        os.chdir(self._prev_cwd)
        self.tmp.cleanup()

    async def test_query_knowledge_hub_over_stdio_returns_traceable_citations(self) -> None:
        """真实 MCP 客户端通过 stdio 连接官方 SDK server，调用工具并验证引用来源存在且相关。"""
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "rag_mcp_server.mcp_server.server"],
            env=None,
            cwd=str(self.cwd),
        )
        # 让子进程知道 src 位置与 index 位置
        params.env = {
            "PYTHONPATH": str(SRC),
            "PYTHONIOENCODING": "utf-8",
        }

        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                tools = await session.list_tools()
                names = [t.name for t in tools.tools]
                self.assertIn("query_knowledge_hub", names)

                result = await session.call_tool(
                    "query_knowledge_hub",
                    {"query": "检索", "index_dir": str(self.index_dir)},
                )

                # 解析 content 文本
                content_text = ""
                for block in result.content:
                    content_text += getattr(block, "text", str(block))

                self.assertIn("检索", content_text)
                self.assertIn(".md", content_text)
                # 命中内容里的 chunk 引用必须能追溯回磁盘真实文件
                # base = root/docs，所以 chunk_id 无 docs- 前缀；document_id
                # 带 default:: 集合前缀（ADR 0016）
                for chunk_id in ("default::guide-md", "default::notes-txt"):
                    self.assertIn(chunk_id, content_text)

    async def test_semantic_search_over_stdio(self) -> None:
        """semantic_search 工具经官方 SDK stdio 可调用（list_tools / call_tool）。"""
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "rag_mcp_server.mcp_server.server"],
            env=None,
            cwd=str(self.cwd),
        )
        params.env = {"PYTHONPATH": str(SRC), "PYTHONIOENCODING": "utf-8"}

        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                names = [t.name for t in tools.tools]
                self.assertIn("semantic_search", names)

                result = await session.call_tool(
                    "semantic_search",
                    {"query": "检索", "index_dir": str(self.index_dir)},
                )
                content_text = "".join(getattr(b, "text", str(b)) for b in result.content)
                self.assertIn("检索", content_text)
                self.assertIn("default::guide-md", content_text)

    async def test_no_hit_returns_friendly_message_not_error(self) -> None:
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "rag_mcp_server.mcp_server.server"],
            cwd=str(self.cwd),
        )
        params.env = {"PYTHONPATH": str(SRC), "PYTHONIOENCODING": "utf-8"}
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    "query_knowledge_hub",
                    {"query": "qxzvw", "index_dir": str(self.index_dir)},
                )
                content_text = "".join(getattr(b, "text", str(b)) for b in result.content)
                self.assertIn("没有找到匹配", content_text)

    async def _server_params(self) -> StdioServerParameters:
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "rag_mcp_server.mcp_server.server"],
            cwd=str(self.cwd),
        )
        params.env = {"PYTHONPATH": str(SRC), "PYTHONIOENCODING": "utf-8"}
        return params

    async def test_all_tools_are_registered(self) -> None:
        """真实 MCP 客户端 list_tools 断言 9 个工具全注册（4 核心 + 5 Agentic，ADR 0019）。"""
        params = await self._server_params()
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                names = {t.name for t in tools.tools}
                expected = {
                    # 4 core tools
                    "query_knowledge_hub", "semantic_search",
                    "list_collections", "get_document_summary",
                    # 5 Agentic tools (ADR 0019)
                    "list_directory", "list_documents", "preview_document",
                    "verify_fact", "self_check",
                }
                self.assertEqual(expected, names)

    async def test_list_collections_over_stdio(self) -> None:
        """list_collections 经真实 stdio 可调用，返回集合清单（ADR 0016 D23）。"""
        params = await self._server_params()
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    "list_collections",
                    {"index_dir": str(self.index_dir)},
                )
                content_text = "".join(getattr(b, "text", str(b)) for b in result.content)
                self.assertIn("共 1 个集合", content_text)
                self.assertIn("default", content_text)
                self.assertIn("documents=", content_text)
                self.assertIn("chunks=", content_text)

    async def test_get_document_summary_over_stdio(self) -> None:
        """get_document_summary 经真实 stdio 可调用；不存在 id 返回友好错误。"""
        params = await self._server_params()
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                # exists -> rich summary; document_id carries the default::
                # collection prefix (ADR 0016)
                result = await session.call_tool(
                    "get_document_summary",
                    {"document_id": "default::notes-txt", "index_dir": str(self.index_dir)},
                )
                content_text = "".join(getattr(b, "text", str(b)) for b in result.content)
                self.assertIn("document_id=default::notes-txt", content_text)
                self.assertIn("chunks=", content_text)
                # missing -> friendly error, not an exception
                result = await session.call_tool(
                    "get_document_summary",
                    {"document_id": "does-not-exist", "index_dir": str(self.index_dir)},
                )
                content_text = "".join(getattr(b, "text", str(b)) for b in result.content)
                self.assertIn("未找到文档", content_text)

    async def test_list_collections_not_ingested_over_stdio(self) -> None:
        """未摄取索引 -> list_collections / get_document_summary 均返回友好提示（非异常）。"""
        params = await self._server_params()
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                for tool, args in (
                    ("list_collections", {"index_dir": "empty-not-ingested"}),
                    ("get_document_summary", {"document_id": "x", "index_dir": "empty-not-ingested"}),
                ):
                    result = await session.call_tool(tool, args)
                    content_text = "".join(getattr(b, "text", str(b)) for b in result.content)
                    self.assertIn("尚未导入", content_text)


if __name__ == "__main__":
    unittest.main()
