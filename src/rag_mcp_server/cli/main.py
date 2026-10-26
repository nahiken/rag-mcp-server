"""CLI entry points for ingest and query.

Both `python -m rag_mcp_server` and the scripts/ entry points call
the application service here; business logic lives only in
application/rag_service.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rag_mcp_server.application import rag_service
from rag_mcp_server.core.errors import RagMcpError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag_mcp_server", description="最小 RAG MCP Server")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest_parser = sub.add_parser("ingest", help="导入 txt/md/pdf 文档并写入向量库")
    ingest_parser.add_argument("path", type=Path, help="单个文件或目录")
    ingest_parser.add_argument("--index-dir", type=Path, default=rag_service.DEFAULT_INDEX_DIR)
    ingest_parser.add_argument(
        "--collection",
        default=None,
        help="目标集合（默认读配置 vector_store.collection；每个集合一个独立向量库，ADR 0016）",
    )
    ingest_parser.add_argument(
        "--embedding-provider",
        default=None,
        help="覆盖 embedding provider（如 fake，离线运行；默认读配置）",
    )
    ingest_parser.add_argument("--json", action="store_true", help="输出 JSON")
    ingest_parser.add_argument(
        "--force",
        action="store_true",
        help="强制重摄：绕过文件级增量跳过（integrity），但内容未变的 chunk 仍不重复写入",
    )

    query_parser = sub.add_parser("query", help="关键词/语义检索")
    query_parser.add_argument("text", help="查询关键词（支持 collection:/source:/tag:/doc_type: 过滤语法）")
    query_parser.add_argument("--index-dir", type=Path, default=rag_service.DEFAULT_INDEX_DIR)
    query_parser.add_argument(
        "--collection",
        default=None,
        help="目标集合（默认读配置；查询文本里的 collection: 语法亦可，显式参数优先，ADR 0016）",
    )
    # None -> rag_service.query reads retrieval.top_k from config (default 3)
    query_parser.add_argument("--top-k", type=int, default=None)
    query_parser.add_argument("--json", action="store_true", help="输出 JSON")
    return parser


def _print_user_error(exc: Exception) -> int:
    print(f"错误：{exc}", file=sys.stderr)
    return 1


def run_ingest(args: argparse.Namespace) -> int:
    try:
        summary = rag_service.ingest(
            args.path,
            index_dir=args.index_dir,
            embedding_provider=args.embedding_provider,
            force=args.force,
            collection=args.collection,
        )
    except (RagMcpError, ValueError) as exc:
        # RagMcpError is the documented hierarchy; ValueError covers factory
        # fail-fast for unknown providers / invalid settings values, so they
        # surface as a readable user error, not a traceback (CONTEXT 用户错误).
        return _print_user_error(exc)

    if args.json:
        payload = {
            "imported": summary.imported,
            "skipped": summary.skipped,
            "total_files": summary.total_files,
            "total_skipped": summary.total_skipped,
            "total_chunks": summary.total_chunks,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for item in summary.imported:
            print(f"已导入 {item['path']}（document_id={item['document_id']}，chunk 数={item['chunk_count']}）")
        for item in summary.skipped:
            print(f"跳过 {item['path']}（{item['reason']}）")
        print(f"汇总：文件 {summary.total_files} 个，跳过 {summary.total_skipped} 个，chunk {summary.total_chunks} 个")
        print(f"chunks 已写入 {rag_service.evidence_dir(args.index_dir, args.collection) / 'chunks.jsonl'}")
    return 0


def run_query(args: argparse.Namespace) -> int:
    try:
        results = rag_service.query(
            args.text,
            index_dir=args.index_dir,
            top_k=args.top_k,
            collection=args.collection,
        )
    except (RagMcpError, ValueError) as exc:
        # invalid argument (e.g. negative/zero top_k, unknown strategy) is a
        # user-facing error, not a traceback; RagMcpError is the documented
        # hierarchy, ValueError covers input-validation fail-fast (P1 fix)
        return _print_user_error(exc)

    if args.json:
        print(
            json.dumps(
                {
                    "query": args.text,
                    "results": [
                        {
                            "chunk_id": r.chunk_id,
                            "source_path": r.source_path,
                            "score": r.score,
                            "text": r.text,
                            "sources": r.sources,
                        }
                        for r in results
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        if not results:
            print("没有找到匹配 chunk")
            return 0
        for rank, result in enumerate(results, start=1):
            print(f"[{rank}] {result.chunk_id}  score={result.score}  source={result.source_path}")
            print(f"    来源: {', '.join(result.sources) if result.sources else '-'}")
            print(f"    {result.text}")
    return 0


def main(argv: list[str] | None = None) -> int:
    _ensure_utf8_stdio()
    args = build_parser().parse_args(argv)
    if args.command == "ingest":
        return run_ingest(args)
    if args.command == "query":
        return run_query(args)
    return 1


def _ensure_utf8_stdio() -> None:
    """Force UTF-8 on Windows consoles so Chinese output is not mangled.

    Python falls back to the ANSI code page (GBK on zh-CN Windows) for
    stdout when it is not a console with a known encoding. Reconfiguring
    the stream makes CLI output and child-process pipes consistently UTF-8.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
