"""Retrieval evaluation gate for the mall knowledge bases (cs_kb / policy_kb / product_kb).

Drop-in extension of scripts/evaluate.py that runs the offline Retrieval gate
per **collection** (evaluate.py only queries the default collection). Reuses
EvalRunner + CustomEvaluator (hit_rate / MRR) and supports fake embedding for
fully-offline, deterministic runs.

Usage:
    python scripts/eval_mall.py --collection cs_kb --index-dir data/index --embedding-provider fake
    python scripts/eval_mall.py --collection policy_kb --test-set <mall golden>.json --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rag_mcp_server.application import rag_service
from rag_mcp_server.config.settings import load_settings
from rag_mcp_server.core.errors import RagMcpError
from rag_mcp_server.libs.evaluator import CustomEvaluator
from rag_mcp_server.observability.evaluation import EvalRunner

_RETRIEVAL_STRATEGIES = ("dense", "sparse", "hybrid")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="按 collection 运行离线 Retrieval 评估门禁")
    parser.add_argument("--collection", required=True,
                        help="要评测的集合名（cs_kb / policy_kb / product_kb）")
    parser.add_argument("--test-set", type=Path, default=None,
                        help="golden 测试集 JSON（含 collection 分组的条），缺省读 mall 侧导出")
    parser.add_argument("--index-dir", type=Path, default=None)
    parser.add_argument("--embedding-provider", default=None,
                        help="覆盖 embedding provider（fake=离线确定性，默认读配置）")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--strategy", choices=_RETRIEVAL_STRATEGIES, default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--min-hit-rate", type=float, default=0.0)
    parser.add_argument("--min-mrr", type=float, default=0.0)
    return parser


def load_test_set(path: Path, collection: str) -> list[dict]:
    """读取 golden 测试集，过滤出属于本 collection 的条。

    每条缺省含 query / expected_sources / reference_answer / collection；
    expected_chunk_ids 由 main 里 _fill_chunk_ids 依据 chunks.jsonl 反查填充
    （数据派生，防手写漂移）。
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("test set 必须是 JSON 数组")
    return [item for item in payload if item.get("collection") == collection]


def _fill_chunk_ids(cases: list[dict], index_dir: Path, collection: str) -> list[dict]:
    """按 evidence chunks.jsonl 的 source_path 反查 expected_chunk_ids。"""
    from rag_mcp_server.ingestion.jsonl_store import read_chunks

    evidence_file = rag_service.evidence_dir(index_dir, collection) / "chunks.jsonl"
    if not evidence_file.exists():
        raise FileNotFoundError(f"集合 {collection} 无 evidence：{evidence_file}")
    src_to_ids: dict[str, list[str]] = {}
    for c in read_chunks(evidence_file):
        src_to_ids.setdefault(c.source_path, []).append(c.chunk_id)
    for item in cases:
        ids: list[str] = []
        for src in item.get("expected_sources", []):
            ids.extend(src_to_ids.get(src, []))
        if not ids:
            # 找不到目标 chunk → 放一个必然 miss 的占位，保留门禁可运行性（下方按 miss 计）
            ids = ["__missing__"]
        item["expected_chunk_ids"] = ids
    return cases


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    index_dir = args.index_dir or Path(load_settings()["index_dir"])
    try:
        cases = load_test_set(args.test_set, args.collection)
        if not cases:
            print(f"集合 {args.collection} 在该 test set 中无匹配用例（跳过）")
            return 0
        cases = _fill_chunk_ids(cases, index_dir, args.collection)
        # 组 EvalRunner 需要的 GoldenTestCase 结构（query/expected_chunk_ids/expected_sources/reference_answer）
        from rag_mcp_server.observability.evaluation import GoldenTestCase
        test_set = [GoldenTestCase(
            query=c["query"],
            expected_chunk_ids=list(c.get("expected_chunk_ids", [])),
            expected_sources=list(c.get("expected_sources", [])),
            reference_answer=c.get("reference_answer", ""),
        ) for c in cases]
        report = EvalRunner().run(
            test_set,
            CustomEvaluator(),
            lambda query: rag_service.query(
                query,
                index_dir=index_dir,
                top_k=args.top_k,
                strategy=args.strategy,
                embedding_provider=args.embedding_provider,
                collection=args.collection,
            ),
        )
    except (OSError, RagMcpError, ValueError, KeyError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps({**report, "collection": args.collection}, ensure_ascii=False, indent=2))
    else:
        print(f"collection={args.collection}  hit_rate={report['hit_rate']:.3f}  mrr={report['mrr']:.3f}")
        for result in report["per_query"]:
            status = "PASS" if result["hit"] else "FAIL"
            print(f"  [{status}] query={result['query'][:36]} hit={result['hit']:.3f} "
                  f"rr={result['rr']:.3f} expected={result['expected_sources'] or result['expected_chunk_ids']}")

    failures = []
    if report["hit_rate"] < args.min_hit_rate:
        failures.append(f"hit_rate={report['hit_rate']:.3f} < {args.min_hit_rate}")
    if report["mrr"] < args.min_mrr:
        failures.append(f"mrr={report['mrr']:.3f} < {args.min_mrr}")
    if failures:
        print(f"评估门禁失败：{'；'.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
