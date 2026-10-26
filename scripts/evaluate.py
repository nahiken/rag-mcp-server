"""Run the offline Retrieval evaluation gate against an existing index."""

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
from rag_mcp_server.observability.evaluation import EvalRunner, load_golden_test_set

DEFAULT_TEST_SET = Path("tests/fixtures/golden_test_set.json")
_RETRIEVAL_STRATEGIES = ("dense", "sparse", "hybrid")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行离线 Retrieval 评估门禁")
    parser.add_argument("--test-set", type=Path, default=DEFAULT_TEST_SET)
    parser.add_argument("--index-dir", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--strategy", choices=_RETRIEVAL_STRATEGIES, default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--min-hit-rate", type=float, default=1.0)
    parser.add_argument("--min-mrr", type=float, default=0.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        index_dir = args.index_dir or Path(load_settings()["index_dir"])
        test_set = load_golden_test_set(args.test_set)
        report = EvalRunner().run(
            test_set,
            CustomEvaluator(),
            lambda query: rag_service.query(
                query,
                index_dir=index_dir,
                top_k=args.top_k,
                strategy=args.strategy,
            ),
        )
    except (OSError, RagMcpError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1

    _print_report(report, as_json=args.json)
    failures = _threshold_failures(report, args.min_hit_rate, args.min_mrr)
    if failures:
        print(f"评估门禁失败：{'；'.join(failures)}", file=sys.stderr)
        return 1
    return 0


def _print_report(report: dict, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    print(f"hit_rate={report['hit_rate']:.3f}  mrr={report['mrr']:.3f}")
    for result in report["per_query"]:
        status = "PASS" if result["hit"] else "FAIL"
        retrieved_ids = ", ".join(result["retrieved_chunk_ids"]) or "-"
        print(
            f"[{status}] query={result['query']} hit={result['hit']:.3f} "
            f"rr={result['rr']:.3f} retrieved=[{retrieved_ids}]"
        )
        if not result["hit"]:
            print(f"       expected_sources={', '.join(result['expected_sources']) or '-'}")


def _threshold_failures(report: dict, min_hit_rate: float, min_mrr: float) -> list[str]:
    failures = []
    if report["hit_rate"] < min_hit_rate:
        failures.append(f"hit_rate={report['hit_rate']:.3f} < min_hit_rate={min_hit_rate:.3f}")
    if report["mrr"] < min_mrr:
        failures.append(f"mrr={report['mrr']:.3f} < min_mrr={min_mrr:.3f}")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
