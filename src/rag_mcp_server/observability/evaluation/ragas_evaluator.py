"""RagasEvaluator — optional Answer-scoring evaluator (P10.7 T3).

The default P10.7 gate is Retrieval-first: CustomEvaluator scores hit_rate/mrr
against golden chunk ids (ADR 0013). RagasEvaluator is an optional extension
for "did the retrieved content actually answer the query" — LLM-scored scalars
(faithfulness / answer relevancy) via the injected BaseLLM seam.

Design (t3-issues/01-ragas-evaluator.md):
- scoring backend is injectable. Default RagasScoringBackend lazily imports
  ragas (find_spec -> ImportError with a clear message when missing, because
  the module must stay importable in environments without ragas). Tests inject
  a mock backend so scoring logic is verifiable without ragas installed.
- evaluate(): an injected backend wins (backend.score); without one it guards
  on RagasScoringBackend() first (missing ragas -> ImportError), then falls
  back to score_with_llm — so installing ragas but not injecting a backend
  yields scalars, never NotImplementedError.
- score_with_llm is the thin LLM scoring path: query + reference + golden
  chunk ids -> scalars parsed from the LLM's JSON reply and clamped to [0,1].
  Bad output raises ValueError; LLM errors bubble unchanged (no gate fallback
  — this is opt-in).

See docs/adr/0013-evaluation-retrieval-gate.md and CONTEXT.md "Ragas 评估扩展".
"""

from __future__ import annotations

import importlib.util
import json

from rag_mcp_server.core.types import RetrievalResult
from rag_mcp_server.libs.evaluator.base import BaseEvaluator
from rag_mcp_server.libs.llm.base import BaseLLM

__all__ = ["RagasEvaluator", "RagasScoringBackend"]

RAGAS_INSTALL_HINT = (
    "ragas 依赖未安装。RagasEvaluator 是可选的 Answer 评估扩展，默认 Retrieval 门禁不依赖它。"
    "安装方式：python -m pip install ragas（见 docs/adr/0013-evaluation-retrieval-gate.md）。"
)

DEFAULT_SCORING_PROMPT = """You are evaluating whether a RAG answer to a query is grounded in the given reference answer.

Score the retrieved answer for:
- faithfulness: how factually consistent the answer is with the reference answer (0 to 1)
- answer_relevancy: how well the answer addresses the query (0 to 1)

Respond with only a JSON object, for example: {"faithfulness": 0.9, "answer_relevancy": 0.8}"""


class RagasScoringBackend:
    """Lazy ragas wrapper: score(query, retrieved, expected, reference) -> dict[str, float]."""

    def __init__(self) -> None:
        if importlib.util.find_spec("ragas") is None:
            raise ImportError(RAGAS_INSTALL_HINT)

    def score(
        self,
        query: str,
        retrieved: list[RetrievalResult],
        expected_chunk_ids: list[str],
        reference_answer: str,
    ) -> dict[str, float]:
        raise NotImplementedError(
            "真实 ragas 编排（dataset/pandas/API）不在 P10.7 第一版范围，见 t3-issues/01-ragas-evaluator.md"
        )


class RagasEvaluator(BaseEvaluator):
    """Score retrieval results with LLM Answer metrics (optional, not the default gate)."""

    def __init__(
        self,
        llm: BaseLLM,
        scoring_backend: RagasScoringBackend | None = None,
        prompt_template: str = DEFAULT_SCORING_PROMPT,
    ) -> None:
        self._llm = llm
        self._scoring_backend = scoring_backend
        self._prompt_template = prompt_template

    def evaluate(
        self,
        query: str,
        retrieved: list[RetrievalResult],
        expected_chunk_ids: list[str],
        reference_answer: str = "",
    ) -> dict[str, float]:
        if self._scoring_backend is not None:
            return self._scoring_backend.score(query, retrieved, expected_chunk_ids, reference_answer)
        # No injected backend: the real ragas backend is the default scoring
        # path, so its missing-dependency guard still applies before we fall
        # back to the thin LLM seam. Installing ragas but not injecting a
        # backend -> scalars via score_with_llm (not NotImplementedError).
        RagasScoringBackend()
        return self.score_with_llm(query, retrieved, expected_chunk_ids, reference_answer)

    # -- thin LLM scoring path ----------------------------------------------
    def score_with_llm(
        self,
        query: str,
        retrieved: list[RetrievalResult],
        expected_chunk_ids: list[str],
        reference_answer: str,
    ) -> dict[str, float]:
        """Score one query against an LLM without the ragas backend (thin, testable)."""
        prompt = self._build_prompt(query, retrieved, expected_chunk_ids, reference_answer)
        raw = self._llm.chat([{"role": "user", "content": prompt}])
        return self._parse(raw)

    def _build_prompt(
        self,
        query: str,
        retrieved: list[RetrievalResult],
        expected_chunk_ids: list[str],
        reference_answer: str,
    ) -> str:
        contexts = "\n".join(f"Chunk: {result.text}" for result in retrieved)
        golden = ", ".join(expected_chunk_ids) if expected_chunk_ids else "无"
        return (
            f"{self._prompt_template}\n\nQuery: {query}\n\nReference answer: {reference_answer}\n\n"
            f"Golden chunks: {golden}\n\n"
            f"Retrieved chunks:\n{contexts}"
        )

    def _parse(self, raw: str) -> dict[str, float]:
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"LLM 判分输出不是有效 JSON：{exc}") from exc

        if not isinstance(parsed, dict):
            raise ValueError(f"LLM 判分输出必须是对象（实际 {type(parsed).__name__}）")
        if not parsed:
            raise ValueError("LLM 判分输出至少需要一个指标")
        metrics: dict[str, float] = {}
        for name, value in parsed.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"LLM 判分输出 {name!r} 必须是数值（实际 {type(value).__name__}）")
            metrics[name] = min(1.0, max(0.0, float(value)))
        return metrics
