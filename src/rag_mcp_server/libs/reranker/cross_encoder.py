"""CrossEncoderReranker — local cross-encoder model for passage re-ranking.

Uses sentence-transformers' CrossEncoder on (query, passage) pairs; score is
the model's raw logit (larger = more relevant). Backend init/loading failure
raises RerankError so the orchestrator (QueryReranker) falls back to the
original order (ADR 0009 "fail -> keep original order"). The model is loaded
lazily on the first ``rerank`` call and cached on the instance, so constructing
a reranker (or the factory returning one) never touches sentence-transformers —
it stays an optional dependency.

``scorer`` is an injection seam for unit tests: a plain callable taking the
(query, passage) pairs and returning a list of scores. When None, the real
model's ``predict`` is used. The reranked RetrievalResult keeps
chunk_id/text/source_path/sources and gets score = the model's score; metadata
stays clean via image_refs_metadata (P10.6 T3 / ADR 0009 pt.3).

See docs/spec-p10-rerank.md, docs/implementation-plans.md (方案 6) and ADR 0009.
"""

from __future__ import annotations

from rag_mcp_server.core.errors import RerankError
from rag_mcp_server.core.types import RetrievalResult, image_refs_metadata
from rag_mcp_server.core.validation import validate_positive_int
from rag_mcp_server.libs.reranker.base import BaseReranker

DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker(BaseReranker):
    """Rerank candidates by a local CrossEncoder scoring (query, passage) pairs."""

    def __init__(
        self,
        model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
        max_length: int = 512,
        scorer=None,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        # Injection seam for unit tests: a callable(pairs) -> list[float]. When
        # None, the real sentence-transformers CrossEncoder is loaded lazily.
        self._scorer = scorer
        self._model = None  # lazy, cached on first rerank

    def _load_model(self):
        """Load (once) and cache the sentence-transformers CrossEncoder.

        Raises RerankError on any failure (missing dependency, download error,
        invalid model path) so the orchestrator falls back to the original order.
        """
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name, max_length=self.max_length)
            except Exception as exc:
                raise RerankError(f"CrossEncoder 模型加载失败：{exc}") from exc
        return self._model

    def _score_pairs(self, pairs: list[tuple[str, str]]):
        """Score (query, passage) pairs via the injected scorer or the model."""
        if self._scorer is not None:
            return self._scorer(pairs)
        return self._load_model().predict(pairs)

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        validate_positive_int("top_k", top_k)
        if not results:
            return []

        pairs = [(query, r.text) for r in results]
        try:
            scores = self._score_pairs(pairs)
            # The scorer / model output shape is not under our control: a list
            # with the wrong length or non-numeric entries must surface as a
            # RerankError (orchestrator falls back to the original order), never
            # a bare IndexError/ValueError/TypeError (review FIX — the sort and
            # result construction are inside the guarded region).
            if len(scores) != len(pairs):
                raise RerankError(
                    f"CrossEncoder 评分数量不匹配：期望 {len(pairs)}，实际 {len(scores)}"
                )
            order = sorted(
                range(len(results)),
                key=lambda i: float(scores[i]),
                reverse=True,
            )[:top_k]
            return [
                RetrievalResult(
                    chunk_id=results[i].chunk_id,
                    source_path=results[i].source_path,
                    score=float(scores[i]),
                    text=results[i].text,
                    sources=list(results[i].sources),
                    metadata=image_refs_metadata(results[i].metadata),
                )
                for i in order
            ]
        except RerankError:
            raise
        except Exception as exc:
            raise RerankError(f"CrossEncoder 评分失败：{exc}") from exc
