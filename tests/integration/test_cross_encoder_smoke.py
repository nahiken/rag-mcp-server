"""Integration smoke: real sentence-transformers CrossEncoder (offline-capable).

Downloads (or loads locally) cross-encoder/ms-marco-MiniLM-L-6-v2 and runs one
real scoring pass to prove the lazy-load -> predict path works end to end. This
is a SLOW test: it downloads ~80MB on first run and is excluded from default
pytest runs (`-m 'not integration'`); run it explicitly when
sentence-transformers is installed and the model is cached locally.

    pip install sentence-transformers
    python -m pytest tests/integration/test_cross_encoder_smoke.py -m slow -q
"""

from __future__ import annotations

import pytest

from rag_mcp_server.core.types import RetrievalResult
from rag_mcp_server.libs.reranker.cross_encoder import CrossEncoderReranker

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
]

pytest.importorskip("sentence_transformers")


def _result(chunk_id: str) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        source_path=f"docs/{chunk_id}.md",
        score=0.0,
        text=f"text of {chunk_id}",
        sources=["dense"],
    )


def test_real_cross_encoder_smoke():
    rr = CrossEncoderReranker()
    results = [_result("a"), _result("b")]
    out = rr.rerank("what is rag", results, top_k=2)

    assert len(out) == 2
    assert {r.chunk_id for r in out} == {"a", "b"}  # same candidates back
    assert rr._model is not None  # the real model loaded lazily and was cached
    assert all(isinstance(r.score, float) for r in out)
