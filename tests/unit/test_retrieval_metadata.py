"""Tests for RetrievalResult.metadata — image_refs passthrough (P10.6 T3).

Red/green contract (.scratch/phase10/subtickets/image-captioner/T3-retrieval-mcp.md):
- RetrievalResult gains an optional ``metadata: dict = {}`` field; existing
  positional/keyword constructions still work (test_core_types unchanged).
- Only ``image_refs`` is copied off the source record/chunk (image_refs_metadata
  helper); every other key stays behind.
- dense / sparse / hybrid retrieval seams copy image_refs; records/chunks
  without image_refs yield an empty dict (behaviour unchanged).
- the LLM reranker preserves image_refs across re-ranking (it rebuilds
  RetrievalResult objects and previously dropped metadata).
- rerank with no image_refs still works (metadata clean, ADR 0009).

Prior art: test_core_types.py (field shape), test_reranker.py (_result helper).
"""

from __future__ import annotations

import json

from rag_mcp_server.core.types import Chunk, RetrievalResult, image_refs_metadata
from rag_mcp_server.libs.llm.fake import FakeLLM
from rag_mcp_server.libs.reranker.llm_reranker import LLMReranker
from rag_mcp_server.retrieval.dense import DenseRetriever
from rag_mcp_server.retrieval.hybrid import fuse_results
from rag_mcp_server.retrieval.sparse_bm25 import SparseBM25


# ---------------------------------------------------------------------------
# image_refs_metadata — the "only copy image_refs" rule
# ---------------------------------------------------------------------------
def test_image_refs_metadata_copies_only_image_refs():
    meta = image_refs_metadata(
        {"image_refs": ["a_p1_1", "a_p1_2"], "content_hash": "x", "title": "t"}
    )
    assert meta == {"image_refs": ["a_p1_1", "a_p1_2"]}


def test_image_refs_metadata_missing_refs_yields_empty():
    assert image_refs_metadata({"content_hash": "x"}) == {}
    assert image_refs_metadata({}) == {}


def test_image_refs_metadata_copies_a_copy():
    src = {"image_refs": ["a_p1_1"]}
    meta = image_refs_metadata(src)
    meta["image_refs"].append("b")
    assert src["image_refs"] == ["a_p1_1"]


# ---------------------------------------------------------------------------
# RetrievalResult field default
# ---------------------------------------------------------------------------
def test_retrieval_result_metadata_defaults_empty():
    r = RetrievalResult(chunk_id="d::0", source_path="p", score=0.5, text="t")
    assert r.metadata == {}


# ---------------------------------------------------------------------------
# dense seam — copies image_refs off the record
# ---------------------------------------------------------------------------
class _FakeRecord:
    def __init__(self, chunk_id, text, metadata, score=0.7):
        self.chunk_id = chunk_id
        self.text = text
        self.metadata = metadata
        self.score = score


def test_dense_copies_image_refs():
    store = _FakeStore([_FakeRecord("d::0", "带图", {"image_refs": ["d_p1_1"]})])
    out = DenseRetriever(store).search([0.1, 0.2, 0.3], top_k=3)
    assert out[0].metadata == {"image_refs": ["d_p1_1"]}


def test_dense_without_image_refs_yields_empty_metadata():
    store = _FakeStore([_FakeRecord("d::0", "无图", {"source_path": "p.md"})])
    out = DenseRetriever(store).search([0.1, 0.2, 0.3], top_k=3)
    assert out[0].metadata == {}


class _FakeStore:
    def __init__(self, records):
        self._records = records

    def query(self, embedding, top_k, where=None):
        if where is not None:
            # minimal metadata filter: keep records whose metadata matches
            # every key of `where` (list fields via $in)
            out = []
            for r in self._records:
                meta = r.metadata
                ok = True
                for key, val in where.items():
                    if isinstance(val, dict) and "$in" in val:
                        if meta.get(key) not in val["$in"]:
                            ok = False
                            break
                    elif meta.get(key) != val:
                        ok = False
                        break
                if ok:
                    out.append(r)
            return out[:top_k]
        return self._records[:top_k]


# ---------------------------------------------------------------------------
# sparse seam — copies image_refs off the chunk
# ---------------------------------------------------------------------------
def test_sparse_copies_image_refs():
    chunks = [
        Chunk(
            document_id="d",
            text="架构图 [IMAGE: d_p1_1] 说明",
            source_path="p.md",
            order=0,
            metadata={"image_refs": ["d_p1_1"], "file_type": "pdf"},
        ),
        Chunk(
            document_id="d",
            text="普通文本说明",
            source_path="p.md",
            order=1,
            metadata={"file_type": "pdf"},
        ),
    ]
    out = SparseBM25(chunks).search("说明", top_k=3)
    by_id = {r.chunk_id: r for r in out}
    assert by_id["d::0"].metadata == {"image_refs": ["d_p1_1"]}
    assert by_id["d::1"].metadata == {}  # no refs -> empty


# ---------------------------------------------------------------------------
# hybrid fuse — carries image_refs from the fused source
# ---------------------------------------------------------------------------
def test_fuse_results_carries_image_refs():
    dense = [RetrievalResult("d::0", "p", 0.8, "带图", ["dense"], {"image_refs": ["d_p1_1"]})]
    sparse = [RetrievalResult("d::0", "p", 0.6, "带图", ["sparse"], {})]
    out = fuse_results(dense, sparse, top_k=3)
    assert out[0].chunk_id == "d::0"
    assert out[0].metadata == {"image_refs": ["d_p1_1"]}


def test_fuse_results_no_refs_yields_empty():
    dense = [RetrievalResult("d::0", "p", 0.8, "无图", ["dense"])]
    sparse = [RetrievalResult("d::0", "p", 0.6, "无图", ["sparse"])]
    out = fuse_results(dense, sparse, top_k=3)
    assert out[0].metadata == {}


# ---------------------------------------------------------------------------
# reranker — preserves image_refs across re-ranking
# ---------------------------------------------------------------------------
def _scored_result(chunk_id, score, image_refs=None):
    metadata = {"image_refs": image_refs} if image_refs else {}
    return RetrievalResult(
        chunk_id=chunk_id,
        source_path=f"docs/{chunk_id}.md",
        score=score,
        text=f"text of {chunk_id}",
        sources=["dense"],
        metadata=metadata,
    )


def _chatting_llm(payload: str) -> FakeLLM:
    """A FakeLLM whose _chat always returns the canned payload (test_reranker)."""
    llm = FakeLLM()
    llm._chat = lambda messages: payload  # type: ignore[method-assign]
    return llm


def test_llm_rerank_preserves_image_refs():
    payload = json.dumps(
        [
            {"passage_id": "a", "score": 1, "reasoning": "r"},
            {"passage_id": "b", "score": 3, "reasoning": "r"},
        ]
    )
    results = [
        _scored_result("a", 0.5, image_refs=["a_p1_1"]),
        _scored_result("b", 0.4, image_refs=["b_p1_2", "b_p1_3"]),
    ]
    out = LLMReranker(_chatting_llm(payload)).rerank("q", results, top_k=2)
    assert [r.chunk_id for r in out] == ["b", "a"]
    assert out[0].metadata == {"image_refs": ["b_p1_2", "b_p1_3"]}
    assert out[1].metadata == {"image_refs": ["a_p1_1"]}


def test_llm_rerank_without_image_refs_keeps_clean_metadata():
    payload = json.dumps([{"passage_id": "a", "score": 3, "reasoning": "r"}])
    out = LLMReranker(_chatting_llm(payload)).rerank("q", [_scored_result("a", 0.5)], top_k=1)
    assert out[0].metadata == {}  # ADR 0009: metadata stays clean


def test_llm_rerank_does_not_mutate_input_metadata():
    payload = json.dumps([{"passage_id": "a", "score": 3, "reasoning": "r"}])
    original = _scored_result("a", 0.5, image_refs=["a_p1_1"])
    LLMReranker(_chatting_llm(payload)).rerank("q", [original], top_k=1)
    assert original.metadata == {"image_refs": ["a_p1_1"]}


def test_llm_rerank_filters_non_image_metadata():
    payload = json.dumps([{"passage_id": "a", "score": 3, "reasoning": "r"}])
    original = _scored_result("a", 0.5, image_refs=["a_p1_1"])
    original.metadata["content_hash"] = "leak"
    out = LLMReranker(_chatting_llm(payload)).rerank("q", [original], top_k=1)
    assert out[0].metadata == {"image_refs": ["a_p1_1"]}
