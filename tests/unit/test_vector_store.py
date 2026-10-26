"""Tests for the BaseVectorStore seam — ChromaVectorStore external behaviour.

Red/green contract (docs/spec-p9-vstore.md, task 4, tickets 04-T1/T2/T3):
- add(records, embeddings) writes rows aligned by position; metadata must carry
  the four keys document_id/source_path/chunk_id/content_hash
- query returns nearest neighbours sorted by similarity descending
  (score = 1/(1+distance)); top_k <= 0 fails fast
- delete(ids) is idempotent (missing ids are silent), empty list is a no-op
- get_by_ids returns present ids in request order, skips missing, empty -> []
- empty inputs short-circuit to empty results, never error
- persist_dir None -> ephemeral in-memory; tmp_path -> reopen readable
- factory create_vector_store instantiates chroma and rejects unknown provider

The only test seam is BaseVectorStore: all tests drive ChromaVectorStore
through it, mirroring the BaseEmbedding single-seam pattern.
"""

from __future__ import annotations

import pytest

from rag_mcp_server.core.errors import VectorStoreError
from rag_mcp_server.libs.vector_store.base import BaseVectorStore, VectorRecord
from rag_mcp_server.libs.vector_store.chroma_store import ChromaVectorStore
from rag_mcp_server.libs.vector_store.factory import create_vector_store


def _records(n: int = 3, dim: int = 4) -> tuple[list[VectorRecord], list[list[float]]]:
    """Build n row-aligned records + embeddings with distinct-ish vectors.

    Embedding i is a one-hot-ish vector dominated by coordinate i (when dim=n),
    so queries for a basis vector rank its own record first deterministically.
    """
    records = [
        VectorRecord(
            chunk_id=f"doc::{i}",
            metadata={
                "document_id": "doc",
                "source_path": f"doc.txt",
                "chunk_id": f"doc::{i}",
                "content_hash": f"hash-{i}",
            },
            text=f"text-{i}",
        )
        for i in range(n)
    ]
    embeddings: list[list[float]] = []
    for i in range(n):
        vec = [0.0] * dim
        if i < dim:
            vec[i] = 1.0
        else:
            vec[0] = 1.0
        embeddings.append(vec)
    return records, embeddings


@pytest.fixture
def store(tmp_path):
    """Default store backed by a real on-disk persist_dir (tmp_path)."""
    return ChromaVectorStore(persist_dir=str(tmp_path), collection="test_chunks")


@pytest.fixture
def ephemeral_store(request):
    """In-memory store for tests that do not need persistence.

    Chroma's EphemeralClient is a process-global in-memory store, so a fixed
    collection name would leak records between tests. A collection name unique
    to each test keeps them isolated.
    """
    return ChromaVectorStore(
        persist_dir=None, collection=f"test_{request.node.name}"
    )


# ---------------------------------------------------------------------------
# 04-T1 · search loop: add + query + ordering + score
# ---------------------------------------------------------------------------
def test_query_returns_similarity_sorted_nearest_neighbours(ephemeral_store):
    records, embeddings = _records()
    ephemeral_store.add(records, embeddings)
    results = ephemeral_store.query([1.0, 0.0, 0.0, 0.0], top_k=3)
    assert [r.chunk_id for r in results] == ["doc::0", "doc::1", "doc::2"]
    # similarity must be descending (most relevant first)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert all(s is not None and isinstance(s, float) for s in scores)


def test_query_score_is_one_over_one_plus_distance(ephemeral_store):
    # identical embedding -> distance 0 -> score = 1/(1+0) = 1.0
    records, embeddings = _records(n=1)
    ephemeral_store.add(records, embeddings)
    results = ephemeral_store.query(embeddings[0], top_k=1)
    assert results[0].chunk_id == "doc::0"
    assert results[0].score == pytest.approx(1.0, rel=1e-5)


def test_query_result_carries_text_and_metadata(ephemeral_store):
    records, embeddings = _records(n=1)
    ephemeral_store.add(records, embeddings)
    (result,) = ephemeral_store.query(embeddings[0], top_k=1)
    assert result.text == "text-0"
    assert result.metadata["document_id"] == "doc"
    assert result.metadata["source_path"] == "doc.txt"
    assert result.metadata["content_hash"] == "hash-0"


def test_query_top_k_bigger_than_collection_returns_all(ephemeral_store):
    records, embeddings = _records(n=2)
    ephemeral_store.add(records, embeddings)
    results = ephemeral_store.query([1.0, 0.0, 0.0, 0.0], top_k=100)
    assert {r.chunk_id for r in results} == {"doc::0", "doc::1"}
    assert len(results) == 2


def test_query_top_k_non_positive_fails_fast(ephemeral_store):
    records, embeddings = _records(n=1)
    ephemeral_store.add(records, embeddings)
    with pytest.raises(ValueError):
        ephemeral_store.query([1.0, 0.0, 0.0, 0.0], top_k=0)
    with pytest.raises(ValueError):
        ephemeral_store.query([1.0, 0.0, 0.0, 0.0], top_k=-1)


def test_add_requires_four_metadata_keys(ephemeral_store):
    record = VectorRecord(
        chunk_id="doc::0",
        metadata={"document_id": "doc", "source_path": "doc.txt"},  # missing two keys
        text="text",
    )
    with pytest.raises(VectorStoreError):
        ephemeral_store.add([record], [[1.0, 0.0, 0.0, 0.0]])


def test_add_rejects_non_dict_metadata(ephemeral_store):
    # metadata must be a dict, not a bare None/str/list (P1)
    for bad in (None, "doc::0", ["k"]):
        record = VectorRecord(chunk_id="doc::0", metadata=bad, text="text")
        with pytest.raises(VectorStoreError):
            ephemeral_store.add([record], [[1.0, 0.0, 0.0, 0.0]])


def test_persist_dir_empty_string_fails_fast():
    # "" would silently write HNSW bin files to the cwd (P2); fail fast instead
    with pytest.raises(ValueError):
        ChromaVectorStore(persist_dir="", collection="test_chunks")


def test_add_rows_must_align(ephemeral_store):
    records, embeddings = _records(n=2)
    with pytest.raises(ValueError):
        ephemeral_store.add(records, embeddings[:1])


def test_empty_add_is_noop(ephemeral_store):
    ephemeral_store.add([], [])
    assert ephemeral_store.query([1.0, 0.0, 0.0, 0.0], top_k=1) == []


def test_query_empty_embedding_returns_empty(ephemeral_store):
    # empty query embedding short-circuits to [] rather than hitting the backend
    records, embeddings = _records(n=1)
    ephemeral_store.add(records, embeddings)
    assert ephemeral_store.query([], top_k=1) == []


def test_base_vector_store_is_abstract():
    with pytest.raises(TypeError):
        BaseVectorStore()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# 04-T2 · lifecycle: delete + get_by_ids
# ---------------------------------------------------------------------------
def test_delete_removes_vector_and_query_no_longer_returns_it(ephemeral_store):
    records, embeddings = _records(n=2)
    ephemeral_store.add(records, embeddings)
    ephemeral_store.delete(["doc::0"])
    results = ephemeral_store.query([1.0, 0.0, 0.0, 0.0], top_k=2)
    assert "doc::0" not in {r.chunk_id for r in results}
    assert "doc::1" in {r.chunk_id for r in results}


def test_delete_missing_id_is_idempotent(ephemeral_store):
    records, embeddings = _records(n=1)
    ephemeral_store.add(records, embeddings)
    ephemeral_store.delete(["does-not-exist"])  # silent
    assert len(ephemeral_store.get_by_ids(["doc::0"])) == 1


def test_delete_empty_list_is_noop(ephemeral_store):
    records, embeddings = _records(n=1)
    ephemeral_store.add(records, embeddings)
    ephemeral_store.delete([])
    assert len(ephemeral_store.get_by_ids(["doc::0"])) == 1


def test_get_by_ids_returns_present_ids_in_request_order(ephemeral_store):
    records, embeddings = _records(n=3)
    ephemeral_store.add(records, embeddings)
    results = ephemeral_store.get_by_ids(["doc::2", "doc::0", "doc::1"])
    assert [r.chunk_id for r in results] == ["doc::2", "doc::0", "doc::1"]


def test_get_by_ids_skips_missing_and_empty_returns_empty(ephemeral_store):
    records, embeddings = _records(n=2)
    ephemeral_store.add(records, embeddings)
    results = ephemeral_store.get_by_ids(["doc::0", "missing", "doc::1"])
    assert [r.chunk_id for r in results] == ["doc::0", "doc::1"]
    assert ephemeral_store.get_by_ids([]) == []


def test_query_where_filters_by_metadata(ephemeral_store):
    # where is passed through to Chroma: only records whose metadata matches
    # are returned (spec: where 透传给 Chroma)
    records, embeddings = _records(n=2)
    ephemeral_store.add(records, embeddings)
    results = ephemeral_store.query(
        [1.0, 0.0, 0.0, 0.0], top_k=5, where={"chunk_id": "doc::1"}
    )
    assert [r.chunk_id for r in results] == ["doc::1"]


# ---------------------------------------------------------------------------
# 04-T3 · persistence + factory
# ---------------------------------------------------------------------------
def test_persist_dir_reopen_reads_back(tmp_path):
    store1 = ChromaVectorStore(persist_dir=str(tmp_path), collection="test_chunks")
    records, embeddings = _records(n=1)
    store1.add(records, embeddings)
    # reopen a new instance on the same persist_dir
    store2 = ChromaVectorStore(persist_dir=str(tmp_path), collection="test_chunks")
    results = store2.get_by_ids(["doc::0"])
    assert len(results) == 1
    assert results[0].text == "text-0"
    assert results[0].score is None  # get_by_ids has no similarity context


def test_ephemeral_store_is_isolated_from_disk(tmp_path):
    ephemeral = ChromaVectorStore(persist_dir=None, collection="test_ephemeral_isolated")
    records, embeddings = _records(n=1)
    ephemeral.add(records, embeddings)
    fresh = ChromaVectorStore(persist_dir=str(tmp_path), collection="test_chunks")
    assert fresh.get_by_ids(["doc::0"]) == []


def test_factory_creates_chroma(tmp_path):
    store = create_vector_store(
        provider="chroma", persist_dir=str(tmp_path), collection="test_chunks"
    )
    assert isinstance(store, ChromaVectorStore)


def test_factory_rejects_unknown_provider():
    with pytest.raises(ValueError):
        create_vector_store(provider="not-a-provider", persist_dir=None, collection="chunks")


def test_factory_accepts_none_persist_dir():
    store = create_vector_store(provider="chroma", persist_dir=None, collection="chunks")
    assert isinstance(store, ChromaVectorStore)


# ---------------------------------------------------------------------------
# 08-T1 · get_all: full store read (sparse corpus rebuild, ADR 0003)
# ---------------------------------------------------------------------------
def test_get_all_returns_every_record(ephemeral_store):
    records, embeddings = _records(n=3)
    ephemeral_store.add(records, embeddings)
    results = ephemeral_store.get_all()
    assert {r.chunk_id for r in results} == {"doc::0", "doc::1", "doc::2"}
    # fields are carried through (text + metadata), score is None (no query)
    by_id = {r.chunk_id: r for r in results}
    for i in range(3):
        assert by_id[f"doc::{i}"].text == f"text-{i}"
        assert by_id[f"doc::{i}"].metadata["source_path"] == "doc.txt"
        assert by_id[f"doc::{i}"].score is None


def test_get_all_empty_store_returns_empty_list(ephemeral_store):
    assert ephemeral_store.get_all() == []


def test_get_all_reflects_deletes(ephemeral_store):
    records, embeddings = _records(n=2)
    ephemeral_store.add(records, embeddings)
    ephemeral_store.delete(["doc::0"])
    assert {r.chunk_id for r in ephemeral_store.get_all()} == {"doc::1"}


def test_get_all_reads_back_after_reopen(tmp_path):
    store1 = ChromaVectorStore(persist_dir=str(tmp_path), collection="test_chunks")
    records, embeddings = _records(n=2)
    store1.add(records, embeddings)
    store2 = ChromaVectorStore(persist_dir=str(tmp_path), collection="test_chunks")
    assert {r.chunk_id for r in store2.get_all()} == {"doc::0", "doc::1"}
