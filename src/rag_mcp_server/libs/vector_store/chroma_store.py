"""ChromaVectorStore — embedded Chroma backed by a configurable persist dir.

Talks to the Chroma Python client. A persist_dir of None creates an ephemeral
in-memory store (unit-test convenience); a real path persists to disk so a
later instance can reopen the same collection (get_or_create). Metric is
cosine; `query` similarity is 1/(1+distance), returned descending.

See docs/spec-p9-vstore.md and ADR 0004.
"""

from __future__ import annotations

from typing import Any

import chromadb

from rag_mcp_server.core.errors import VectorStoreError
from rag_mcp_server.core.validation import validate_positive_int
from rag_mcp_server.libs.vector_store.base import (
    REQUIRED_METADATA_KEYS,
    BaseVectorStore,
    VectorRecord,
)

DEFAULT_COLLECTION = "rag_chunks"


def _sim_from_distance(distance: float) -> float:
    """Map Chroma distance to a similarity in (0, 1], larger is more relevant."""
    return 1.0 / (1.0 + distance)


class ChromaVectorStore(BaseVectorStore):
    """Embedded Chroma vector store.

    `persist_dir`: None -> ephemeral in-memory; a path -> persistent, so a
    fresh instance on the same dir can reopen the same collection.
    """

    def __init__(self, persist_dir: str | None, collection: str = DEFAULT_COLLECTION) -> None:
        if not collection or not isinstance(collection, str) or not collection.strip():
            raise ValueError("collection 不能为空")
        # persist_dir="" would silently write Chroma's HNSW bin files to the
        # current working directory; treat any non-None value as requiring a
        # non-empty string (None -> ephemeral in-memory).
        if persist_dir is not None and (
            not isinstance(persist_dir, str) or not persist_dir.strip()
        ):
            raise ValueError("persist_dir 不能为空字符串")
        self.collection = collection
        self.persist_dir = persist_dir
        client = (
            chromadb.PersistentClient(path=persist_dir)
            if persist_dir is not None
            else chromadb.EphemeralClient()
        )
        self._client = client
        self._collection = client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )

    # -- BaseVectorStore ----------------------------------------------------

    def add(self, records: list[VectorRecord], embeddings: list[list[float]]) -> None:
        if len(records) != len(embeddings):
            raise ValueError(
                f"records 与 embeddings 行数必须一致：{len(records)} vs {len(embeddings)}"
            )
        if not records:
            return
        for record in records:
            if not isinstance(record.metadata, dict):
                raise VectorStoreError(
                    f"record {record.chunk_id!r} 的 metadata 必须是 dict，得到 {type(record.metadata).__name__}"
                )
            missing = [key for key in REQUIRED_METADATA_KEYS if key not in record.metadata]
            if missing:
                raise VectorStoreError(
                    f"record {record.chunk_id!r} 缺少必填 metadata 键：{', '.join(missing)}"
                )
        try:
            self._collection.add(
                ids=[r.chunk_id for r in records],
                embeddings=embeddings,
                documents=[r.text for r in records],
                metadatas=[r.metadata for r in records],
            )
        except Exception as exc:  # Chroma wraps backend errors
            raise VectorStoreError(f"Chroma add 失败：{exc}") from exc

    def query(
        self,
        query_embedding: list[float],
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> list[VectorRecord]:
        validate_positive_int("top_k", top_k)
        if not query_embedding:
            return []  # empty input short-circuits to an empty result (spec)
        try:
            # clamp to the collection size so "top_k larger than the store"
            # always returns everything, regardless of backend behaviour;
            # n_results must be >= 1, and an empty store yields no results.
            n_results = max(1, min(top_k, self._collection.count()))
            response = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
            )
        except Exception as exc:
            raise VectorStoreError(f"Chroma query 失败：{exc}") from exc

        ids: list[list[str]] = response.get("ids", [[]])
        metadatas: list[list[dict]] = response.get("metadatas", [[]])
        documents: list[list[str]] = response.get("documents", [[]])
        distances: list[list[float]] = response.get("distances", [[]])

        first_ids = ids[0] if ids else []
        first_metas = metadatas[0] if metadatas else []
        first_docs = documents[0] if documents else []
        first_dists = distances[0] if distances else []

        results: list[VectorRecord] = []
        for i, chunk_id in enumerate(first_ids):
            results.append(
                VectorRecord(
                    chunk_id=chunk_id,
                    metadata=first_metas[i] if i < len(first_metas) else {},
                    text=first_docs[i] if i < len(first_docs) else "",
                    score=_sim_from_distance(first_dists[i]) if i < len(first_dists) else None,
                )
            )
        return results

    def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        try:
            self._collection.delete(ids=ids)
        except Exception as exc:
            raise VectorStoreError(f"Chroma delete 失败：{exc}") from exc

    def get_by_ids(self, ids: list[str]) -> list[VectorRecord]:
        if not ids:
            return []
        try:
            response = self._collection.get(
                ids=ids,
                include=["metadatas", "documents"],
            )
        except Exception as exc:
            raise VectorStoreError(f"Chroma get 失败：{exc}") from exc

        raw_ids: list[str] = response.get("ids", [])
        metadatas: list[dict] = response.get("metadatas", [])
        documents: list[str] = response.get("documents", [])
        by_id: dict[str, VectorRecord] = {}
        for i, id_ in enumerate(raw_ids):
            by_id[id_] = VectorRecord(
                chunk_id=id_,
                metadata=metadatas[i] if i < len(metadatas) and isinstance(metadatas[i], dict) else {},
                text=documents[i] if i < len(documents) else "",
                score=None,
            )
        # Chroma does not guarantee request order; reorder by the requested ids.
        return [by_id[id_] for id_ in ids if id_ in by_id]

    def get_all(self, where: dict[str, Any] | None = None) -> list[VectorRecord]:
        try:
            response = self._collection.get(
                where=where,
                include=["metadatas", "documents"],
            )
        except Exception as exc:
            raise VectorStoreError(f"Chroma get_all 失败：{exc}") from exc

        raw_ids: list[str] = response.get("ids", [])
        metadatas: list[dict] = response.get("metadatas", [])
        documents: list[str] = response.get("documents", [])
        results: list[VectorRecord] = []
        for i, id_ in enumerate(raw_ids):
            results.append(
                VectorRecord(
                    chunk_id=id_,
                    metadata=metadatas[i] if i < len(metadatas) and isinstance(metadatas[i], dict) else {},
                    text=documents[i] if i < len(documents) else "",
                    score=None,
                )
            )
        return results
