"""Document lifecycle management — list/detail/delete/stats.

The single source of truth is the vector store: every method aggregates from
``store.get_all()`` by ``metadata["document_id"]`` (CONTEXT.md "文档生命周期").
delete clears only Chroma — sparse (BM25) is rebuilt per query from the store
(ADR 0003), so it is automatically consistent after a delete; no separate index
to maintain. chunks.jsonl evidence is left untouched by default (it records what
was ingested, not the post-delete state); when an ``evidence_dir`` is injected
(ADR 0021 persisted sparse), delete also rewrites the snapshot minus the deleted
document so a persisted corpus never scores deleted chunks.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

from rag_mcp_server.ingestion.storage.image_storage import ImageStorage
from rag_mcp_server.libs.loader.file_integrity import FileIntegrityChecker
from rag_mcp_server.libs.vector_store.base import BaseVectorStore, VectorRecord

logger = logging.getLogger(__name__)


def _doc_id_of(record: VectorRecord) -> str:
    """The document_id a record belongs to (records always carry it)."""
    value = record.metadata.get("document_id")
    return value if isinstance(value, str) else ""


class DocumentManager:
    """Manage documents as groups of chunks in the vector store."""

    def __init__(
        self,
        store: BaseVectorStore,
        integrity: FileIntegrityChecker | None = None,
        image_storage: ImageStorage | None = None,
        evidence_dir: Path | None = None,
    ) -> None:
        self.store = store
        # P10.5 T3: when injected, delete() also removes the file's integrity
        # record so a re-ingest of the deleted file is not skipped. None keeps
        # the existing delete behaviour (Chroma only).
        self.integrity = integrity
        # P10.6 ticket 08: when injected, delete() also clears the document's
        # images (best-effort) so no orphan files are left behind. None keeps
        # the existing delete behaviour.
        self.image_storage = image_storage
        # ADR 0021: when injected (the collection's evidence dir), delete() also
        # rewrites the chunks.jsonl snapshot minus the deleted document, so a
        # persisted sparse corpus never scores deleted chunks. Best-effort,
        # like integrity/image — never aborts the Chroma delete.
        self.evidence_dir = evidence_dir

    def _records_for(self, document_id: str) -> list[VectorRecord]:
        """The store records whose document_id matches (empty when missing)."""
        return [
            record
            for record in self.store.get_all()
            if _doc_id_of(record) == document_id
        ]

    def list(self) -> list[dict]:
        """Return one summary dict per document: document_id/source_path/chunk_count."""
        grouped: dict[str, dict] = {}
        for record in self.store.get_all():
            document_id = _doc_id_of(record)
            if not document_id:
                continue
            entry = grouped.setdefault(
                document_id,
                {
                    "document_id": document_id,
                    "source_path": str(record.metadata.get("source_path", "")),
                    "chunk_count": 0,
                },
            )
            entry["chunk_count"] += 1
        return sorted(grouped.values(), key=lambda d: d["document_id"])

    def detail(self, document_id: str) -> list[VectorRecord]:
        """Return the chunks belonging to a document (missing -> empty list)."""
        return self._records_for(document_id)

    def browse(self) -> list[dict]:
        """Every document with its chunks, in ONE get_all() scan (C6).

        Mirrors ``stats_detailed``'s single-scan pattern so a Dashboard page
        never runs N+1 full-store scans (one per document). Each entry carries
        document_id / source_path / chunk_count / chunks (chunk_id + text +
        metadata), sorted by document_id. Empty store -> [].
        """
        grouped: dict[str, dict] = {}
        for record in self.store.get_all():
            document_id = _doc_id_of(record)
            if not document_id:
                continue
            entry = grouped.setdefault(
                document_id,
                {
                    "document_id": document_id,
                    "source_path": str(record.metadata.get("source_path", "")),
                    "chunk_count": 0,
                    "chunks": [],
                },
            )
            entry["chunk_count"] += 1
            entry["chunks"].append(
                {
                    "chunk_id": record.chunk_id,
                    "text": record.text,
                    # shallow copy so a caller mutating the returned dict never
                    # aliases the store record's metadata
                    "metadata": dict(record.metadata),
                }
            )
        return sorted(grouped.values(), key=lambda d: d["document_id"])

    def delete(self, document_id: str) -> int:
        """Delete a document's chunks from the store; return how many were removed.

        Missing document -> 0 (idempotent). Only Chroma is cleared; sparse
        rebuilds per query so it is automatically consistent (ADR 0003). When
        an integrity checker is injected (T3), each deleted chunk's file_hash
        (written into record metadata at ingest time) is also removed from the
        integrity DB so re-ingesting that file is no longer skipped.

        Non-core cleanups (integrity record removal) are best-effort: an
        exception there must never abort the Chroma clear, which is the delete
        contract (FIX-1 — "观察永不改变行为"). Cleanups run *before*
        store.delete so a failed store.delete still leaves integrity cleaned
        (re-ingest recovers, content_hash guards against duplicates).
        """
        records = self._records_for(document_id)
        chunk_ids = [record.chunk_id for record in records]
        if not chunk_ids:
            return 0
        if self.integrity is not None:
            # A document's chunks share one file_hash; remove it once (FIX-2,
            # avoids N redundant DELETE statements / connections for one file).
            file_hashes = {
                record.metadata.get("file_hash")
                for record in records
                if isinstance(record.metadata.get("file_hash"), str)
            }
            for file_hash in file_hashes:
                try:
                    self.integrity.remove_record(file_hash)
                except Exception as exc:  # noqa: BLE001 — cleanup must not abort delete
                    logger.warning(
                        "remove_record(%s) 异常，已忽略（delete 仍清 Chroma）：%s",
                        file_hash,
                        exc,
                    )
        if self.image_storage is not None:
            # P10.6 ticket 08: clear the document's images best-effort so a
            # deleted document leaves no orphan files behind. Same isolation
            # contract as integrity cleanup above.
            try:
                self.image_storage.delete_by_document(document_id)
            except Exception as exc:  # noqa: BLE001 — cleanup must not abort delete
                logger.warning(
                    "delete_by_document(%s) 异常，已忽略（delete 仍清 Chroma）：%s",
                    document_id,
                    exc,
                )
        self.store.delete(chunk_ids)
        if self.evidence_dir is not None:
            # ADR 0021: rewrite the chunks.jsonl snapshot minus the deleted
            # document, so a persisted sparse corpus never scores deleted
            # chunks. Best-effort — a failure must not abort the delete.
            try:
                self._rewrite_evidence_without(document_id)
            except Exception as exc:  # noqa: BLE001 — cleanup must not abort delete
                logger.warning(
                    "evidence 重写(%s) 异常，已忽略（delete 仍完成）：%s",
                    document_id,
                    exc,
                )
        return len(chunk_ids)

    def _rewrite_evidence_without(self, document_id: str) -> None:
        """Rewrite chunks.jsonl in the evidence dir without this document."""
        from rag_mcp_server.ingestion.jsonl_store import read_chunks, write_chunks

        chunks_file = self.evidence_dir / "chunks.jsonl"
        if not chunks_file.is_file():
            return
        existing = read_chunks(chunks_file)
        kept = [c for c in existing if c.document_id != document_id]
        write_chunks(kept, chunks_file)

    def stats(self) -> dict:
        """Return {"documents": n, "chunks": n} over the whole store."""
        by_document: dict[str, int] = defaultdict(int)
        total_chunks = 0
        for record in self.store.get_all():
            document_id = _doc_id_of(record)
            if not document_id:
                continue
            by_document[document_id] += 1
            total_chunks += 1
        return {"documents": len(by_document), "chunks": total_chunks}

    def stats_detailed(self) -> list[dict]:
        """Rich per-document aggregation for MCP/Dashboard (P10.8 ticket 12).

        Same grouping as ``list()`` (group by ``metadata["document_id"]``, skip
        empty ids, sort by document_id) but adds optional keys on top of the
        frozen ``list`` contract:

        - ``image_count`` — only present when an ``image_storage`` is injected
          (= ``len(list_by_document(doc_id))``). Without one we genuinely cannot
          know the count, so the key is omitted rather than claimed as 0
          (CONTEXT.md "文档摘要").
        - ``title`` / ``summary`` — present only when the record metadata has
          them (P10.4 MetadataEnricher output, optional).

        Empty store -> ``[]`` (US13). Existing ``list/stats/detail/delete``
        contracts are untouched. A single ``get_all()`` scan feeds both the
        grouping and the optional title/summary keys (review FIX-2 — no N+1
        re-scan per document).
        """
        grouped: dict[str, dict] = {}
        # one get_all() scan: group by document_id AND harvest the first
        # title/summary each document carries (no second full-table pass).
        for record in self.store.get_all():
            document_id = _doc_id_of(record)
            if not document_id:
                continue
            entry = grouped.setdefault(
                document_id,
                {
                    "document_id": document_id,
                    "source_path": str(record.metadata.get("source_path", "")),
                    "chunk_count": 0,
                },
            )
            entry["chunk_count"] += 1
            meta = record.metadata
            if "title" in meta and "title" not in entry:
                entry["title"] = meta["title"]
            if "summary" in meta and "summary" not in entry:
                entry["summary"] = meta["summary"]
        docs = sorted(grouped.values(), key=lambda d: d["document_id"])
        if not docs:
            return docs
        if self.image_storage is not None:
            for entry in docs:
                try:
                    entry["image_count"] = len(
                        self.image_storage.list_by_document(entry["document_id"])
                    )
                except Exception as exc:  # noqa: BLE001 — stats must not raise
                    logger.warning(
                        "stats_detailed: image count for %s 失败，省略该键：%s",
                        entry["document_id"],
                        exc,
                    )
        return docs
