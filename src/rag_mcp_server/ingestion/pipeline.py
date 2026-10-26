"""Ingestion pipeline: txt/md/pdf -> chunk -> embed -> upsert vector store.

The single test seam for ingestion behaviour (docs/spec-p9-ingest.md, task 6):
everything above talks to this interface. Embedding and vector store are
injected (FakeEmbedding + temp Chroma in unit tests; Ollama + persistent
Chroma in real wiring), so the pipeline itself never builds providers.

Idempotency: each chunk gets a content_hash (sha256 of the chunk text) as one
of the four metadata keys; the pipeline diffs the new chunk ids against what
the store already has (get_by_ids) and skips chunks whose content_hash already
exists, adding only new ones. JSONL evidence (a stable snapshot of the chunk
ids ingested for this index) is kept alongside the vector store.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from rag_mcp_server.core.errors import ChunkStoreError, DocumentLoadError
from rag_mcp_server.core.types import DEFAULT_COLLECTION, Chunk, Document
from rag_mcp_server.ingestion.chunker import DEFAULT_MAX_CHARS, chunk_document
from rag_mcp_server.ingestion.splitter import (
    chunk_document_fixed,
    chunk_document_semantic,
)
from rag_mcp_server.ingestion.jsonl_store import (
    read_chunks,
    write_chunks,
    write_ingestion_report,
)
from rag_mcp_server.ingestion.loader import load_document, resolve_targets
from rag_mcp_server.ingestion.storage.image_storage import ImageStorage
from rag_mcp_server.ingestion.transform.base import BaseTransform
from rag_mcp_server.libs.embedding.base import BaseEmbedding
from rag_mcp_server.libs.loader.file_integrity import FileIntegrityChecker
from rag_mcp_server.libs.vector_store.base import BaseVectorStore, VectorRecord

logger = logging.getLogger(__name__)

# Transform-injected metadata keys forwarded into the vector-store record
# (whitelist — ADR 0011 pt.3). Extend here when a new transform adds keys
# (P10.6 image_refs). Non-whitelisted chunk.metadata (e.g. file_type) stays
# dropped, preserving the current 4-key record contract.
TRANSFORM_METADATA_KEYS = (
    "title",
    "summary",
    "tags",
    "refined_by",
    "enriched_by",
    "refine_fallback_reason",
    "enrich_fallback_reason",
    "refine_error",
    "enrich_error",
    "image_refs",
    "caption_error",
)


def content_hash_of(text: str) -> str:
    """Content-hash key for idempotency: sha256 of the chunk text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class IngestionSummary:
    imported: list[dict] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)
    total_files: int = 0
    total_skipped: int = 0
    total_chunks: int = 0


class IngestPipeline:
    """Ingests importable documents into the vector store, idempotently."""

    def __init__(
        self,
        embedding: BaseEmbedding,
        store: BaseVectorStore,
        index_dir: Path,
        transforms: list[BaseTransform] | None = None,
        integrity: FileIntegrityChecker | None = None,
        image_storage: ImageStorage | None = None,
        collection: str = DEFAULT_COLLECTION,
        chunker: Callable[[Document], list[Chunk]] | None = None,
    ) -> None:
        self.embedding = embedding
        self.store = store
        self.index_dir = Path(index_dir)
        # ADR 0016 D19: evidence lives under index_dir/{collection}/ so each
        # collection keeps an independent chunk snapshot / report.
        self.collection = collection
        self._evidence_dir = self.index_dir / collection
        # default empty = existing behaviour unchanged (P10.4)
        self.transforms: list[BaseTransform] = transforms or []
        # default None = no file-level incremental skip (P10.5 T1); a caller
        # that injects a checker opts into skipping unchanged files
        self.integrity = integrity
        # default None = no image extraction (P10.6 ticket 08); a caller that
        # injects ImageStorage opts into the pdfplumber channel. Extraction is
        # a sibling channel: text path (markitdown) is untouched, images are
        # saved and referenced via [IMAGE: id] placeholders.
        self.image_storage = image_storage
        # T5 (ADR 0020): chunker injection seam. None -> the splitter is chosen
        # per settings in ingest() (paragraph default, fixed_length/semantic
        # opt-in); an injected chunker wins over settings.
        self.chunker = chunker

    def _progress(
        self,
        on_progress: Callable[[str, int, int], None] | None,
        stage: str,
        current: int,
        total: int,
    ) -> None:
        """Fire the on_progress callback best-effort (never breaks the ingest).

        A broken progress bar (Dashboard) must not destroy a real ingest: the
        callback is observation, so its exceptions are swallowed with a warning
        (aligned with the trace write path being best-effort).
        """
        if on_progress is None:
            return
        try:
            on_progress(stage, current, total)
        except Exception as exc:  # noqa: BLE001 — observation must never raise
            logger.warning("on_progress(%s) 回调异常，已忽略：%s", stage, exc)

    def _chunk_document(self, document: Document, settings: dict) -> tuple[list[Chunk], str]:
        """Chunk a document through the configured splitter (ADR 0020 T5).

        Returns ``(chunks, method)`` where ``method`` is the splitter that
        actually ran — an injected ``chunker`` reports ``"custom"`` so a trace
        never mislabels it as the configured ``paragraph`` etc. An explicitly
        injected ``chunker`` (constructor) wins over settings. Otherwise
        ``settings.ingestion.splitter`` selects the variant: "paragraph"
        (default) keeps the existing ``chunk_document`` behaviour;
        "fixed_length" and "semantic" opt into the new splitters with their
        parameters read from settings (missing keys fall back to code defaults).
        An unknown value degrades to paragraph chunking with a warning (soft,
        never breaks an ingest). Semantic chunking reuses this pipeline's
        embedding and degrades internally to paragraph chunking when the
        embedding is unavailable.
        """
        if self.chunker is not None:
            return self.chunker(document), "custom"
        ingestion_cfg = settings.get("ingestion", {}) or {}
        splitter = ingestion_cfg.get("splitter", "paragraph")
        if splitter == "paragraph":
            return chunk_document(document), splitter
        if splitter == "fixed_length":
            chunk_size = ingestion_cfg.get("chunk_size", DEFAULT_MAX_CHARS)
            overlap = ingestion_cfg.get("overlap", 100)
            return (
                chunk_document_fixed(
                    document, chunk_size=chunk_size, overlap=overlap
                ),
                splitter,
            )
        if splitter == "semantic":
            max_chars = ingestion_cfg.get("chunk_size", DEFAULT_MAX_CHARS)
            threshold = ingestion_cfg.get("semantic_threshold", 0.6)
            return (
                chunk_document_semantic(
                    document,
                    self.embedding,
                    max_chars=max_chars,
                    threshold=threshold,
                ),
                splitter,
            )
        logger.warning("未知 ingestion.splitter=%r，回退段落分块", splitter)
        return chunk_document(document), "paragraph"

    def ingest(
        self,
        source: Path,
        base: Path | None = None,
        trace_dir: Path | None = None,
        on_progress: Callable[[str, int, int], None] | None = None,
        integrity: FileIntegrityChecker | None = None,
        force: bool = False,
    ) -> IngestionSummary:
        """Ingest all importable files under ``source`` into the vector store.

        ``trace_dir`` is where the ingestion trace is written (default: logs/).
        ``on_progress(stage, current, total)`` is called as each stage of each
        file completes (file-level current/total: total = files to process,
        current = 1-based file index). Stage names mirror the trace stages
        (load/split/transform/embed/upsert/skip); transform only fires when
        transforms are configured; empty/idempotent files still fire
        embed/upsert. Observation never changes the return value or the ingest
        behaviour (callback exceptions are swallowed, None is a no-op).

        ``integrity`` is a FileIntegrityChecker for file-level incremental skip
        (P10.5 T1); it overrides the constructor-injected checker for this
        call. None here keeps the constructor value (or, when the constructor
        also has None, no incremental skip at all). ``force`` (P10.5 T2) skips
        the should_skip check so every file re-runs the full chain — a forced
        re-ingest bypasses the performance layer but NOT the chunk-level
        content_hash idempotency (unchanged chunks are still not duplicated).
        """
        from rag_mcp_server.config.settings import load_settings
        from rag_mcp_server.core.trace import DisabledTrace, TraceContext, trace_is_enabled

        settings = load_settings()
        trace = TraceContext("ingestion") if trace_is_enabled(settings) else DisabledTrace()
        # Only a real TraceContext persists data; a DisabledTrace discards it,
        # so skip building the (potentially large) chunk-text detail when off.
        tracing_enabled = isinstance(trace, TraceContext)

        source = Path(source)
        if not source.exists():
            raise DocumentLoadError(f"导入路径不存在：{source}")
        base = Path(base) if base is not None else (source if source.is_dir() else source.parent)
        files, skipped = resolve_targets(source)

        checker = integrity if integrity is not None else self.integrity

        summary = IngestionSummary()
        added_chunks: list[Chunk] = []
        total = len(files)
        for current, file_path in enumerate(files, start=1):
            relative = file_path.relative_to(base).as_posix()
            # file-level incremental skip (P10.5 T1): an unchanged file that was
            # last ingested successfully is skipped without load/embed/upsert.
            # Performance optimisation only — chunk-level content_hash still
            # guarantees correctness. force (P10.5 T2) bypasses this check.
            file_hash: str | None = None
            if checker is not None:
                # The hash must carry the file's *identity* (relative path) as
                # well as its content: two files with identical bytes but
                # different paths are two documents that must BOTH be indexed
                # (F1 regression — a content-only hash wrongly skipped the
                # second on first ingest). A hash read that fails (permission /
                # deleted between resolve and here) degrades to None, meaning
                # "cannot decide" — the file proceeds to load, where
                # load_document turns unreadable files into a skip (F2
                # regression — a bare read_bytes used to abort the whole run).
                # The hash is computed even when forcing: mark_success below
                # needs it to refresh the record after a forced re-ingest
                # (force skips the *check*, not the hash).
                try:
                    file_hash = hashlib.sha256(
                        relative.encode("utf-8") + b"\0" + file_path.read_bytes()
                    ).hexdigest()
                except OSError:
                    file_hash = None
                if not force and file_hash is not None and checker.should_skip(file_hash):
                    trace.record_stage(
                        "integrity_skip",
                        {"file": file_path.name, "reason": "unchanged"},
                    )
                    summary.skipped.append(
                        {"path": relative, "reason": "unchanged (integrity)"}
                    )
                    summary.total_skipped += 1
                    continue

            self._progress(on_progress, "load", current, total)
            try:
                document = load_document(file_path, base=base, collection=self.collection)
            except DocumentLoadError as exc:
                # corrupt / textless PDF or unreadable file -> skipped, not fatal
                trace.record_stage(
                    "skip",
                    {"file": file_path.name, "reason": str(exc)},
                )
                self._progress(on_progress, "skip", current, total)
                if checker is not None and file_hash is not None:
                    # failed != success: the next ingest must retry, not skip
                    checker.mark_failed(file_hash, str(exc))
                summary.skipped.append({"path": relative, "reason": str(exc)})
                summary.total_skipped += 1
                continue

            # load stats land on the trace only after the file actually loaded
            # (skip paths above record their own stage, never a fabricated load).
            trace.record_stage(
                "load",
                {
                    "file": file_path.name,
                    "text_length": len(document.text),
                    "text_preview": document.text[:200],
                },
            )

            # P10.6 ticket 08: pdfplumber image channel (sibling to the
            # markitdown text path). When image_storage is injected and the
            # file is a PDF, extract images, save them, and append
            # "[IMAGE: id]" placeholders so chunk_document distributes them
            # into the text that gets embedded. Extraction failure marks
            # has_unprocessed_images (Q16 strict-degrade) but never blocks the
            # text ingest (Q3 — images are an optional enhancement).
            if self.image_storage is not None and document.file_type == "pdf":
                document = self._extract_and_store_images(document, file_path)

            # the whole single-file chain (chunk -> transforms -> embed ->
            # upsert -> mark_success) runs inside one guard so a non-
            # DocumentLoadError failure (e.g. a transform or store bug) can
            # never leave a stale "success" record behind: the file is marked
            # failed (retried next ingest) and the error re-raises rather than
            # silently producing a half-indexed, wrongly-skipped file.
            try:
                doc_chunks, split_method = self._chunk_document(document, settings)
                # The split trace carries the per-chunk texts so the Dashboard's
                # 摄取追踪 page can show "✂️ 切出了哪些块、每块全文" — the same
                # richness the reference project has. Observation only: the data
                # dict grows but never changes the ingest result. The per-chunk
                # list is built only when tracing is enabled (DisabledTrace would
                # discard it) — no wasted allocation for observability that is off.
                split_data: dict = {
                    "method": split_method,
                    "chunk_count": len(doc_chunks),
                    "avg_chunk_size": (
                        sum(len(c.text) for c in doc_chunks) // len(doc_chunks)
                        if doc_chunks
                        else 0
                    ),
                }
                if tracing_enabled:
                    split_data["chunks"] = [
                        {
                            "chunk_id": c.chunk_id,
                            "order": i,
                            "char_len": len(c.text),
                            "text": c.text,
                        }
                        for i, c in enumerate(doc_chunks)
                    ]
                trace.record_stage("split", split_data)
                self._progress(on_progress, "split", current, total)
                if self.transforms and doc_chunks:
                    doc_chunks = self._apply_transforms(doc_chunks, trace)
                    self._progress(on_progress, "transform", current, total)
                added = self._upsert_chunks(doc_chunks, trace=trace, on_progress=on_progress, current=current, total=total, file_hash=file_hash)
                added_chunks.extend(added)
                if checker is not None and file_hash is not None:
                    # chunk_count records how many chunks this ingest actually
                    # added/changed for the file (len(added), the same number that
                    # lands in summary.total_chunks) — not how many the chunker
                    # produced. On a partial update (file grew) those differ, and
                    # the integrity record should mirror what is really in the
                    # store, so DocumentManager.delete -> remove_record -> re-ingest
                    # restores the file faithfully.
                    checker.mark_success(file_hash, relative, chunk_count=len(added))
            except Exception:
                if checker is not None and file_hash is not None:
                    checker.mark_failed(file_hash, "unexpected ingest error")
                raise
            summary.imported.append(
                {
                    "path": document.source_path,
                    "document_id": document.id,
                    "chunk_count": len(added),
                }
            )
            summary.total_files += 1
            summary.total_chunks += len(added)

        for skipped_file in skipped:
            relative = skipped_file.path.relative_to(base).as_posix()
            trace.record_stage(
                "skip",
                {"file": skipped_file.path.name, "reason": skipped_file.reason},
            )
            summary.skipped.append({"path": relative, "reason": skipped_file.reason})
            summary.total_skipped += 1

        # JSONL evidence + report (kept alongside the vector store)
        self._write_evidence(summary, added_chunks)

        self._finish_trace(trace, settings, trace_dir)
        return summary

    def _finish_trace(self, trace, settings: dict, trace_dir: Path | None) -> None:
        """Finalize + persist the ingestion trace (best-effort, never raises)."""
        from rag_mcp_server.core.trace import DisabledTrace, resolve_trace_dir
        from rag_mcp_server.observability.logger import write_trace

        if isinstance(trace, DisabledTrace):
            return
        trace.finish()
        write_trace(trace.to_dict(), resolve_trace_dir(settings, trace_dir))

    def _extract_and_store_images(self, document: Document, file_path: Path) -> Document:
        """Run the pdfplumber image channel; return the enriched Document.

        Extracts images from ``file_path``, saves each via image_storage, and
        appends ``[IMAGE: {image_id}]`` placeholders to the document text so
        chunk_document distributes them (Q4 data flow). On a channel failure,
        sets ``metadata["has_unprocessed_images"]=True`` (Q16 strict-degrade,
        only observed via trace — never blocks the text ingest).
        """
        from rag_mcp_server.ingestion.pdf_images import extract_pdf_images

        images, degraded = extract_pdf_images(file_path, document.id)
        metadata = dict(document.metadata or {})
        if degraded:
            metadata["has_unprocessed_images"] = True
        if not images:
            return Document(
                id=document.id,
                text=document.text,
                source_path=document.source_path,
                file_type=document.file_type,
                metadata=metadata,
            )
        stored_refs: list[str] = []
        images_meta: list[dict] = []
        for image_id, data, mime, page in images:
            self.image_storage.save_image(
                image_id,
                data,
                mime_type=mime,
                document_id=document.id,
                page_num=page,
            )
            stored_refs.append(image_id)
            images_meta.append(
                {
                    "id": image_id,
                    "page": page,
                    "mime_type": mime,
                }
            )
        metadata["images"] = images_meta
        # placeholder appended at end of text; the
        # captioner (ticket 09) replaces it with a semantic caption
        placeholder = "\n".join(f"[IMAGE: {image_id}]" for image_id in stored_refs)
        text = document.text + ("\n" if document.text else "") + placeholder
        return Document(
            id=document.id,
            text=text,
            source_path=document.source_path,
            file_type=document.file_type,
            metadata=metadata,
        )

    def _apply_transforms(self, chunks: list[Chunk], trace) -> list[Chunk]:
        """Run the configured transforms serially (each feeds the next).

        Records one 'transform' trace stage (only when transforms are present —
        the default-empty path records nothing, keeping the four-stage shape).
        """
        applied = [type(t).__name__ for t in self.transforms]
        out = chunks
        for transform in self.transforms:
            out = transform.transform(out, trace)
        refined_by: dict[str, int] = {}
        enriched_by: dict[str, int] = {}
        for chunk in out:
            r = chunk.metadata.get("refined_by")
            if r:
                refined_by[r] = refined_by.get(r, 0) + 1
            e = chunk.metadata.get("enriched_by")
            if e:
                enriched_by[e] = enriched_by.get(e, 0) + 1
        trace.record_stage(
            "transform",
            {
                "applied": applied,
                "total_chunks": len(out),
                "refined_by": refined_by,
                "enriched_by": enriched_by,
            },
        )
        return out

    def _upsert_chunks(
        self,
        chunks: list[Chunk],
        trace,
        on_progress: Callable[[str, int, int], None] | None = None,
        current: int = 0,
        total: int = 0,
        file_hash: str | None = None,
    ) -> list[Chunk]:
        """Embed and store the changed chunks; return the ones added/changed.

        Diff by chunk_id: a chunk whose content_hash already matches the stored
        record is skipped (idempotent). A chunk_id that already exists with
        *different* content (partial edit, order shift) is re-stored with the
        new text (delete + add) so the vector store never silently keeps the
        stale first write. A new chunk_id is added even when its text matches
        another chunk's (dedup key is the id, not the hash).

        ``trace`` records embed/upsert even when every chunk is idempotently
        skipped or there is nothing to chunk at all, so a re-ingest or a blank
        document still shows the full four-stage shape per file (spec US7).
        ``on_progress`` mirrors that: embed/upsert fire for empty/idempotent
        files too, so the progress bar moves monotonically and stays aligned
        with the trace stage shape.
        """
        if not chunks:
            trace.record_stage("embed", {"chunks": 0, "empty": True})
            trace.record_stage("upsert", {"added": 0, "empty": True})
            self._progress(on_progress, "embed", current, total)
            self._progress(on_progress, "upsert", current, total)
            return []

        existing = self.store.get_by_ids([c.chunk_id for c in chunks])
        existing_by_id = {r.chunk_id: r for r in existing}

        to_add: list[Chunk] = []
        to_replace: list[Chunk] = []
        for chunk in chunks:
            chunk_hash = content_hash_of(chunk.text)
            old = existing_by_id.get(chunk.chunk_id)
            if old is not None and old.metadata.get("content_hash") == chunk_hash:
                continue  # identical content already ingested
            chunk.metadata["content_hash"] = chunk_hash
            if old is not None:
                to_replace.append(chunk)  # same id, changed content
            else:
                to_add.append(chunk)

        for chunk in to_replace:
            self.store.delete([chunk.chunk_id])
        changed = to_add + to_replace

        if changed:
            records: list[VectorRecord] = []
            embeddings: list[list[float]] = []
            for chunk in changed:
                # 4 mandatory keys + whitelisted transform keys (ADR 0011 pt.3);
                # other chunk.metadata (e.g. file_type) stays dropped.
                record_meta = {
                    "document_id": chunk.document_id,
                    "source_path": chunk.source_path,
                    "chunk_id": chunk.chunk_id,
                    "content_hash": chunk.metadata["content_hash"],
                }
                if file_hash is not None:
                    # optional key, written only when integrity is active (T3):
                    # DocumentManager.delete reads it to remove_record — no need
                    # to re-read the file to reverse the hash. integrity=None
                    # keeps the exact 4-key contract (test_..._exact_four_keys).
                    record_meta["file_hash"] = file_hash
                for key in TRANSFORM_METADATA_KEYS:
                    if key in chunk.metadata:
                        record_meta[key] = chunk.metadata[key]
                records.append(
                    VectorRecord(
                        chunk_id=chunk.chunk_id,
                        metadata=record_meta,
                        text=chunk.text,
                    )
                )
                embeddings.append(self.embedding.embed_texts([chunk.text])[0])
            trace.record_stage("embed", {"chunks": len(changed)})
            self._progress(on_progress, "embed", current, total)
            self.store.add(records, embeddings)
            trace.record_stage("upsert", {"added": len(changed)})
            self._progress(on_progress, "upsert", current, total)
        else:
            trace.record_stage("embed", {"chunks": 0, "idempotent": True})
            trace.record_stage("upsert", {"added": 0, "idempotent": True})
            self._progress(on_progress, "embed", current, total)
            self._progress(on_progress, "upsert", current, total)

        return changed

    def _write_evidence(self, summary: IngestionSummary, added_chunks: list[Chunk]) -> None:
        """Persist a stable snapshot of ingested chunks + the ingestion report.

        chunks.jsonl is rewritten each ingest as a deduplicated snapshot of the
        chunk ids ingested for this index, so it stays consistent with the
        vector store (which keeps first write per id) and serves as evidence.
        Lives under ``index_dir/{collection}/`` (ADR 0016 D19).
        """
        chunks_file = self._evidence_dir / "chunks.jsonl"
        report_file = self._evidence_dir / "ingestion_report.json"

        existing: list[Chunk] = []
        if chunks_file.is_file():
            try:
                existing = read_chunks(chunks_file)
            except ChunkStoreError as exc:
                # evidence read failure must not abort the ingest, but should
                # not be silent either — the store is the source of truth
                logger.warning("chunks.jsonl 读取失败，重建快照：%s", exc)
                existing = []
        existing_by_id = {c.chunk_id: c for c in existing}
        for chunk in added_chunks:
            # a changed chunk replaces its old evidence row so the snapshot
            # stays aligned with the vector store (which stores new text)
            existing_by_id[chunk.chunk_id] = chunk
        write_chunks(list(existing_by_id.values()), chunks_file)

        write_ingestion_report(
            {
                "imported": summary.imported,
                "skipped": summary.skipped,
                "total_files": summary.total_files,
                "total_skipped": summary.total_skipped,
                "total_chunks": summary.total_chunks,
            },
            report_file,
        )
