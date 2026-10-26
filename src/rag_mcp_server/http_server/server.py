"""Starlette HTTP (REST) app for the RAG knowledge base.

Reuses the existing use-case seam (``rag_service.ingest/query``) and
``DocumentManager`` exactly as the MCP server does — no business logic is
re-implemented here, only request/response mapping. All blocking calls
(embedding via Ollama, Chroma writes) run in the threadpool; writes
(upload/delete) are serialized with an asyncio.Lock so Chroma never sees two
concurrent writers.

Endpoints
---------
GET    /health                     -> ok probe
POST   /upload                     -> multipart file (+ optional collection/force) -> ingest
GET    /documents                  -> list documents (+ size/mtime from data/uploads)
GET    /documents/{document_id}    -> detail/preview: chunk texts (?full=1 returns untruncated)
DELETE /documents/{document_id}    -> delete a document's chunks (idempotent)
POST   /query                      -> {text, top_k?, collection?} -> retrieval results
GET    /collections                -> ingested collection names

Offline fallback: set env ``RAG_HTTP_EMBEDDING_PROVIDER=fake`` to bypass Ollama
for both upload and query — the API is exercisable end-to-end, but retrieval
quality is meaningless under ``fake`` (throwaway embedding), so it is meant for
smoke-testing / tests, not real retrieval.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from pathlib import Path

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from rag_mcp_server.application import rag_service
from rag_mcp_server.core.errors import (
    ConfigError,
    DocumentLoadError,
    EmbeddingError,
    RagMcpError,
    VectorStoreError,
)
from rag_mcp_server.core.types import DEFAULT_COLLECTION
from rag_mcp_server.document_manager.manager import DocumentManager

logger = logging.getLogger(__name__)

# Relative to the CWD where the server is launched (RAG-MCP-SERVER root in the
# provided start script). Uploaded source files are persisted here so the list
# endpoint can report size/mtime and re-ingest/delete stay consistent.
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB, matches the frontend guard
ALLOWED_EXTS = {".txt", ".md", ".pdf"}

# A single asyncio.Lock serializes Chroma write operations (upload/delete).
_write_lock = asyncio.Lock()


def _embedding_provider() -> str | None:
    """Embedding provider override for the ingest & query pipelines (None = use settings).

    Set env ``RAG_HTTP_EMBEDDING_PROVIDER=fake`` to bypass Ollama: upload and
    query then run fully offline against a throwaway embedding. Retrieval
    quality is meaningless under ``fake`` (dimensions/space differ from the
    stored vectors), so it is meant for exercising the API / tests, not for
    real retrieval — the docs in the module docstring reflect that.
    """
    value = os.getenv("RAG_HTTP_EMBEDDING_PROVIDER", "").strip()
    return value or None


def _safe_filename(name: str) -> str:
    """Keep the original basename (incl. CJK); strip path separators / control chars."""
    name = (name or "upload.txt").replace("\\", "/").rsplit("/", 1)[-1].strip()
    cleaned = "".join(ch for ch in name if ch.isalnum() or ch in " ._-()（）【】[]，,、")
    cleaned = cleaned.strip(" .") or "upload.txt"
    return cleaned


def _make_manager(collection: str | None = None) -> DocumentManager:
    """A DocumentManager wired to settings store + integrity, like the MCP server."""
    from rag_mcp_server.config.settings import load_settings
    from rag_mcp_server.libs.loader.file_integrity import FileIntegrityChecker
    from rag_mcp_server.libs.vector_store.factory import create_vector_store

    settings = load_settings()
    vstore = settings["vector_store"]
    effective = collection or DEFAULT_COLLECTION
    store = create_vector_store(
        provider=vstore["provider"],
        persist_dir=vstore.get("persist_dir"),
        collection=effective,
    )
    return DocumentManager(
        store=store,
        integrity=FileIntegrityChecker(),
        evidence_dir=rag_service.evidence_dir(rag_service.DEFAULT_INDEX_DIR, effective),
    )


def _chunk_payload(records, truncate: int | None = None) -> list[dict]:
    """Render a document's chunks as a JSON payload: id/order/char_len/text.

    Shared by ``GET /documents/{id}`` and the ``POST /upload`` response so both
    produce identical chunk dicts. ``truncate`` caps ``text``: ``None`` (the
    default) returns full chunk texts (upload confirmation / ``?full=1``
    preview); an explicit ``truncate=500`` keeps the lean default of the
    ``GET /documents/{id}`` response for callers that only need a preview.
    """
    chunks = []
    for i, rec in enumerate(records):
        text = rec.text
        if truncate is not None:
            text = text[:truncate]
        chunks.append(
            {
                "chunk_id": rec.chunk_id,
                "order": i,
                "char_len": len(rec.text),
                "text": text,
            }
        )
    return chunks


def _error_response(exc: Exception) -> JSONResponse:
    if isinstance(exc, DocumentLoadError):
        return JSONResponse({"ok": False, "error": f"文件无法解析/加载：{exc}"}, status_code=400)
    if isinstance(exc, EmbeddingError) or "ollama" in str(exc).lower():
        return JSONResponse(
            {
                "ok": False,
                "error": "embedding 服务不可用：请启动 Ollama 或设置 RAG_HTTP_EMBEDDING_PROVIDER=fake",
            },
            status_code=503,
        )
    if isinstance(exc, (ConfigError, VectorStoreError, RagMcpError)):
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    logger.exception("RAG HTTP unhandled error")
    return JSONResponse({"ok": False, "error": "服务器内部错误"}, status_code=500)


# ---- endpoints --------------------------------------------------------------

async def health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "service": "rag-http"})


async def upload(request: Request) -> JSONResponse:
    form = await request.form()
    file = form.get("file")
    if file is None:
        return JSONResponse({"ok": False, "error": "缺少 multipart 字段 file"}, status_code=400)

    data = await file.read()
    if not data:
        return JSONResponse({"ok": False, "error": "空文件"}, status_code=400)
    if len(data) > MAX_UPLOAD_BYTES:
        return JSONResponse({"ok": False, "error": "文件超过 5MB 上限"}, status_code=413)

    filename = _safe_filename(file.filename or "upload.txt")
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        return JSONResponse(
            {"ok": False, "error": f"仅支持 {', '.join(sorted(ALLOWED_EXTS))}"},
            status_code=400,
        )

    collection = (form.get("collection") or DEFAULT_COLLECTION).strip() or DEFAULT_COLLECTION
    force = str(form.get("force", "")).lower() in {"1", "true", "yes"}

    # Persist the source file under data/uploads so list/preview can stat it and
    # a later re-ingest/delete resolves to the same path.
    dest = UPLOAD_DIR / filename
    if dest.exists():
        dest = UPLOAD_DIR / f"{int(time.time() * 1000)}_{filename}"
    dest.write_bytes(data)

    provider = _embedding_provider()

    # Collect which pipeline stages ran (load/split/embed/upsert) so the upload
    # response can drive a staged checklist in the admin UI. Observation only:
    # the progress callback never alters the ingest. Single-file uploads fire
    # each stage at most once, so dedupe later with dict.fromkeys (order kept).
    stage_names: list[str] = []

    def _on_progress(stage: str, current: int, total: int) -> None:
        stage_names.append(stage)

    try:
        async with _write_lock:
            summary = await run_in_threadpool(
                rag_service.ingest,
                dest,
                index_dir=rag_service.DEFAULT_INDEX_DIR,
                embedding_provider=provider,
                force=force,
                collection=collection,
                on_progress=_on_progress,
            )
    except Exception as exc:  # noqa: BLE001 — map to friendly HTTP
        return _error_response(exc)

    imported = list(summary.imported)
    if not imported:
        # The file was skipped (corrupt / textless / integrity-unchanged) — the
        # UI must treat it as a failure with a reason, never as a successful
        # ingest of 0 chunks (P2: don't blame "an old server" for a real skip).
        reason = summary.skipped[0].get("reason", "文件无法解析或内容未变化，已跳过")
        return JSONResponse(
            {"ok": False, "error": f"文件未摄取：{reason}", "skipped": len(summary.skipped)},
            status_code=200,
        )

    doc_id = str(imported[0].get("document_id", ""))
    total_chunks = int(imported[0].get("chunk_count", 0))
    # ordered, deduplicated stage list for the UI checklist (dict.fromkeys keeps
    # first-seen order); split carries the resulting chunk count so the ✂️ step
    # reads "分块(N)" — the shape the admin page renders.
    ordered_stages = [{"stage": s} for s in dict.fromkeys(stage_names)]
    for entry in ordered_stages:
        if entry["stage"] == "split":
            entry["chunk_count"] = total_chunks
    payload: dict = {
        "ok": True,
        "document_id": doc_id,
        "filename": filename,
        "collection": collection,
        "imported": len(imported),
        "skipped": len(summary.skipped),
        "total_chunks": total_chunks,
        "stages": ordered_stages,
    }
    # full chunk texts so the upload confirmation can immediately show the
    # ✂️ split result ("切成 N 块、每块全文") without a second round-trip
    try:
        records = await run_in_threadpool(_make_manager(collection).detail, doc_id)
        payload["chunks"] = _chunk_payload(records, truncate=None)
    except Exception:  # noqa: BLE001 — detail is best-effort on top of a done ingest
        payload["chunks"] = []
    return JSONResponse(payload)


async def list_documents(request: Request) -> JSONResponse:
    collection = (request.query_params.get("collection") or DEFAULT_COLLECTION).strip()
    try:
        docs = await run_in_threadpool(_make_manager(collection).list)
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc)

    out = []
    for doc in docs:
        source = doc.get("source_path", "")
        name = Path(source).name or source
        size = None
        mtime = None
        fpath = UPLOAD_DIR / name
        if fpath.is_file():
            try:
                stat = fpath.stat()
                size = stat.st_size
                mtime = int(stat.st_mtime * 1000)
            except OSError:
                pass
        out.append(
            {
                "document_id": doc.get("document_id"),
                "source_path": name,
                "chunk_count": doc.get("chunk_count", 0),
                "size_bytes": size,
                "modified_at": mtime,
            }
        )
    return JSONResponse({"collection": collection, "documents": out})


async def document_detail(request: Request) -> JSONResponse:
    document_id = request.path_params.get("document_id", "")
    collection = (request.query_params.get("collection") or DEFAULT_COLLECTION).strip()
    # ?full=1 returns untruncated chunk texts (admin 分块预览); the default
    # stays 500-char-truncated for backward compatibility with lean callers.
    full = request.query_params.get("full", "").lower() in {"1", "true", "yes"}
    try:
        records = await run_in_threadpool(_make_manager(collection).detail, document_id)
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc)

    if not records:
        return JSONResponse({"ok": False, "error": f"未找到文档 {document_id}"}, status_code=404)

    source = records[0].metadata.get("source_path", "")
    chunks = _chunk_payload(records, truncate=None if full else 500)
    return JSONResponse(
        {"document_id": document_id, "source_path": source, "chunk_count": len(chunks), "chunks": chunks}
    )


async def delete_document(request: Request) -> JSONResponse:
    document_id = request.path_params.get("document_id", "")
    collection = (request.query_params.get("collection") or DEFAULT_COLLECTION).strip()
    try:
        async with _write_lock:
            deleted = await run_in_threadpool(_make_manager(collection).delete, document_id)
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc)
    return JSONResponse({"ok": True, "deleted": deleted})


async def query_documents(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        body = {}
    text = str(body.get("text", "") or "").strip()
    if not text:
        return JSONResponse({"query": "", "collection": DEFAULT_COLLECTION, "top_k": 0, "results": []})

    top_k = body.get("top_k")
    collection = str(body.get("collection") or DEFAULT_COLLECTION).strip()

    try:
        results = await run_in_threadpool(
            rag_service.query,
            text,
            index_dir=rag_service.DEFAULT_INDEX_DIR,
            embedding_provider=_embedding_provider(),
            top_k=top_k,
            collection=collection,
        )
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc)

    return JSONResponse(
        {
            "query": text,
            "collection": collection,
            "top_k": len(results),
            "results": [
                {
                    "chunk_id": r.chunk_id,
                    "source_path": r.source_path,
                    "score": round(float(r.score), 4),
                    "text": r.text[:500],
                }
                for r in results
            ],
        }
    )


async def collections(request: Request) -> JSONResponse:
    try:
        names = await run_in_threadpool(rag_service.collection_names, rag_service.DEFAULT_INDEX_DIR)
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc)
    return JSONResponse({"collections": names, "selected": DEFAULT_COLLECTION})


def build_app() -> Starlette:
    app = Starlette(
        routes=[
            Route("/health", health),
            Route("/upload", upload, methods=["POST"]),
            Route("/documents", list_documents),
            Route("/documents/{document_id}", document_detail),
            Route("/documents/{document_id}", delete_document, methods=["DELETE"]),
            Route("/query", query_documents, methods=["POST"]),
            Route("/collections", collections),
        ]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app


app = build_app()
