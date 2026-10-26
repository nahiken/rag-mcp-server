"""Use-case orchestration for the RAG loop.

CLI, scripts, and future MCP tools all call this service instead of
re-implementing business flow.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from rag_mcp_server.core.errors import ConfigError
from rag_mcp_server.core.trace import DisabledTrace
from rag_mcp_server.core.types import DEFAULT_COLLECTION, Chunk, RetrievalResult
from rag_mcp_server.ingestion.jsonl_store import read_chunks
from rag_mcp_server.ingestion.pipeline import IngestPipeline, IngestionSummary
from rag_mcp_server.libs.loader.file_integrity import FileIntegrityChecker
from rag_mcp_server.libs.reranker.base import BaseReranker
from rag_mcp_server.libs.vector_store.base import BaseVectorStore
from rag_mcp_server.retrieval.dense import DenseRetriever
from rag_mcp_server.retrieval.hybrid import (
    CANDIDATE_MULTIPLIER,
    DEFAULT_RRF_K,
    fuse_results,
)
from rag_mcp_server.retrieval.query_processor import build_where
from rag_mcp_server.retrieval.sparse_bm25 import SparseBM25

DEFAULT_INDEX_DIR = Path("data/index")
DEFAULT_CHUNKS_FILE = DEFAULT_INDEX_DIR / "chunks.jsonl"
DEFAULT_REPORT_FILE = DEFAULT_INDEX_DIR / "ingestion_report.json"
DEFAULT_TOP_K = 3

_RETRIEVAL_STRATEGIES = ("dense", "sparse", "hybrid")
_RERANK_PROVIDERS = ("none", "llm", "cross_encoder")


def _default_pipeline(
    index_dir: Path,
    embedding_provider: str | None = None,
    collection: str = DEFAULT_COLLECTION,
) -> IngestPipeline:
    """Assemble a pipeline from config (real Ollama embedding + Chroma store).

    ``embedding_provider`` overrides the configured provider (e.g. "fake" for
    offline runs); None uses the configured value. The vector store uses the
    configured persist_dir + collection so a separate query process reading the
    same settings sees the same store. Offline runs must point their settings
    persist_dir outside the repo (tests use a temp config); the default
    settings.yaml persist_dir (data/chroma) is gitignored.

    ``collection`` selects the Chroma collection (ADR 0016): the store is built
    with that collection name so ingest/query on a collection always hit the
    same data. Transforms (P10.4): assembled from settings
    ``ingestion.transform``; the LLM path is off by default (use_llm: false ->
    pure rules), and when off the pipeline gets no transforms at all so
    existing ingest behaviour is unchanged. Image channel (P10.6): when the
    captioner is explicitly enabled (default off), a single ImageStorage is
    shared between the pipeline (writes extracted PDF images) and the captioner
    (looks them up for Vision captions) so the settings path really extracts
    images -> placeholders -> captions. Integrity (P10.8 closeout, spec
    decision 2): a FileIntegrityChecker is always built and injected, so
    file-level incremental skip is ON by default on the real wiring path (no
    settings switch, Q16; --force is the escape hatch). POTENTIAL-1 (ADR 0015
    known limits): fresh ingests from assembly get the checker; existing stores
    are not migrated.
    """
    from rag_mcp_server.config.settings import load_settings
    from rag_mcp_server.ingestion.storage.image_storage import ImageStorage
    from rag_mcp_server.libs.embedding.factory import create_embedding
    from rag_mcp_server.libs.vector_store.factory import create_vector_store

    settings = load_settings()
    provider = embedding_provider or settings["embedding"]["provider"]
    if provider == "fake":
        embedding = create_embedding(provider)
    else:
        # api_key (ADR 0018) forwarded only when non-empty so the ollama
        # constructor signature (no api_key param) is never violated; empty /
        # absent leaves it to the provider's env fallback (OPENAI_API_KEY).
        kwargs: dict = {
            "base_url": settings["embedding"]["base_url"],
            "model": settings["embedding"]["model"],
        }
        api_key = settings["embedding"].get("api_key")
        if api_key:
            kwargs["api_key"] = api_key
        embedding = create_embedding(provider, **kwargs)
    store = create_vector_store(
        provider=settings["vector_store"]["provider"],
        persist_dir=settings["vector_store"]["persist_dir"],
        collection=collection,
    )
    # T2 review FIX-1: the image channel (pipeline) and the captioner share ONE
    # ImageStorage instance, so the paths the captioner looks up are exactly the
    # paths the pipeline wrote. Previously image_storage was never injected here
    # -> no [IMAGE: id] placeholders -> use_vision was inert on the settings
    # path. Only built when the captioner is explicitly enabled (default off).
    image_storage = None
    captioner_cfg = (
        settings.get("ingestion", {})
        .get("transform", {})
        .get("image_captioner", {})
    )
    if captioner_cfg.get("enabled", False):
        image_storage = ImageStorage()
    transforms = _transforms_from_settings(settings, image_storage=image_storage)
    return IngestPipeline(
        embedding=embedding,
        store=store,
        index_dir=index_dir,
        transforms=transforms,
        image_storage=image_storage,
        integrity=FileIntegrityChecker(),
        collection=collection,
    )


def _transforms_from_settings(settings: dict, image_storage=None) -> list:
    """Assemble transforms from settings ingestion.transform (default off).

    Reads defensively (settings shallow-merge loses deep defaults): a missing
    section or use_llm:false yields no transforms / pure-rule transforms, never
    a crash. LLM path only when a transform's use_llm is true (and then the
    configured LLM is injected). The image captioner (P10.6 T1) is assembled
    only when explicitly enabled (default off, preserving the existing 4-stage
    trace / 4-key record contracts).

    ``image_storage`` is the shared ImageStorage built by the caller
    (_default_pipeline, T2 review FIX-1) so the captioner's Vision lookup reads
    exactly the paths the pipeline wrote. When None (direct callers / tests),
    a captioner that needs Vision builds its own instance — a sibling default
    rooted at data/images.
    """
    from rag_mcp_server.ingestion.storage.image_storage import ImageStorage
    from rag_mcp_server.ingestion.transform import ChunkRefiner, ImageCaptioner, MetadataEnricher

    transform_cfg = settings.get("ingestion", {}).get("transform", {})
    transforms = []
    refiner_cfg = transform_cfg.get("chunk_refiner", {})
    if refiner_cfg.get("use_llm", False):
        transforms.append(ChunkRefiner(use_llm=True, llm=_llm_from_settings(settings)))
    enricher_cfg = transform_cfg.get("metadata_enricher", {})
    if enricher_cfg.get("use_llm", False):
        transforms.append(MetadataEnricher(use_llm=True, llm=_llm_from_settings(settings)))
    captioner_cfg = transform_cfg.get("image_captioner", {})
    if captioner_cfg.get("enabled", False):
        use_vision = captioner_cfg.get("use_vision", False)
        vision_llm = None
        storage = image_storage
        if use_vision:
            vision_llm = _vision_llm_from_settings(settings)
            if storage is None:
                storage = ImageStorage()
        transforms.append(
            ImageCaptioner(vision_llm=vision_llm, image_storage=storage, use_vision=use_vision)
        )
    return transforms


def ingest(
    source: Path,
    base: Path | None = None,
    index_dir: Path = DEFAULT_INDEX_DIR,
    pipeline: IngestPipeline | None = None,
    embedding_provider: str | None = None,
    trace_dir: Path | None = None,
    on_progress: Callable[[str, int, int], None] | None = None,
    integrity: FileIntegrityChecker | None = None,
    force: bool = False,
    collection: str = DEFAULT_COLLECTION,
) -> IngestionSummary:
    """Import supported files under source into the vector store + evidence.

    ``pipeline`` is injected for tests (FakeEmbedding + temp Chroma) and to
    keep the CLI offline; when None, a pipeline is assembled from settings.
    ``embedding_provider`` overrides the configured embedding provider (e.g.
    "fake") when assembling the default pipeline. ``trace_dir`` is where the
    ingestion trace is written (default: settings trace.dir, i.e. logs/);
    observation never changes the return value. ``on_progress(stage, current,
    total)`` is forwarded to the pipeline (Dashboard real-time progress; None
    is a no-op, callback exceptions are swallowed). ``integrity`` is a
    FileIntegrityChecker for file-level incremental skip (P10.5 T1); None keeps
    the existing behaviour (no skipping). ``force`` (P10.5 T2) is forwarded to
    the pipeline: it bypasses the file-level integrity skip so every file
    re-runs the full chain, but NOT the chunk-level content_hash idempotency.

    ``collection`` (ADR 0016) selects the Chroma collection and the evidence
    subdir (``index_dir/{collection}/``); the default pipeline builds the store
    against that collection, so ingesting into different collections keeps the
    same file's chunks separate. None uses the settings default
    (``settings.vector_store.collection``), falling back to ``default``.
    """
    if not collection:
        from rag_mcp_server.config.settings import load_settings as _ls

        configured = _ls().get("vector_store", {}).get("collection")
        collection = configured or DEFAULT_COLLECTION
    if pipeline is None:
        pipeline = _default_pipeline(index_dir, embedding_provider=embedding_provider, collection=collection)
    return pipeline.ingest(
        source, base=base, trace_dir=trace_dir, on_progress=on_progress, integrity=integrity, force=force
    )


def evidence_dir(index_dir: Path, collection: str | None = DEFAULT_COLLECTION) -> Path:
    """The collection-scoped evidence directory (ADR 0016 D19).

    Every ingest/query/tool reads/writes evidence under ``index_dir/{collection}/``
    (chunks.jsonl / ingestion_report.json). Single source for the path formula so
    the guard, the CLI print and the loader never spell it differently. None
    means the default collection.
    """
    return Path(index_dir) / (collection or DEFAULT_COLLECTION)


def collection_names(index_dir: Path) -> list[str]:
    """The ingested collection names under ``index_dir``, sorted.

    Evidence lives under ``index_dir/{collection}/chunks.jsonl`` (ADR 0016
    D19), so the set of collections with a chunk snapshot is exactly the set of
    ingested collections — no need to enumerate Chroma collections directly.
    Shared by the agentic ``list_directory`` and MCP ``list_collections`` tools
    (C7: one source for the enumeration).
    """
    if not index_dir.is_dir():
        return []
    names: list[str] = []
    for entry in sorted(index_dir.iterdir()):
        if entry.is_dir() and (entry / "chunks.jsonl").is_file():
            names.append(entry.name)
    return names


def load_all_chunks(index_dir: Path = DEFAULT_INDEX_DIR, collection: str = DEFAULT_COLLECTION) -> list[Chunk]:
    return read_chunks(evidence_dir(index_dir, collection) / "chunks.jsonl")


def query(
    text: str,
    index_dir: Path = DEFAULT_INDEX_DIR,
    top_k: int | None = None,
    pipeline: IngestPipeline | None = None,
    embedding_provider: str | None = None,
    strategy: str | None = None,
    trace_dir: Path | None = None,
    rerank: bool | None = None,
    reranker: BaseReranker | None = None,
    filters: dict | None = None,
    collection: str = DEFAULT_COLLECTION,
) -> list[RetrievalResult]:
    """Run config-driven retrieval over the vector store.

    ``pipeline`` is injected for tests (FakeEmbedding + temp Chroma), symmetric
    with ingest; when None, a pipeline is assembled from settings.
    ``embedding_provider`` overrides the configured embedding provider (e.g.
    "fake" for offline runs) when assembling that default pipeline — symmetric
    with ``ingest``; None uses the configured value. ``strategy``
    overrides the configured retrieval.strategy (dense/sparse/hybrid); None
    uses settings. ``top_k`` overrides the configured retrieval.top_k; None
    uses settings (default 3). ``trace_dir`` is where the query trace is
    written (default: settings trace.dir, i.e. logs/); observation never
    changes the return value.

    ``filters`` (ADR 0016) is a metadata filter applied at retrieval time
    (Chroma `where` over record metadata, e.g. {"source_path": "a.md", "tags":
    {"$in": ["x"]}}). The filter must happen in retrieval, not post-hoc,
    because RetrievalResult.metadata only carries image_refs (P10.6 T3). None
    = no filter (existing behaviour unchanged). ``collection`` selects the
    Chroma collection and the sparse corpus is rebuilt from that collection's
    store.

    ``rerank`` (bool | None) overrides the configured rerank.enabled; None uses
    settings (default off). ``reranker`` is the injection seam (symmetric with
    ``pipeline``): when given it is used directly; else, when reranking is
    enabled, a backend is assembled from settings (rerank.provider + llm).
    Reranking runs after every strategy branch (dense/sparse/hybrid); on backend
    failure the original order is kept and rerank.fallback=true is written to
    the trace — the result contract is unchanged.

    Filter syntax (ADR 0016 D18): the query text may carry ``source:`` /
    ``tag:`` / ``doc_type:`` / ``collection:`` tokens. ``source/tag/doc_type``
    are stripped from the text and applied as metadata filters (the `where`
    clause) at retrieval time — this is the single place the syntax is wired
    in, so CLI and MCP get it for free (spec T1: "过滤语法解析在 service 层").
    ``collection:`` selects the collection unless the ``collection`` argument
    is given (argument wins, D18). An explicit ``filters`` argument overrides
    the metadata filters parsed from the text (per-key: the explicit value wins
    for that key). The stripped ``clean_text`` is what gets embedded / sparse-
    scored, so filter tokens never pollute the search.

    Empty/whitespace query text returns [] before any strategy runs (no query
    intent); top_k must be a positive integer (fail-fast, consistent across
    strategies). The retrieval strategy is validated regardless of text so a
    broken config is never silently hidden behind an empty-text result.
    """
    from rag_mcp_server.config.settings import load_settings
    from rag_mcp_server.core.trace import TraceContext, trace_is_enabled
    from rag_mcp_server.core.validation import validate_positive_int
    from rag_mcp_server.retrieval.query_processor import (
        parse_collection,
        parse_filters,
    )

    settings = load_settings()
    # C1: the query trace carries the query text in top-level metadata so the
    # Dashboard's query_traces page can label rows (result_count/top_k are
    # merged in at finish via update_metadata).
    trace = (
        TraceContext("query", metadata={"query": text})
        if trace_is_enabled(settings)
        else _NOOP_TRACE
    )

    strategy = settings["retrieval"]["strategy"] if strategy is None else strategy
    if strategy not in _RETRIEVAL_STRATEGIES:
        raise ConfigError(
            f"未知检索策略：{strategy}（可选：{', '.join(_RETRIEVAL_STRATEGIES)}）"
        )

    if top_k is None:
        top_k = settings["retrieval"].get("top_k", DEFAULT_TOP_K)
    validate_positive_int("top_k", top_k)

    # rerank is a quality enhancement (ADR 0009): when enabled, the strategy
    # retrieves a top_k * CANDIDATE_MULTIPLIER candidate pool so passages
    # outside the initial top_k can be promoted (US12), then the rerank stage
    # truncates back to top_k. Disabled (default) -> retrieval_top_k == top_k
    # and the retrieval path is unchanged.
    rerank_enabled = settings["rerank"].get("enabled", False) if rerank is None else rerank
    retrieval_top_k = top_k * CANDIDATE_MULTIPLIER if rerank_enabled else top_k

    # Filter syntax wiring (ADR 0016 D18): strip source:/tag:/doc_type: from
    # the text into metadata filters, and collection: into the collection
    # selector. collection arg wins over the text selector; an explicit filters
    # arg wins per-key over text-parsed filters.
    clean_text, text_filters = parse_filters(text)
    effective_filters = {**(text_filters or {}), **(filters or {})}
    if not collection:
        # collection arg > text selector > settings default > code default
        # (ADR 0016 D16: settings.vector_store.collection is the default)
        configured = settings.get("vector_store", {}).get("collection")
        collection = parse_collection(text) or configured or DEFAULT_COLLECTION

    t_start = time.monotonic()
    if not clean_text or not clean_text.strip():
        trace.record_stage(
            "query_processing",
            {"validated": False, "reason": "empty_text"},
            elapsed_ms=(time.monotonic() - t_start) * 1000.0,
        )
        trace.update_metadata(result_count=0, top_k=top_k)
        _finish_trace(trace, settings, trace_dir)
        return []

    if pipeline is None:
        pipeline = _default_pipeline(index_dir, embedding_provider=embedding_provider, collection=collection)

    store = pipeline.store
    where = build_where(effective_filters)  # None = no filter (existing behaviour)
    if strategy == "sparse":
        corpus = _sparse_corpus(settings, index_dir, store, collection, where)
        sparse = SparseBM25(corpus)
        t_retrieval = time.monotonic()
        results = sparse.search(clean_text, top_k=retrieval_top_k)
        trace.record_stage(
            "query_processing",
            {"strategy": strategy, "top_k": top_k, "collection": collection},
            elapsed_ms=(t_retrieval - t_start) * 1000.0,
        )
        trace.record_stage(
            "sparse_retrieval",
            {"corpus_size": len(corpus)},
            elapsed_ms=(time.monotonic() - t_retrieval) * 1000.0,
        )
    else:
        t_embed = time.monotonic()
        embedding = pipeline.embedding.embed_texts([clean_text])[0]
        trace.record_stage(
            "query_processing",
            {"strategy": strategy, "top_k": top_k, "embedding_dim": len(embedding), "collection": collection},
            elapsed_ms=(t_embed - t_start) * 1000.0,
        )
        if strategy == "dense":
            dense = DenseRetriever(store)
            t_retrieval = time.monotonic()
            results = dense.search(embedding, top_k=retrieval_top_k, where=where)
            trace.record_stage(
                "dense_retrieval",
                {"embedding_dim": len(embedding)},
                elapsed_ms=(time.monotonic() - t_retrieval) * 1000.0,
            )
        else:
            rrf_k = settings["retrieval"].get("rrf_k", DEFAULT_RRF_K)
            # corpus build (sparse) and dense/sparse/fusion all run inside the
            # hybrid search; we time the sub-steps by instrumenting around the
            # retrievers we hand to fuse_results.
            dense = DenseRetriever(store)
            corpus = _sparse_corpus(settings, index_dir, store, collection, where)
            sparse = SparseBM25(corpus)
            candidate_k = top_k * CANDIDATE_MULTIPLIER
            t_dense = time.monotonic()
            dense_results = dense.search(embedding, top_k=candidate_k, where=where)
            trace.record_stage(
                "dense_retrieval",
                {"embedding_dim": len(embedding)},
                elapsed_ms=(time.monotonic() - t_dense) * 1000.0,
            )
            t_sparse = time.monotonic()
            sparse_results = sparse.search(clean_text, top_k=candidate_k)
            trace.record_stage(
                "sparse_retrieval",
                {"corpus_size": len(corpus)},
                elapsed_ms=(time.monotonic() - t_sparse) * 1000.0,
            )
            t_fusion = time.monotonic()
            results = fuse_results(dense_results, sparse_results, top_k=retrieval_top_k, rrf_k=rrf_k)
            trace.record_stage(
                "fusion",
                {"rrf_k": rrf_k},
                elapsed_ms=(time.monotonic() - t_fusion) * 1000.0,
            )

    if rerank_enabled:
        input_k = len(results)  # 精排输入 = 放大的候选池（top_k * CANDIDATE_MULTIPLIER）
        results, fallback, reason = _apply_rerank(clean_text, results, top_k, reranker, settings)
        trace.record_stage(
            "rerank",
            {"fallback": fallback, "reason": reason, "input_k": input_k, "output_k": len(results)},
        )

    # C8: record the actual result count in the trace metadata (the Dashboard's
    # result_count is the real returned count, not the requested top_k).
    trace.update_metadata(result_count=len(results), top_k=top_k)
    _finish_trace(trace, settings, trace_dir)
    return results


def _finish_trace(trace, settings: dict, trace_dir: Path | None) -> None:
    """Persist a finished trace at its resolved location (disabled -> no-op)."""
    from rag_mcp_server.core.trace import resolve_trace_dir
    from rag_mcp_server.observability.logger import write_trace

    if isinstance(trace, DisabledTrace):
        return
    trace.finish()
    write_trace(trace.to_dict(), resolve_trace_dir(settings, trace_dir))


_NOOP_TRACE = DisabledTrace()


def _apply_rerank(
    query_text: str,
    results: list[RetrievalResult],
    top_k: int,
    reranker: BaseReranker | None,
    settings: dict,
) -> tuple[list[RetrievalResult], bool, str | None]:
    """Run the reranking backend over results; return (results, fallback, reason).

    ``reranker`` is the injection seam (tests pass an LLMRanker with FakeLLM).
    When None, a backend is assembled from settings: rerank.provider — "llm" uses
    the configured LLM, "cross_encoder" uses settings rerank.model. Unknown
    providers fail fast (ConfigError, aligned with _RETRIEVAL_STRATEGIES); "none"
    keeps the original order. On backend failure the original order is returned
    with fallback=True (ADR 0009) — the query never breaks.
    """
    from rag_mcp_server.libs.reranker.orchestrator import QueryReranker

    backend = reranker
    if backend is None:
        provider = settings["rerank"].get("provider", "none")
        if provider not in _RERANK_PROVIDERS:
            raise ConfigError(
                f"未知精排 provider：{provider}（可选：{', '.join(_RERANK_PROVIDERS)}）"
            )
        if provider == "llm":
            from rag_mcp_server.libs.reranker.factory import create_reranker

            backend = create_reranker("llm", llm=_llm_from_settings(settings))
        elif provider == "cross_encoder":
            from rag_mcp_server.libs.reranker.cross_encoder import (
                DEFAULT_CROSS_ENCODER_MODEL,
            )
            from rag_mcp_server.libs.reranker.factory import create_reranker

            model_name = settings["rerank"].get("model") or DEFAULT_CROSS_ENCODER_MODEL
            backend = create_reranker("cross_encoder", model_name=model_name)
        else:
            backend = None  # provider "none" -> keep order
    orchestrator = QueryReranker(backend)
    out = orchestrator.rerank(query_text, results, top_k)
    return out, orchestrator.last_fallback, orchestrator.last_reason


def _llm_from_settings(settings: dict):
    """Assemble the configured LLM (settings llm: provider/base_url/model).

    Assembly mirrors embedding (rag_service._default_pipeline passes config
    params to the factory by hand); keep it a private helper so P10.4/P10.7
    consumers can reuse it without duplicating the wiring. ``api_key`` (ADR
    0018) is passed through when set — a non-empty value overrides env, an
    empty/absent value leaves it unset so the provider constructor falls back
    to OPENAI_API_KEY. It is only forwarded when non-empty so the ollama/fake
    constructor signatures (no api_key param) are never violated.
    """
    from rag_mcp_server.libs.llm.factory import create_llm

    llm_cfg = settings.get("llm") or {}
    provider = llm_cfg.get("provider", "ollama")
    if provider == "fake":
        return create_llm(provider)
    kwargs: dict = {
        "base_url": llm_cfg.get("base_url", ""),
        "model": llm_cfg.get("model", ""),
    }
    api_key = llm_cfg.get("api_key")
    if api_key:
        kwargs["api_key"] = api_key
    return create_llm(provider, **kwargs)


def _vision_llm_from_settings(settings: dict):
    """Assemble the configured Vision LLM (settings image_captioner.vision_*).

    Mirrors _llm_from_settings: a real provider is built with the configured
    model/base_url; "fake" returns FakeVisionLLM (offline tests). Unknown
    providers fail fast (ValueError) at assembly time.
    """
    from rag_mcp_server.libs.llm.vision_factory import create_vision_llm

    vision_cfg = settings.get("ingestion", {}).get("transform", {}).get("image_captioner", {})
    provider = vision_cfg.get("vision_provider", "ollama")
    if provider == "fake":
        return create_vision_llm(provider)
    return create_vision_llm(
        provider,
        model=vision_cfg.get("vision_model", ""),
        base_url=vision_cfg.get("vision_base_url", ""),
    )


def _corpus_from_store(store: BaseVectorStore, where: dict | None = None) -> list[Chunk]:
    """Rebuild the sparse corpus as Chunks from the vector store (ADR 0003).

    SparseBM25's contract is list[Chunk]; the store returns VectorRecords, so
    the orchestrator maps at this seam: chunk_id/metadata/text -> Chunk. The
    order is parsed back from the chunk_id suffix (``<document_id>::<order>``).

    ``where`` (ADR 0016) narrows the corpus to records whose metadata matches,
    so a filtered query's sparse route only scores matching chunks — the same
    filter the dense route applies via the store's ``where``.
    """
    chunks: list[Chunk] = []
    for record in store.get_all(where=where):
        document_id = str(record.metadata.get("document_id", ""))
        source_path = str(record.metadata.get("source_path", ""))
        # chunk_id is "{document_id}::{order}"; document_id itself carries a
        # "{collection}::{slug}" prefix (ADR 0016), so the order is the segment
        # AFTER the LAST "::", not the first.
        _, sep, order_str = record.chunk_id.rpartition("::")
        order = int(order_str) if sep and order_str.isdigit() else 0
        chunks.append(
            Chunk(
                document_id=document_id,
                text=record.text,
                source_path=source_path,
                order=order,
                metadata=dict(record.metadata),
            )
        )
    return chunks


def _sparse_corpus(
    settings: dict,
    index_dir: Path,
    store: BaseVectorStore,
    collection: str,
    where: dict | None = None,
) -> list[Chunk]:
    """The sparse corpus for a query — persistent snapshot or rebuild (ADR 0021).

    When ``retrieval.bm25_persist`` is on and the collection's chunks.jsonl
    evidence snapshot exists (written by every ingest, ADR 0016 D19), the
    corpus is loaded from that snapshot (``load_all_chunks``) — avoiding a
    full vector-store scan per query. A ``where`` filter, however, forces the
    rebuild from the store (C2): the snapshot is an unfiltered corpus and
    re-implementing Chroma's ``where`` matching by hand on snapshot chunks
    would diverge from the dense route. So filtered sparse/hybrid queries
    always score a where-honoring corpus, exactly like ``dense.search``.
    Otherwise it falls back to the ADR 0003 rebuild from the store
    (``_corpus_from_store``), which is the default and the soft-degradation
    path when the snapshot is missing/stale.
    """
    persist = settings.get("retrieval", {}).get("bm25_persist", False)
    if persist and not where:
        chunks_file = evidence_dir(index_dir, collection) / "chunks.jsonl"
        if chunks_file.is_file():
            try:
                return load_all_chunks(index_dir, collection)
            except Exception:
                # snapshot unreadable -> fall back to rebuild (soft-degradation)
                pass
    return _corpus_from_store(store, where=where)
