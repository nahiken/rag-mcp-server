"""Dashboard data access — knowledge base stats for the overview page.

Streamlit-free (ADR 0014). The overview page's document/chunk numbers come from
``DocumentManager.stats()`` (unchanged contract — ``stats_detailed`` belongs to
the MCP tools ticket). All functions are pure: they take paths/data and return
plain structures.
"""

from __future__ import annotations

import logging
from typing import Any

from rag_mcp_server.document_manager.manager import DocumentManager
from rag_mcp_server.libs.vector_store.factory import create_vector_store

logger = logging.getLogger(__name__)


def knowledge_stats(settings: dict[str, Any]) -> dict[str, int]:
    """Knowledge base document/chunk stats from the vector store, or zero cards.

    The single source of truth for document/chunk counts is the vector store
    (DocumentManager aggregates over ``store.get_all()``, CONTEXT.md "文档生命周
    期"). We build the store through the existing create_vector_store seam (same
    wiring as the pipeline) and read ``DocumentManager.stats()`` — the existing
    contract, unchanged (``stats_detailed`` belongs to the MCP tools ticket).

    A store that cannot be built (no index yet, provider missing, lock issue)
    is not an error for the Dashboard: the overview shows zero counts instead of
    crashing (US 7 / degraded cards). The failure is still logged so a real bug
    (bad settings, corrupted store) is not silently swallowed as zero cards (S5).
    """
    try:
        vstore = settings.get("vector_store", {})
        store = create_vector_store(
            vstore.get("provider", "chroma"),
            vstore.get("persist_dir"),
            vstore.get("collection", "rag_chunks"),
        )
        return DocumentManager(store).stats()
    except Exception:  # noqa: BLE001 — degraded cards are the Dashboard contract
        logger.warning("knowledge_stats: vector store unavailable, showing zero cards", exc_info=True)
        return {"documents": 0, "chunks": 0}


def summary(stats: dict[str, int]) -> list[dict[str, Any]]:
    """Map ``DocumentManager.stats()`` -> number-card list for the overview page."""
    return [
        {"label": "documents", "value": stats.get("documents", 0)},
        {"label": "chunks", "value": stats.get("chunks", 0)},
    ]


def browse_documents(settings: dict[str, Any]) -> list[dict[str, Any]]:
    """Documents with their chunks, for the data_browser page (ADR 0014 ext).

    Reads the vector store (single source of truth) via ``DocumentManager.browse``
    — one ``get_all()`` scan groups docs AND harvests their chunks, so a large
    store is not scanned N+1 times (C6). Each returned doc carries document_id /
    source_path / chunk_count plus its chunks (chunk_id + text + metadata). A
    broken store degrades to [] (never a crash; the page shows an empty state,
    S5 degraded contract).
    """
    try:
        vstore = settings.get("vector_store", {})
        store = create_vector_store(
            vstore.get("provider", "chroma"),
            vstore.get("persist_dir"),
            vstore.get("collection", "default"),  # ADR 0016 D22 default
        )
        return DocumentManager(store).browse()
    except Exception:  # noqa: BLE001 — degraded empty state is the Dashboard contract
        logger.warning("browse_documents: vector store unavailable, returning []", exc_info=True)
        return []
