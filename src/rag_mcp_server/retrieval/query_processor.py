"""Query preprocessing: strip `key:value` filter syntax, keep the search text.

Parsed filters map to retrieval-time metadata filters (Chroma `where`). The
two roles are distinct (ADR 0016 D18):

- ``collection`` selects the vector-store collection (the ``collection`` arg
  of rag_service.query / MCP tools). ``collection:`` in the query text maps to
  the same role, and when both are given the explicit argument wins.
- ``source:`` / ``doc_type:`` / ``tag:`` filter *within* the selected
  collection (record-metadata fields) — they become the `where` clause.

Unknown `key:value` tokens are left in the search text as normal keywords
(they are not a recognized filter, so they must not silently disappear).
"""
from __future__ import annotations

import re
from typing import Any

FILTER_PATTERN = re.compile(r"(\w+):([^\s]+)")

# alias -> record-metadata field; value is (metadata_key, is_list_field)
_METADATA_FILTER_FIELDS = {
    "doc_type": ("file_type", False),
    "type": ("file_type", False),
    "t": ("file_type", False),
    "source": ("source_path", False),
    "src": ("source_path", False),
    "s": ("source_path", False),
    "tag": ("tags", True),
    "tags": ("tags", True),
}

# collection aliases map to the collection selector, not a record-metadata key
_COLLECTION_ALIASES = ("collection", "col", "c")


def parse_filters(query: str) -> tuple[str, dict[str, Any]]:
    """Extract filter syntax from a query; return (clean_query, filters).

    ``filters`` holds ``source_path`` / ``file_type`` / ``tags`` keys ready to
    build a Chroma ``where`` (see build_where). The ``collection`` filter is
    NOT put here — it is surfaced separately via ``parse_collection`` so the
    caller can apply it as the collection selector (argument wins).
    """
    filters: dict[str, Any] = {}
    tags: list[str] = []
    remaining: list[str] = []

    for token in query.split():
        m = FILTER_PATTERN.fullmatch(token)
        if not m:
            remaining.append(token)
            continue
        key, value = m.group(1).lower(), m.group(2)
        if key in _COLLECTION_ALIASES:
            # collection: is a collection selector, not search content — it is
            # stripped from the clean query (parse_collection surfaces it) and
            # must not be searched as a keyword.
            continue
        field = _METADATA_FILTER_FIELDS.get(key)
        if field is None:
            remaining.append(token)  # unknown filter stays a keyword
            continue
        meta_key, is_list = field
        if is_list:
            tags.extend(value.split(","))
        else:
            filters[meta_key] = value

    if tags:
        filters["tags"] = {"$in": tags}
    return " ".join(remaining), filters


def parse_collection(query: str) -> str | None:
    """Extract the ``collection:xxx`` selector from a query, if any.

    Returns the first `collection:` / `col:` / `c:` token's value, or None.
    The caller combines it with its own collection argument: when both are
    given, the argument wins (ADR 0016 D18).
    """
    for token in query.split():
        m = FILTER_PATTERN.fullmatch(token)
        if not m:
            continue
        key, value = m.group(1).lower(), m.group(2)
        if key in _COLLECTION_ALIASES:
            return value
    return None


def build_where(filters: dict[str, Any]) -> dict[str, Any] | None:
    """Convert parsed metadata filters to a Chroma `where` clause.

    None for no filter (so callers can pass where=None unchanged). Multiple
    keys are ANDed by Chroma; list fields use $in (any-match).
    """
    if not filters:
        return None
    return {k: v for k, v in filters.items()}
