"""Shared input validation helpers (fail-fast guards)."""

from __future__ import annotations

import re

# A credential-looking substring like "user:pass@" in a proxy-controlled body
# must never reach an error message (CONTEXT: sensitive config never in error
# text). Fold the body to one line, drop such substrings, then truncate.
_DEFAULT_MAX_BODY_CHARS = 200


def validate_positive_int(name: str, value: object) -> int:
    """Validate a config value is a positive integer; return it (fail fast)."""
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} 必须是大于 0 的整数，得到 {value!r}")
    return value


def summary_body(text: str, max_chars: int = _DEFAULT_MAX_BODY_CHARS) -> str:
    """Flatten, credential-scrub and truncate a response body for error text.

    A non-2xx response body is proxy-controlled and could echo credentials
    (e.g. an API key in a userinfo-shaped substring). Fold it to one line,
    drop any ``user:pass@`` substring, then truncate — shared by LLM/embedding
    providers so every error path scrubs identically (review FIX).
    """
    flat = " ".join(text.split())
    flat = re.sub(r"[^\s:@/]+:[^\s:@/]+@", "<redacted>", flat)
    return flat[:max_chars]
