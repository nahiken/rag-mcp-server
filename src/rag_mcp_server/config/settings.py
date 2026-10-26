"""Settings loading: `config/settings.yaml` over code defaults.

Merge rule (per DEV_SPEC phase9): shallow merge + explicit field read.
A present top-level section is shallow-merged over the default section, so
unspecified nested keys keep their code defaults. Unknown top-level sections
are preserved as-is for forward compatibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Code defaults: the single source of truth when no config file exists.
DEFAULTS: dict[str, Any] = {
    "llm": {
        "provider": "ollama",
        "base_url": "http://127.0.0.1:11451",
        "model": "qwen2.5:7b",
        # ADR 0018 D26: empty string -> env fallback (OPENAI_API_KEY); set a
        # real key to override, leave empty to read from the environment.
        "api_key": "",
    },
    "embedding": {
        "provider": "ollama",
        "base_url": "http://127.0.0.1:11451",
        "model": "nomic-embed-text",
        # ADR 0018 D26: empty string -> env fallback (OPENAI_API_KEY).
        "api_key": "",
    },
    "vector_store": {
        "provider": "chroma",
        "persist_dir": "data/chroma",
        # ADR 0016 D22: the default collection is "default" (multi-collection);
        # this is the settings-side default, aligned with DEFAULT_COLLECTION.
        "collection": "default",
    },
    "retrieval": {
        "strategy": "hybrid",
        "top_k": 3,
        "rrf_k": 60,
        # ADR 0021: 持久 BM25 索引（默认关）。开启后 sparse 查询从 chunks.jsonl
        # 语料快照构造（免扫向量库），否则每查询重建（ADR 0003）。
        "bm25_persist": False,
    },
    "rerank": {
        "enabled": False,
        "provider": "none",
        # cross_encoder 的本地/官方模型名（默认官方；本地路径也支持）。
        "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    },
    "ingestion": {
        # T5 (ADR 0020): 分块策略 paragraph（默认，段落分块）/
        # fixed_length（固定长度+重叠）/ semantic（语义分块）。
        "splitter": "paragraph",
        "chunk_size": 800,  # fixed_length 窗口 / semantic 超长续切的 max_chars
        "overlap": 100,  # fixed_length 窗口重叠字符数
        "semantic_threshold": 0.6,  # semantic 相邻段落相似度阈值
        "transform": {
            "chunk_refiner": {"use_llm": False},
            "metadata_enricher": {"use_llm": False},
            "image_captioner": {
                "enabled": False,
                "use_vision": False,
                "vision_provider": "ollama",
                "vision_model": "qwen2.5vl:7b",
                "vision_base_url": "http://127.0.0.1:11451",
            },
        },
    },
    "trace": {
        "enabled": True,
        "dir": "logs",
    },
    "index_dir": "data/index",
}

DEFAULT_CONFIG_DIR = Path("config")
DEFAULT_SETTINGS_FILE = DEFAULT_CONFIG_DIR / "settings.yaml"


def _merge_section(default_value: Any, override: Any) -> Any:
    """Merge one top-level value.

    Sections (dict defaults) are shallow-merged so unspecified nested keys keep
    defaults; flat values (e.g. index_dir) use the file value when present.
    A missing override (None) returns a copy of the dict default so callers can
    never alias/mutate the module-level DEFAULTS.
    """
    if isinstance(default_value, dict):
        if not isinstance(override, dict):
            return dict(default_value)
        merged = dict(default_value)
        for key, value in override.items():
            merged[key] = value
        return merged
    return default_value if override is None else override


def load_settings(
    config_dir: Path | None = None,
    path: Path | None = None,
) -> dict[str, Any]:
    """Return merged settings.

    - `path` given: load exactly that file (missing file treated as empty).
    - otherwise look for `config_dir/settings.yaml` (defaults to ./config).
    """
    target = path if path is not None else (config_dir or DEFAULT_CONFIG_DIR) / "settings.yaml"
    file_data: dict[str, Any] = {}
    if Path(target).is_file():
        with open(target, encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
            if isinstance(loaded, dict):
                file_data = loaded

    merged: dict[str, Any] = {}
    for key, default_value in DEFAULTS.items():
        if key in file_data:
            merged[key] = _merge_section(default_value, file_data[key])
        else:
            # missing section: copy the default so callers can never alias /
            # mutate the module-level DEFAULTS (regression: E1/E2)
            merged[key] = _merge_section(default_value, None)
    # unknown top-level sections are preserved verbatim
    for key in file_data:
        if key not in merged:
            merged[key] = file_data[key]
    return merged
