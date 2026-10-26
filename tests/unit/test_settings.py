"""Tests for config/settings.py — config loading with code-default fallback.

Red/green contract:
- no config file  -> returns code defaults
- yaml present    -> listed field overrides default
- missing field   -> falls back to default
- merge is shallow: nested sections keep their unspecified keys
"""

from __future__ import annotations

import pytest

from rag_mcp_server.config import settings as settings_module
from rag_mcp_server.config.settings import load_settings


@pytest.fixture()
def config_dir(tmp_path):
    """Point load_settings at an empty (or populated) config directory."""
    (tmp_path / "settings.yaml").write_text("", encoding="utf-8")
    return tmp_path


def test_no_config_file_returns_code_defaults(tmp_path):
    # config dir has no settings.yaml
    settings = load_settings(config_dir=tmp_path)
    assert settings["embedding"]["provider"] == "ollama"
    assert settings["embedding"]["model"] == "nomic-embed-text"
    assert settings["vector_store"]["provider"] == "chroma"
    assert settings["retrieval"]["top_k"] == 3


def test_yaml_field_overrides_default(config_dir):
    (config_dir / "settings.yaml").write_text(
        "embedding:\n  model: other-embed\n", encoding="utf-8"
    )
    settings = load_settings(config_dir=config_dir)
    assert settings["embedding"]["model"] == "other-embed"
    # unspecified nested keys keep defaults
    assert settings["embedding"]["provider"] == "ollama"
    assert settings["embedding"]["base_url"] == "http://127.0.0.1:11451"


def test_missing_field_falls_back_to_default(config_dir):
    (config_dir / "settings.yaml").write_text(
        "retrieval:\n  top_k: 5\n", encoding="utf-8"
    )
    settings = load_settings(config_dir=config_dir)
    assert settings["retrieval"]["top_k"] == 5
    assert settings["retrieval"]["rrf_k"] == 60  # not present in yaml


def test_rerank_section_defaults_off_and_merges_override(config_dir):
    """P10.3 + T3: rerank defaults off (enabled false, provider none, model set);
    a yaml section overrides nested keys while unspecified ones keep defaults."""
    settings = load_settings(config_dir=config_dir)  # no yaml rerank section
    assert settings["rerank"] == {
        "enabled": False,
        "provider": "none",
        "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    }

    (config_dir / "settings.yaml").write_text(
        "rerank:\n  enabled: true\n", encoding="utf-8"
    )
    settings = load_settings(config_dir=config_dir)
    assert settings["rerank"]["enabled"] is True
    assert settings["rerank"]["provider"] == "none"  # unspecified keeps default
    assert settings["rerank"]["model"] == "cross-encoder/ms-marco-MiniLM-L-6-v2"


def test_ingestion_transform_section_defaults_off(config_dir):
    """P10.4/P10.6: ingestion.transform defaults use_llm/enabled false; a yaml
    override flips one while the other keeps its default (shallow merge)."""
    settings = load_settings(config_dir=config_dir)  # no yaml ingestion section
    assert settings["ingestion"]["transform"]["chunk_refiner"]["use_llm"] is False
    assert settings["ingestion"]["transform"]["metadata_enricher"]["use_llm"] is False
    assert settings["ingestion"]["transform"]["image_captioner"]["enabled"] is False

    (config_dir / "settings.yaml").write_text(
        "ingestion:\n  transform:\n    chunk_refiner:\n      use_llm: true\n",
        encoding="utf-8",
    )
    settings = load_settings(config_dir=config_dir)
    # NOTE shallow merge: a yaml `ingestion.transform` replaces the whole
    # `transform` dict, so the unspecified metadata_enricher sub-dict is lost —
    # consumers must read defensively with .get (as _transforms_from_settings does).
    assert settings["ingestion"]["transform"]["chunk_refiner"]["use_llm"] is True


def test_unknown_top_level_field_is_kept(config_dir):
    (config_dir / "settings.yaml").write_text(
        "custom_section:\n  flag: true\n", encoding="utf-8"
    )
    settings = load_settings(config_dir=config_dir)
    assert settings["custom_section"] == {"flag": True}


def test_load_settings_accepts_explicit_path(tmp_path):
    yaml_file = tmp_path / "custom.yaml"
    yaml_file.write_text("index_dir: data/custom\n", encoding="utf-8")
    settings = load_settings(path=yaml_file)
    assert settings["index_dir"] == "data/custom"


def test_code_defaults_are_exported():
    # defaults are introspectable so callers can reference them
    assert settings_module.DEFAULTS["llm"]["model"] == "qwen2.5:7b"


def test_absent_section_does_not_alias_defaults(tmp_path):
    # Regression: a missing section must not return the DEFAULTS object itself.
    # Mutating the returned dict would otherwise pollute module-level defaults.
    (tmp_path / "settings.yaml").write_text("# empty\n", encoding="utf-8")
    settings = load_settings(config_dir=tmp_path)
    before = settings_module.DEFAULTS["embedding"]["model"]
    settings["embedding"]["model"] = "HACKED"
    assert settings_module.DEFAULTS["embedding"]["model"] == before
    assert settings["embedding"] is not settings_module.DEFAULTS["embedding"]


def test_partial_yaml_section_returns_distinct_object(tmp_path):
    # A section that is present in yaml but unspecified keys must still be a
    # distinct dict, never the DEFAULTS object.
    (tmp_path / "settings.yaml").write_text(
        "llm:\n  model: gpt-x\n", encoding="utf-8"
    )
    settings = load_settings(config_dir=tmp_path)
    assert settings["embedding"] is not settings_module.DEFAULTS["embedding"]
