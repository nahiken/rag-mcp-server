"""LLM providers: messages -> text, pluggable via factory.

Top-level convenience exports for consumers (P10.3/P10.4/P10.7). LLMError is
defined in core/errors and re-exported here so consumers need not know where
it lives. P10.6 T2 adds the vision extension (BaseVisionLLM/ImageInput/Fake/
Ollama + create_vision_llm), re-exported here too.
"""

from __future__ import annotations

from rag_mcp_server.core.errors import LLMError
from rag_mcp_server.libs.llm.base import BaseLLM
from rag_mcp_server.libs.llm.base_vision import (
    BaseVisionLLM,
    FakeVisionLLM,
    ImageInput,
    OllamaVisionLLM,
)
from rag_mcp_server.libs.llm.factory import create_llm
from rag_mcp_server.libs.llm.fake import FakeLLM
from rag_mcp_server.libs.llm.vision_factory import create_vision_llm

__all__ = [
    "BaseLLM",
    "create_llm",
    "FakeLLM",
    "LLMError",
    "BaseVisionLLM",
    "ImageInput",
    "FakeVisionLLM",
    "OllamaVisionLLM",
    "create_vision_llm",
]
