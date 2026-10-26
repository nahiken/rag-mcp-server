"""Vision LLM provider factory — instantiate by config provider string.

Mirror of libs/llm/factory.create_llm (ADR 0012 pt.3): ``fake`` returns a
FakeVisionLLM (unit tests / offline); ``ollama`` returns an OllamaVisionLLM
backed by a configured OllamaLLM (real local backend, lazy import keeps unit
tests free of httpx). Unknown providers fail fast.
"""

from __future__ import annotations

from rag_mcp_server.libs.llm.base_vision import FakeVisionLLM

KNOWN_VISION_PROVIDERS = ("fake", "ollama")


def create_vision_llm(provider: str, **kwargs):
    """Build a vision LLM provider by name.

    ``fake`` returns FakeVisionLLM; ``ollama`` returns OllamaVisionLLM wrapping
    an OllamaLLM built from the given model/base_url/timeout (transport is the
    mock seam for unit tests). Unknown providers raise ValueError (fail fast).
    """
    if provider == "fake":
        return FakeVisionLLM(**kwargs)
    if provider == "ollama":
        from rag_mcp_server.libs.llm.ollama import OllamaLLM

        from rag_mcp_server.libs.llm.base_vision import OllamaVisionLLM

        llm = OllamaLLM(
            model=kwargs.get("model", ""),
            base_url=kwargs.get("base_url", ""),
            timeout=kwargs.get("timeout", 30.0),
            transport=kwargs.get("transport"),
        )
        # T2 review FIX-3: no prompt passed here — the captioner owns the single
        # VISION_PROMPT and hands it to chat_with_image, so a real request sends
        # exactly one instruction (previously two near-identical prompts were
        # concatenated: this default + the captioner's).
        return OllamaVisionLLM(llm)
    raise ValueError(
        f"未知 Vision LLM provider: {provider}（可选：{', '.join(KNOWN_VISION_PROVIDERS)}）"
    )
