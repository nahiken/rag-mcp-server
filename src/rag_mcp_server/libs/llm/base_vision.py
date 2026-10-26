"""Vision LLM contract — chat with an image (P10.6, ticket 09 T2).

Visual extension of the pluggable LLM layer (P10.2, ADR 0008/0012). The text
path ``BaseLLM.chat(messages) -> str`` already forwards extra message keys
(``images``) verbatim to the backend; this module adds the vision vocabulary on
top: ``BaseVisionLLM.chat_with_image(text, image_path) -> str``.

- ``BaseVisionLLM`` — abstract single-text-single-image chat (spec P10.6).
- ``ImageInput`` — value object; exactly one of ``path``/``data``/``base64``
  must be given (``__post_init__`` validates), exposing ``mime`` and
  ``as_bytes()`` so both real and fake providers consume one shape.
- ``FakeVisionLLM`` — deterministic echo (mirrors FakeLLM): no network, records
  ``calls``/``call_count`` so "same image captioned once" assertions read them
  directly.
- ``OllamaVisionLLM`` — real provider backed by a local Ollama server: wraps an
  injected ``OllamaLLM`` and sends the image as the ``images`` message key
  (``data:{mime};base64,...``). Failures map to LLMError via the wrapped LLM.

The captioner (ingestion/transform/image_captioner.py) is the only consumer so
far (T2). See docs/adr/0012-vision-llm.md.
"""

from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

__all__ = ["BaseVisionLLM", "ImageInput", "FakeVisionLLM", "OllamaVisionLLM"]


@dataclass
class ImageInput:
    """Exactly one of path/data/base64; exposes mime + bytes for providers."""

    path: Path | None = None
    data: bytes | None = None
    base64: str | None = None
    mime: str = "image/png"

    def __post_init__(self) -> None:
        provided = sum(x is not None for x in (self.path, self.data, self.base64))
        if provided != 1:
            raise ValueError(
                f"ImageInput 必须且只能提供 path/data/base64 之一（实际 {provided} 个）"
            )
        if self.path is not None:
            path = Path(self.path)
            if not path.is_file():
                raise FileNotFoundError(f"图片文件不存在：{path}")
        if self.base64 is not None and not self.base64.strip():
            raise ValueError("ImageInput.base64 不能为空字符串")

    @property
    def source(self) -> str:
        """Short human label of the input kind (for prompts/logs).

        Uniform tag across all three kinds (T2 review FIX-6): "path",
        "base64-image" or "inline-bytes" — never a bare path string, so
        callers/logs can rely on one stable shape.
        """
        if self.path is not None:
            return "path"
        if self.base64 is not None:
            return "base64-image"
        return "inline-bytes"

    def as_bytes(self) -> bytes:
        """The decoded image bytes regardless of which input kind was used."""
        if self.path is not None:
            return Path(self.path).read_bytes()
        if self.base64 is not None:
            return base64.b64decode(self.base64)
        assert self.data is not None  # __post_init__ guarantees exactly one
        return self.data

    def as_data_url(self) -> str:
        """RFC 2397 data URL (``data:{mime};base64,...``) for message payloads."""
        return f"data:{self.mime};base64,{base64.b64encode(self.as_bytes()).decode('ascii')}"


class BaseVisionLLM(ABC):
    """Chat with a single text prompt plus a single image; returns text."""

    @abstractmethod
    def chat_with_image(self, text: str, image_path: Path | bytes) -> str:
        """Return the model's description/reply for ``text`` + ``image_path``."""


class FakeVisionLLM(BaseVisionLLM):
    """Deterministic vision LLM; echoes the prompt unless a table matches.

    Mirrors FakeLLM (libs/llm/fake.py): deterministic, no network, records every
    call in ``calls``/``call_count`` so cache/fallback assertions read them
    directly. Accepts and ignores extra kwargs (aligned with FakeLLM/FakeEmbedding).
    """

    def __init__(self, responses: dict[str, str] | None = None, **kwargs) -> None:
        self.responses = responses or {}
        self.calls: list[dict] = []
        self.call_count = 0

    def chat_with_image(self, text: str, image_path: Path | bytes) -> str:
        self.calls.append({"text": text, "image_path": image_path})
        self.call_count += 1
        if text in self.responses:
            return self.responses[text]
        return text


class OllamaVisionLLM(BaseVisionLLM):
    """Vision via a local Ollama server (``/api/chat`` with an ``images`` key).

    Wraps an injected ``OllamaLLM`` (constructed by the factory with the
    configured vision model/base_url); the image is sent as the ``images``
    message key ``data:{mime};base64,...`` (the P10.2 extension point). All
    provider failures map to LLMError through the wrapped chat call.

    T2 review FIX-3: the prompt is owned by the caller (the captioner's
    ``VISION_PROMPT``) — this class never prepends its own default, so a real
    request sends exactly one instruction, not two concatenated ones.
    """

    def __init__(self, llm) -> None:
        self._llm = llm

    def chat_with_image(self, text: str, image_path: Path | bytes) -> str:
        if isinstance(image_path, (str, Path)):
            image = ImageInput(path=Path(image_path))
        else:
            image = ImageInput(data=image_path)
        messages = [
            {
                "role": "user",
                "content": text,
                "images": [image.as_data_url()],
            }
        ]
        return self._llm.chat(messages)
