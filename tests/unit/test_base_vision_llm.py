"""Tests for the Vision LLM layer — BaseVisionLLM/ImageInput/Fake/Ollama/factory.

Red/green contract (docs/adr/0012-vision-llm.md, ticket 09 T2):
- ImageInput: exactly one of path/data/base64; __post_init__ validates;
  as_bytes()/as_data_url() round-trip; mime carried
- FakeVisionLLM: deterministic echo of the prompt; records calls/call_count;
  accepts **kwargs (mirrors FakeLLM)
- OllamaVisionLLM: wraps an OllamaLLM; sends the image as the `images` message
  key (data:{mime};base64,...); failures map to LLMError via the wrapped chat
- create_vision_llm: fake -> FakeVisionLLM; ollama -> OllamaVisionLLM; unknown
  provider -> ValueError (fail fast)

The only seam is the libs/llm/base_vision.py module. All offline.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from rag_mcp_server.core.errors import LLMError, RagMcpError
from rag_mcp_server.libs.llm.base_vision import (
    BaseVisionLLM,
    FakeVisionLLM,
    ImageInput,
    OllamaVisionLLM,
)


# ---------------------------------------------------------------------------
# ImageInput — exactly-one validation + bytes/mime
# ---------------------------------------------------------------------------
def test_image_input_requires_exactly_one_source(tmp_path):
    p = tmp_path / "img.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\nfakedata")
    ok = ImageInput(path=p)
    assert ok.as_bytes() == p.read_bytes()

    with pytest.raises(ValueError):
        ImageInput()  # zero sources
    with pytest.raises(ValueError):
        ImageInput(path=p, data=b"x")  # two sources
    with pytest.raises(ValueError):
        ImageInput(base64="")  # empty base64


def test_image_input_path_must_exist(tmp_path):
    with pytest.raises(FileNotFoundError):
        ImageInput(path=tmp_path / "missing.png")


def test_image_input_data_and_base64_roundtrip():
    by_data = ImageInput(data=b"\x89PNG\x00\x01")
    assert by_data.as_bytes() == b"\x89PNG\x00\x01"
    assert by_data.source == "inline-bytes"

    by_b64 = ImageInput(base64="aGVsbG8=", mime="image/jpeg")
    assert by_b64.as_bytes() == b"hello"
    assert by_b64.mime == "image/jpeg"
    assert by_b64.source == "base64-image"


def test_image_input_source_is_uniform_tag_not_path(tmp_path):
    """T2 review FIX-6: ``source`` is a stable tag for every input kind — a
    path input yields "path", not a bare filesystem path string."""
    p = tmp_path / "img.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\nABC")
    assert ImageInput(path=p).source == "path"
    assert ImageInput(data=b"x").source == "inline-bytes"
    assert ImageInput(base64="eA==").source == "base64-image"


def test_image_input_as_data_url():
    inp = ImageInput(data=b"hi", mime="image/png")
    assert inp.as_data_url() == "data:image/png;base64,aGk="


# ---------------------------------------------------------------------------
# BaseVisionLLM — abstract contract
# ---------------------------------------------------------------------------
class EchoVision(BaseVisionLLM):
    def chat_with_image(self, text, image_path):
        return f"{text}|{Path(image_path).name}"


def test_base_vision_llm_is_abstract():
    with pytest.raises(TypeError):
        BaseVisionLLM()  # type: ignore[abstract]


def test_vision_subclass_returns_text():
    v = EchoVision()
    assert v.chat_with_image("hello", Path("a.png")) == "hello|a.png"


# ---------------------------------------------------------------------------
# FakeVisionLLM — deterministic echo + call recording
# ---------------------------------------------------------------------------
def test_fake_vision_echoes_prompt():
    v = FakeVisionLLM()
    assert v.chat_with_image("描述这张图", Path("a.png")) == "描述这张图"


def test_fake_vision_responses_table_overrides():
    v = FakeVisionLLM(responses={"描述这张图": "一张架构图"})
    assert v.chat_with_image("描述这张图", Path("a.png")) == "一张架构图"


def test_fake_vision_records_calls_and_call_count():
    v = FakeVisionLLM()
    v.chat_with_image("q1", Path("a.png"))
    v.chat_with_image("q2", b"bytes")
    assert v.call_count == 2
    assert v.calls == [
        {"text": "q1", "image_path": Path("a.png")},
        {"text": "q2", "image_path": b"bytes"},
    ]


def test_fake_vision_ignores_extra_kwargs():
    v = FakeVisionLLM(model="ignored")
    assert v.chat_with_image("hi", Path("a.png")) == "hi"


# ---------------------------------------------------------------------------
# OllamaVisionLLM — wraps OllamaLLM, sends `images` message key
# ---------------------------------------------------------------------------
def make_vision_llm(responses: list[dict]) -> tuple[OllamaVisionLLM, object]:
    """OllamaVisionLLM over a MockTransport replaying canned JSON responses."""
    from rag_mcp_server.libs.llm.ollama import OllamaLLM

    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append({"body": __import__("json").loads(request.content)})
        resp = responses.pop(0)
        return httpx.Response(status_code=resp.get("status", 200), json=resp.get("json", resp))

    transport = httpx.MockTransport(handler)
    transport.calls = calls  # type: ignore[attr-defined]
    llm = OllamaLLM(model="vision", base_url="http://127.0.0.1:11451", transport=transport)
    return OllamaVisionLLM(llm), transport


def test_ollama_vision_sends_image_as_data_url_message_key(tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\nABC")
    v, t = make_vision_llm([{"json": {"message": {"role": "assistant", "content": "一张架构图"}}}])

    out = v.chat_with_image("这张图是什么？", img)
    assert out == "一张架构图"

    msg = t.calls[0]["body"]["messages"][0]  # type: ignore[attr-defined]
    # FIX-3: the prompt is the caller's text verbatim — exactly ONE instruction,
    # no second default prompt prepended (a double prompt used to be sent).
    assert msg["content"] == "这张图是什么？"
    assert msg["images"][0].startswith("data:image/png;base64,")


def test_ollama_vision_sends_single_instruction_not_double_prompt(tmp_path):
    """T2 review FIX-3: a real request body carries one instruction — the
    captioner's VISION_PROMPT — not that prompt concatenated with the
    OllamaVisionLLM's own default ("请用一句话描述这张图片。")."""
    from rag_mcp_server.ingestion.transform.image_captioner import VISION_PROMPT

    img = tmp_path / "a.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\nABC")
    v, t = make_vision_llm([{"json": {"message": {"role": "assistant", "content": "架构图"}}}])

    v.chat_with_image(VISION_PROMPT, img)
    msg = t.calls[0]["body"]["messages"][0]  # type: ignore[attr-defined]
    assert msg["content"] == VISION_PROMPT
    assert "请用一句话描述这张图片。\n" not in msg["content"]  # no double prompt


def test_ollama_vision_accepts_bytes_input():
    v, t = make_vision_llm([{"json": {"message": {"role": "assistant", "content": "ok"}}}])
    out = v.chat_with_image("q", b"\x89PNG\r\n\x1a\nABC")
    assert out == "ok"
    msg = t.calls[0]["body"]["messages"][0]  # type: ignore[attr-defined]
    assert msg["images"][0].startswith("data:image/png;base64,")


def test_ollama_vision_provider_failure_maps_to_llm_error():
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    from rag_mcp_server.libs.llm.ollama import OllamaLLM

    llm = OllamaLLM(model="m", transport=httpx.MockTransport(boom))
    v = OllamaVisionLLM(llm)
    with pytest.raises(LLMError):
        v.chat_with_image("q", b"\x89PNG\r\n\x1a\nABC")


def test_ollama_vision_error_is_rag_mcp_error():
    assert issubclass(LLMError, RagMcpError)


# ---------------------------------------------------------------------------
# create_vision_llm — factory (fail fast)
# ---------------------------------------------------------------------------
def test_create_vision_fake_returns_fake_vision():
    from rag_mcp_server.libs.llm.vision_factory import create_vision_llm

    assert isinstance(create_vision_llm("fake"), FakeVisionLLM)


def test_create_vision_ollama_returns_ollama_vision():
    from rag_mcp_server.libs.llm.vision_factory import create_vision_llm

    v = create_vision_llm("ollama", model="m", base_url="http://127.0.0.1:11451")
    assert isinstance(v, OllamaVisionLLM)


def test_create_vision_unknown_provider_raises_value_error():
    from rag_mcp_server.libs.llm.vision_factory import create_vision_llm

    with pytest.raises(ValueError) as ei:
        create_vision_llm("openai")
    assert "openai" in str(ei.value)
    assert "fake" in str(ei.value)
    assert "ollama" in str(ei.value)


# ---------------------------------------------------------------------------
# Package exports — libs/llm/__init__.py
# ---------------------------------------------------------------------------
def test_libs_llm_exports_vision_symbols():
    import rag_mcp_server.libs.llm as llm_pkg

    assert llm_pkg.BaseVisionLLM is BaseVisionLLM
    assert llm_pkg.ImageInput is ImageInput
    assert llm_pkg.FakeVisionLLM is FakeVisionLLM
    assert llm_pkg.OllamaVisionLLM is OllamaVisionLLM
    assert callable(llm_pkg.create_vision_llm)
