"""Tests for ImageCaptioner — deterministic caption replacement (P10.6, ticket 09 T1).

Red/green contract (.scratch/phase10/subtickets/image-captioner/T1):
- ImageCaptioner is a BaseTransform: transform(chunks, trace) -> list[Chunk]
- deterministic rule (default, no Vision): [IMAGE: id] placeholder is replaced
  with a readable caption "[图片描述: ...]" derived from the image id
- caption is idempotent: same input text -> same caption
- a chunk with no [IMAGE: id] placeholder is unchanged
- a hard per-chunk failure keeps the original chunk (never blocks the batch)
- settings assembly: image_captioner.enabled=true -> pipeline hangs the
  captioner; disabled/missing -> no captioner, behaviour unchanged

The only seam is the transform module. All offline. Prior art: test_chunk_refiner.py.
"""

from __future__ import annotations

import pytest

from rag_mcp_server.core.types import Chunk
from rag_mcp_server.ingestion.transform.image_captioner import ImageCaptioner


def _chunk(text: str, order: int = 0) -> Chunk:
    return Chunk(
        document_id="doc-pdf",
        text=text,
        source_path="doc.pdf",
        order=order,
        metadata={"file_type": "pdf"},
    )


# -- deterministic caption (default, no Vision) -----------------------------
def test_replaces_image_placeholder_with_readable_caption():
    text = "正文。\n[IMAGE: doc-pdf_p1_1]\n继续。"
    out = ImageCaptioner().transform([_chunk(text)])
    result = out[0].text
    assert "[IMAGE: doc-pdf_p1_1]" not in result  # placeholder gone
    assert "图片" in result  # readable caption present
    assert "doc-pdf_p1_1" in result  # image id kept for traceability


def test_caption_is_idempotent():
    text = "[IMAGE: doc-pdf_p1_1]"
    cap = ImageCaptioner()
    first = cap.transform([_chunk(text)])[0].text
    second = cap.transform([_chunk(text)])[0].text
    assert first == second


def test_multiple_images_get_separate_captions():
    text = "[IMAGE: doc-pdf_p1_1]\n[IMAGE: doc-pdf_p2_1]"
    out = ImageCaptioner().transform([_chunk(text)])[0].text
    assert out.count("图片") == 2
    assert "doc-pdf_p1_1" in out
    assert "doc-pdf_p2_1" in out


def test_chunk_without_image_placeholder_is_unchanged():
    text = "纯文本，没有图片。"
    cap = ImageCaptioner()
    original = _chunk(text)
    out = cap.transform([original])
    assert out[0].text == text
    assert out[0].metadata.get("image_refs") is None


def test_preserves_metadata_and_identity():
    chunk = _chunk("[IMAGE: doc-pdf_p1_1]", order=3)
    out = ImageCaptioner().transform([chunk])
    result = out[0]
    assert result.document_id == "doc-pdf"
    assert result.source_path == "doc.pdf"
    assert result.order == 3
    assert result.metadata["file_type"] == "pdf"


def test_image_refs_metadata_preserved():
    chunk = _chunk("[IMAGE: doc-pdf_p1_1]")
    chunk.metadata["image_refs"] = ["doc-pdf_p1_1"]
    out = ImageCaptioner().transform([chunk])
    assert out[0].metadata["image_refs"] == ["doc-pdf_p1_1"]


def test_per_chunk_failure_keeps_original_chunk():
    # P8: a subclass raises from the real _caption_text seam (no monkeypatch),
    # mirroring test_chunk_refiner.py — the batch must keep the failing chunk.
    class FlakyCaptioner(ImageCaptioner):
        def _caption_text(self, text: str) -> str:
            raise RuntimeError("caption boom")

    chunk = _chunk("[IMAGE: doc-pdf_p1_1]")
    out = FlakyCaptioner().transform([chunk])
    # hard failure keeps the original chunk (BaseTransform skeleton contract)
    assert out[0].text == chunk.text
    assert out[0].metadata.get("caption_error") is not None


def test_batch_other_chunks_continue_after_failure():
    class FlakyCaptioner(ImageCaptioner):
        def _caption_text(self, text: str) -> str:
            raise RuntimeError("caption boom")

    good = _chunk("没有图片的文本")
    bad = _chunk("[IMAGE: doc-pdf_p1_1]")
    out = FlakyCaptioner().transform([bad, good])
    assert out[0].text == bad.text  # bad chunk kept original
    assert out[1].text == good.text  # good chunk continues


# -- settings assembly (default off, consistent with existing contracts) ------
def test_transforms_from_settings_assembles_captioner_when_enabled():
    from rag_mcp_server.application import rag_service

    transforms = rag_service._transforms_from_settings(
        {"ingestion": {"transform": {"image_captioner": {"enabled": True}}}}
    )
    assert any(type(t).__name__ == "ImageCaptioner" for t in transforms)


def test_transforms_from_settings_default_off_has_no_captioner():
    from rag_mcp_server.application import rag_service

    transforms = rag_service._transforms_from_settings({"ingestion": {"transform": {}}})
    assert all(type(t).__name__ != "ImageCaptioner" for t in transforms)


def test_caption_error_is_whitelisted():
    """T1 review FIX-1: caption_error must forward into the vector record,
    mirroring refine_error/enrich_error (audit F4/F6)."""
    from rag_mcp_server.ingestion.pipeline import TRANSFORM_METADATA_KEYS

    assert "caption_error" in TRANSFORM_METADATA_KEYS


def test_content_hash_reflects_caption():
    """T1 review DOC-2: content_hash is computed from the *post-transform*
    text, so a chunk whose caption replaced the [IMAGE: id] placeholder hashes
    differently from the same chunk without caption — the caption is embedded,
    not the mechanical id."""
    from rag_mcp_server.ingestion.pipeline import content_hash_of

    text = "正文。\n[IMAGE: doc-pdf_p1_1]\n继续。"
    out = ImageCaptioner().transform([_chunk(text)])[0]
    assert "[IMAGE: " not in out.text  # placeholder replaced by caption
    assert content_hash_of(out.text) != content_hash_of(text)


# -- Vision upgrade (T2) ------------------------------------------------------
def _img_chunk(text: str, refs: list[str]) -> Chunk:
    chunk = _chunk(text)
    chunk.metadata["image_refs"] = refs
    return chunk


def _storage(tmp_path) -> "object":
    """A minimal ImageStorage writing one file per saved image (offline)."""
    from rag_mcp_server.ingestion.storage.image_storage import ImageStorage

    return ImageStorage(
        db_path=tmp_path / "db" / "image_index.db",
        images_root=tmp_path / "images",
    )


def _seed_image(storage, image_id: str, data: bytes = b"\x89PNG\r\n\x1a\nABC") -> None:
    storage.save_image(image_id, data, mime_type="image/png")


def test_vision_path_uses_semantic_caption(tmp_path):
    from rag_mcp_server.libs.llm.base_vision import FakeVisionLLM

    storage = _storage(tmp_path)
    _seed_image(storage, "doc-pdf_p1_1")
    vision = FakeVisionLLM(responses={"请用一句话描述这张图片的内容。": "一幅红白相间的架构示意图"})
    cap = ImageCaptioner(vision_llm=vision, image_storage=storage, use_vision=True)

    out = cap.transform([_img_chunk("[IMAGE: doc-pdf_p1_1]", ["doc-pdf_p1_1"])])[0]
    assert out.text == "[图片描述: 一幅红白相间的架构示意图]"
    assert out.metadata["image_refs"] == ["doc-pdf_p1_1"]  # refs preserved
    assert vision.call_count == 1


def test_vision_same_image_cached_across_chunks(tmp_path):
    """Q5/US7: the same image referenced by multiple chunks calls Vision once."""
    from rag_mcp_server.libs.llm.base_vision import FakeVisionLLM

    storage = _storage(tmp_path)
    _seed_image(storage, "doc-pdf_p1_1")
    vision = FakeVisionLLM(responses={"请用一句话描述这张图片的内容。": "架构图"})
    cap = ImageCaptioner(vision_llm=vision, image_storage=storage, use_vision=True)

    chunks = [
        _img_chunk("前文 [IMAGE: doc-pdf_p1_1]", ["doc-pdf_p1_1"]),
        _img_chunk("后文 [IMAGE: doc-pdf_p1_1]", ["doc-pdf_p1_1"]),
    ]
    out = cap.transform(chunks)
    assert out[0].text == "前文 [图片描述: 架构图]"
    assert out[1].text == "后文 [图片描述: 架构图]"
    assert vision.call_count == 1  # cached, not per chunk


def test_vision_missing_image_degrades_to_deterministic(tmp_path):
    """Q2: an image not in storage degrades to the deterministic caption, no error."""
    from rag_mcp_server.libs.llm.base_vision import FakeVisionLLM

    storage = _storage(tmp_path)  # image NOT saved
    vision = FakeVisionLLM(responses={"请用一句话描述这张图片的内容。": "不应被用到"})
    cap = ImageCaptioner(vision_llm=vision, image_storage=storage, use_vision=True)

    out = cap.transform([_img_chunk("[IMAGE: doc-pdf_p1_1]", ["doc-pdf_p1_1"])])[0]
    assert out.text == "[图片描述: doc-pdf_p1_1（第 1 页图 1）]"
    assert out.metadata.get("caption_error") is None  # soft degrade, no error key
    assert vision.call_count == 0  # lookup None -> never called Vision


def test_vision_exception_degrades_to_deterministic(tmp_path):
    """Q1/Q2: a Vision/provider exception falls back to the deterministic caption."""
    from rag_mcp_server.libs.llm.base_vision import BaseVisionLLM

    class BoomVision(BaseVisionLLM):
        def chat_with_image(self, text, image_path):
            raise RuntimeError("vision down")

    storage = _storage(tmp_path)
    _seed_image(storage, "doc-pdf_p1_1")
    cap = ImageCaptioner(vision_llm=BoomVision(), image_storage=storage, use_vision=True)

    out = cap.transform([_img_chunk("[IMAGE: doc-pdf_p1_1]", ["doc-pdf_p1_1"])])[0]
    assert out.text == "[图片描述: doc-pdf_p1_1（第 1 页图 1）]"
    assert out.metadata.get("caption_error") is None  # soft degrade, no error key
    assert "[IMAGE:" not in out.text  # placeholder still replaced


def test_vision_disabled_falls_back_to_deterministic(tmp_path):
    """use_vision=False (default) ignores an injected vision_llm."""
    from rag_mcp_server.libs.llm.base_vision import FakeVisionLLM

    storage = _storage(tmp_path)
    _seed_image(storage, "doc-pdf_p1_1")
    vision = FakeVisionLLM(responses={"请用一句话描述这张图片的内容。": "语义"})
    cap = ImageCaptioner(vision_llm=vision, image_storage=storage, use_vision=False)

    out = cap.transform([_img_chunk("[IMAGE: doc-pdf_p1_1]", ["doc-pdf_p1_1"])])[0]
    assert out.text == "[图片描述: doc-pdf_p1_1（第 1 页图 1）]"
    assert vision.call_count == 0


def test_vision_without_storage_is_deterministic(tmp_path):
    """No image_storage: use_vision is inert (Q4 — path cannot be resolved)."""
    from rag_mcp_server.libs.llm.base_vision import FakeVisionLLM

    vision = FakeVisionLLM()
    cap = ImageCaptioner(vision_llm=vision, image_storage=None, use_vision=True)

    out = cap.transform([_img_chunk("[IMAGE: doc-pdf_p1_1]", ["doc-pdf_p1_1"])])[0]
    assert out.text == "[图片描述: doc-pdf_p1_1（第 1 页图 1）]"
    assert vision.call_count == 0


def test_vision_empty_caption_degrades_to_deterministic(tmp_path):
    """A Vision LLM returning empty text degrades to the deterministic caption."""
    from rag_mcp_server.libs.llm.base_vision import FakeVisionLLM

    storage = _storage(tmp_path)
    _seed_image(storage, "doc-pdf_p1_1")
    vision = FakeVisionLLM(responses={"请用一句话描述这张图片的内容。": "   "})  # blank caption
    cap = ImageCaptioner(vision_llm=vision, image_storage=storage, use_vision=True)

    out = cap.transform([_img_chunk("[IMAGE: doc-pdf_p1_1]", ["doc-pdf_p1_1"])])[0]
    assert out.text == "[图片描述: doc-pdf_p1_1（第 1 页图 1）]"
    assert vision.call_count == 1


def test_transforms_from_settings_assembles_captioner_with_vision(tmp_path):
    """T2 assembly: use_vision=true wires a vision_llm + image_storage in."""
    from rag_mcp_server.application import rag_service

    transforms = rag_service._transforms_from_settings(
        {
            "ingestion": {
                "transform": {
                    "image_captioner": {
                        "enabled": True,
                        "use_vision": True,
                        "vision_provider": "fake",
                    }
                }
            }
        }
    )
    captioner = next(t for t in transforms if type(t).__name__ == "ImageCaptioner")
    assert captioner._use_vision is True
    assert captioner._vision_llm is not None
    assert captioner._image_storage is not None


def test_transforms_from_settings_reuses_injected_image_storage(tmp_path):
    """T2 review FIX-1: a caller-supplied ImageStorage is shared with the
    captioner — the captioner's Vision lookup reads exactly the paths the
    pipeline wrote, instead of a sibling data/images default."""
    from rag_mcp_server.application import rag_service
    from rag_mcp_server.ingestion.storage.image_storage import ImageStorage

    storage = ImageStorage(db_path=tmp_path / "db" / "image_index.db", images_root=tmp_path / "images")
    transforms = rag_service._transforms_from_settings(
        {
            "ingestion": {
                "transform": {
                    "image_captioner": {
                        "enabled": True,
                        "use_vision": True,
                        "vision_provider": "fake",
                    }
                }
            }
        },
        image_storage=storage,
    )
    captioner = next(t for t in transforms if type(t).__name__ == "ImageCaptioner")
    assert captioner._image_storage is storage  # same instance, not a rebuild


def test_default_pipeline_wires_shared_image_storage_when_captioner_enabled(
    tmp_path, monkeypatch
):
    """T2 review FIX-1 (headline): the settings-driven pipeline must carry an
    image_storage so the ingest path really extracts PDF images -> [IMAGE: id]
    placeholders -> Vision captions. The captioner shares that same instance
    (not a sibling default), so its lookup reads the paths the pipeline wrote."""
    import rag_mcp_server.config.settings as settings_mod
    from rag_mcp_server.application import rag_service
    from rag_mcp_server.ingestion.storage.image_storage import ImageStorage

    monkeypatch.chdir(tmp_path)  # ImageStorage() default root is CWD-relative
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    (cfg_dir / "settings.yaml").write_text(
        "embedding:\n  provider: fake\n"
        "vector_store:\n  provider: chroma\n  persist_dir: null\n  collection: test_fix1_enabled\n"
        "ingestion:\n  transform:\n    image_captioner:\n      enabled: true\n"
        "      use_vision: true\n      vision_provider: fake\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(settings_mod, "DEFAULT_CONFIG_DIR", cfg_dir)

    pipe = rag_service._default_pipeline(tmp_path / "index")
    captioner = next(t for t in pipe.transforms if type(t).__name__ == "ImageCaptioner")
    assert isinstance(pipe.image_storage, ImageStorage)
    assert pipe.image_storage is not None
    assert captioner._image_storage is pipe.image_storage  # shared instance
    assert captioner._use_vision is True


def test_default_pipeline_no_image_storage_when_captioner_disabled(tmp_path, monkeypatch):
    """T2 review FIX-1: the default (captioner disabled) keeps image_storage
    None — no image extraction, existing ingest behaviour unchanged."""
    import rag_mcp_server.config.settings as settings_mod
    from rag_mcp_server.application import rag_service

    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    (cfg_dir / "settings.yaml").write_text(
        "embedding:\n  provider: fake\n"
        "vector_store:\n  provider: chroma\n  persist_dir: null\n  collection: test_fix1_off\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(settings_mod, "DEFAULT_CONFIG_DIR", cfg_dir)

    pipe = rag_service._default_pipeline(tmp_path / "index")
    assert pipe.image_storage is None
    assert all(type(t).__name__ != "ImageCaptioner" for t in pipe.transforms)


# -- pipeline integration: caption replaces placeholder in embedded text -------
def test_pipeline_with_captioner_embeds_caption_not_placeholder(tmp_path):
    """T1 acceptance: ingest an image PDF with captioner enabled -> the chunk
    text embedded in the store contains the readable caption, not [IMAGE: id]."""
    from rag_mcp_server.ingestion.pipeline import IngestPipeline
    from rag_mcp_server.ingestion.storage.image_storage import ImageStorage
    from rag_mcp_server.ingestion.transform.image_captioner import ImageCaptioner
    from rag_mcp_server.libs.embedding.fake import FakeEmbedding
    from rag_mcp_server.libs.vector_store.chroma_store import ChromaVectorStore

    # reuse the text+image PDF fixture builder (kept inline to avoid cross-test deps)
    import io
    from PIL import Image as PILImage

    img = PILImage.new("RGB", (8, 8), "red")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    jpeg = buf.getvalue()
    content = b"BT /F1 24 Tf 72 720 Td (Hello image PDF) Tj ET\nq 100 0 0 100 72 500 cm /Im1 Do Q"
    hdr = (
        b"6 0 obj << /Type /XObject /Subtype /Image /Width 8 /Height 8 "
        b"/ColorSpace /DeviceRGB /Filter /DCTDecode /Length "
    )
    body6 = hdr + str(len(jpeg)).encode() + b" >> stream\n" + jpeg + b"\nendstream endobj\n"
    bodies = [
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n",
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n",
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> /XObject << /Im1 6 0 R >> >> >> endobj\n",
        b"4 0 obj << /Length " + str(len(content)).encode() + b" >> stream\n" + content + b"\nendstream endobj\n",
        b"5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n",
        body6,
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for body in bodies:
        offsets.append(len(out))
        out += body
    xref = len(out)
    out += b"xref\n0 " + str(len(bodies) + 1).encode() + b"\n"
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        b"trailer << /Size " + str(len(bodies) + 1).encode()
        + b" /Root 1 0 R >>\nstartxref\n" + str(xref).encode() + b"\n%%EOF\n"
    )

    src = tmp_path / "source"
    src.mkdir()
    (src / "doc.pdf").write_bytes(bytes(out))
    embedding = FakeEmbedding(dim=4)
    store = ChromaVectorStore(
        persist_dir=str(tmp_path / "storage" / "chroma"), collection="test_captioner_e2e"
    )
    image_storage = ImageStorage(
        db_path=tmp_path / "db" / "image_index.db",
        images_root=tmp_path / "images",
    )
    pipe = IngestPipeline(
        embedding=embedding,
        store=store,
        index_dir=tmp_path / "storage" / "index",
        image_storage=image_storage,
        transforms=[ImageCaptioner()],
    )
    pipe.ingest(src, base=src)
    records = store.get_all()
    assert records
    # embedded text has the readable caption, no mechanical placeholder
    assert all("[IMAGE: " not in r.text for r in records)
    assert any("图片" in r.text for r in records)
    # image_refs still present (whitelisted metadata survives)
    assert any(r.metadata.get("image_refs") for r in records)
