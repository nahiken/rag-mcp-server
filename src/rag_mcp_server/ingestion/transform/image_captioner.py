"""ImageCaptioner — generate captions for referenced images (P10.6, ticket 09).

BaseTransform that replaces the ``[IMAGE: id]`` placeholder (emitted by the
pdfplumber image channel, ticket 08) with a readable caption. Default behaviour
is a **deterministic** rule (Q7): the caption is derived from the image id
(``{document_id}_p{page}_{seq}``), so it is idempotent, offline, and needs no
Vision LLM. The caption replaces the placeholder, so the final embedded text
has no mechanical id and ``content_hash`` reflects the caption (ADR 0010).

Vision upgrade (ticket 09 T2): an injected BaseVisionLLM + ImageStorage replaces
the deterministic caption with a **semantic** one per image. Same image
referenced by multiple chunks is captioned once (instance-level cache, rebuilt
per captioner construction). Vision disabled/unavailable/exception or a missing
image_storage degrades to the deterministic caption (Q2) — the placeholder
still gets replaced so the image stays searchable, and ``caption_error`` is
reserved for hard per-chunk failures (Q1): soft Vision degradation records no
error key.

Contract (docs/completion-plan-p10.md P10.6 + issue 09):
- ``[IMAGE: id]`` -> ``[图片描述: caption]`` (readable, idempotent by default)
- chunk with no placeholder is unchanged
- a hard per-chunk failure keeps the original chunk (BaseTransform skeleton),
  marking ``caption_error`` — never blocks the batch
- Vision soft-degradation keeps the deterministic caption, image_refs intact,
  no error marker
- injected keys stay on chunk.metadata (image_refs is preserved; the caption
  lives in the text, not metadata)
"""

from __future__ import annotations

from pathlib import Path

from rag_mcp_server.core.types import Chunk
from rag_mcp_server.ingestion.pdf_images import IMAGE_ID_SUFFIX_RE, IMAGE_PLACEHOLDER_RE
from rag_mcp_server.ingestion.storage.image_storage import ImageStorage
from rag_mcp_server.ingestion.transform.base import BaseTransform, _with_metadata
from rag_mcp_server.libs.llm.base_vision import BaseVisionLLM

# The prompt handed to a Vision LLM when use_vision is on. Concise and
# deterministic-friendly; the caption text is the only free-form part.
VISION_PROMPT = "请用一句话描述这张图片的内容。"


def deterministic_caption(image_id: str) -> str:
    """A readable, idempotent caption derived from the image id.

    ``doc-pdf_p1_1`` -> ``doc-pdf_p1_1（第 1 页图 1）``. The id is kept so the
    caption stays traceable back to the stored image; the page/seq give a
    human hint. Unparsable ids fall back to the id alone.
    """
    match = IMAGE_ID_SUFFIX_RE.search(image_id)
    if match:
        page, seq = match.group(1), match.group(2)
        return f"{image_id}（第 {page} 页图 {seq}）"
    return image_id


def caption_text_for(text: str) -> str:
    """Replace every ``[IMAGE: id]`` placeholder with a deterministic caption."""
    def _replace(match: re.Match) -> str:
        image_id = match.group(1).strip()
        return f"[图片描述: {deterministic_caption(image_id)}]"

    return IMAGE_PLACEHOLDER_RE.sub(_replace, text)


class ImageCaptioner(BaseTransform):
    """Caption referenced images by deterministic rule or an injected Vision LLM."""

    def __init__(
        self,
        vision_llm: BaseVisionLLM | None = None,
        image_storage: ImageStorage | None = None,
        use_vision: bool = False,
    ) -> None:
        self._vision_llm = vision_llm
        self._image_storage = image_storage
        self._use_vision = use_vision and vision_llm is not None and image_storage is not None
        # Q5: instance-level cache — the same image_id is captioned once even
        # when many chunks reference it (US7: "同一图片只调用一次 Vision").
        # Rebuilt on every captioner construction (each transform assembles its
        # own instance); not process-global (T2 review FIX-6).
        self._vision_cache: dict[str, str] = {}

    def _caption_text(self, text: str) -> str:
        """Replace every ``[IMAGE: id]`` placeholder with a caption.

        Vision path (use_vision): each image is looked up in image_storage and
        handed to the injected Vision LLM for a semantic caption; failures /
        missing images fall back to the deterministic caption (Q2), never
        raising. Deterministic path (default): the pure-rule caption.
        """
        if not self._use_vision:
            return caption_text_for(text)

        def _replace(match: re.Match) -> str:
            image_id = match.group(1).strip()
            return f"[图片描述: {self._caption_for(image_id)}]"

        return IMAGE_PLACEHOLDER_RE.sub(_replace, text)

    def _caption_for(self, image_id: str) -> str:
        """Semantic caption for one image, cached; deterministic on any failure.

        ``image_storage.lookup`` returns a Path (the stored file). A missing
        image (lookup None) or any Vision/provider exception degrades to the
        deterministic caption (Q2) — the placeholder is still replaced so the
        image stays searchable, and soft degradation records no error (Q1).
        """
        cached = self._vision_cache.get(image_id)
        if cached is not None:
            return cached
        try:
            path = self._image_storage.lookup(image_id)
            caption = (
                self._vision_llm.chat_with_image(VISION_PROMPT, path)
                if path is not None
                else deterministic_caption(image_id)
            )
            if not caption or not caption.strip():
                caption = deterministic_caption(image_id)
        except Exception:  # noqa: BLE001 — soft degrade, never block the batch
            caption = deterministic_caption(image_id)
        self._vision_cache[image_id] = caption
        return caption

    def _process_one(self, chunk: Chunk) -> Chunk:
        try:
            new_text = self._caption_text(chunk.text)
            if new_text == chunk.text:
                return chunk  # no placeholder -> unchanged
            return _with_metadata(chunk, {}, text=new_text)
        except Exception as exc:  # noqa: BLE001 — per-chunk degradation
            return _with_metadata(
                chunk,
                {"caption_error": str(exc)},
            )
