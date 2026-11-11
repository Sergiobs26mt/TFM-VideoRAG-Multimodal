"""Pruebas unitarias de ``local_vectorizer`` (modelo simulado, sin pesos ni red)."""

from __future__ import annotations

import asyncio
import importlib
import time
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image

from src.embeddings import local_vectorizer
from src.embeddings.local_vectorizer import (
    IMAGE_TAG,
    QWEN_QUERY_INSTRUCTION,
    LocalMultimodalVectorizer,
    load_sentence_transformer,
    select_device,
    torch_dtype_for,
)
from src.models import MultimodalQuery, VectorizationError, VectorizedChunk, VideoChunk

GEMMA_PROMPTS = {
    "Document": "title: none | text: ",
    "SearchQuery": "task: search result | query: ",
}
CHUNK = VideoChunk(
    index=3, start_frame=0, end_frame=9, start_time=1.5, end_time=4.0, transcript="hola"
)


class FakeModel:
    """Sustituye a SentenceTransformer registrando llamadas y concurrencia."""

    def __init__(
        self,
        prompts: dict[str, str] | None = None,
        size: int = 768,
        result: NDArray[np.float32] | None = None,
        delay: float = 0.0,
    ) -> None:
        self.prompts = prompts or {}
        self._result = (
            result if result is not None else np.arange(1, size + 1, dtype=np.float32)
        )
        self._delay = delay
        self.calls: list[tuple[Any, dict[str, Any]]] = []
        self.active = 0
        self.peak = 0

    def encode(
        self, inputs: Any, convert_to_numpy: bool = True, **kwargs: Any
    ) -> NDArray[np.float32]:
        self.active += 1
        self.peak = max(self.peak, self.active)
        time.sleep(self._delay)
        self.active -= 1
        self.calls.append((inputs, kwargs))
        return self._result


def _frame(bgr: tuple[int, int, int] = (255, 0, 0)) -> NDArray[np.uint8]:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    frame[:, :] = bgr
    return frame


def _frames(count: int = 3) -> list[NDArray[np.uint8]]:
    return [_frame() for _ in range(count)]


def _png(frame: NDArray[np.uint8]) -> bytes:
    ok, buffer = cv2.imencode(".png", frame)
    assert ok
    return buffer.tobytes()


def _qwen(model: FakeModel | None = None, dim: int = 768) -> LocalMultimodalVectorizer:
    return LocalMultimodalVectorizer(
        "qwen3-vl-2b", model or FakeModel(size=2048), "mps", dim
    )


def _gemma(model: FakeModel | None = None, dim: int = 768) -> LocalMultimodalVectorizer:
    return LocalMultimodalVectorizer(
        "gemma2", model or FakeModel(GEMMA_PROMPTS), "mps", dim
    )


def _fake_torch(mps: bool) -> SimpleNamespace:
    return SimpleNamespace(
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
        bfloat16="bf16",
        float32="f32",
    )


def test_select_device_prefers_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(local_vectorizer, "_import_torch", lambda: _fake_torch(True))
    assert select_device() == "mps"


def test_select_device_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(local_vectorizer, "_import_torch", lambda: _fake_torch(False))
    assert select_device() == "cpu"


def test_torch_dtype_for_is_bfloat16_on_mps_and_float32_on_cpu() -> None:
    torch_module = _fake_torch(True)
    assert torch_dtype_for("mps", torch_module) == "bf16"
    assert torch_dtype_for("cpu", torch_module) == "f32"


def test_load_sentence_transformer_import_error_mentions_install_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = importlib.import_module

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "sentence_transformers":
            raise ImportError("no instalado")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    with pytest.raises(VectorizationError, match=r"sentence-transformers\[image\]"):
        load_sentence_transformer("google/embeddinggemma-2", "cpu")


def test_unknown_backend_raises_value_error() -> None:
    with pytest.raises(ValueError):
        LocalMultimodalVectorizer("gemini", FakeModel(), "cpu", 3072)


def test_device_label_follows_device() -> None:
    assert _qwen().device_label == "Local MPS"
    cpu = LocalMultimodalVectorizer("gemma2", FakeModel(GEMMA_PROMPTS), "cpu", 768)
    assert cpu.device_label == "Local CPU"
    assert cpu.backend == "gemma2"


def test_qwen_chunk_input_has_three_rgb_pil_images_and_transcript() -> None:
    inputs = _qwen().build_chunk_input("hola", _frames())
    assert isinstance(inputs, dict)
    assert inputs["text"] == "hola"
    images = inputs["image"]
    assert len(images) == 3
    assert all(isinstance(img, Image.Image) and img.mode == "RGB" for img in images)
    # BGR (255, 0, 0) es azul y debe llegar como RGB (0, 0, 255)
    assert images[0].getpixel((0, 0)) == (0, 0, 255)


def test_qwen_chunk_without_transcript_omits_text_key() -> None:
    for transcript in (None, "   "):
        inputs = _qwen().build_chunk_input(transcript, _frames())
        assert isinstance(inputs, dict)
        assert "text" not in inputs
        assert len(inputs["image"]) == 3


def test_qwen_query_uses_retrieval_instruction() -> None:
    inputs, kwargs = _qwen().build_query_input("que es una variable")
    assert inputs == "que es una variable"
    assert kwargs == {"prompt": QWEN_QUERY_INSTRUCTION}


def test_qwen_image_query_is_dict_and_image_only_omits_text() -> None:
    both, _ = _qwen().build_query_input(
        MultimodalQuery(text="rojo", images=(_png(_frame()),))
    )
    only, kwargs = _qwen().build_query_input(MultimodalQuery(images=(_png(_frame()),)))
    assert isinstance(both, dict) and both["text"] == "rojo" and len(both["image"]) == 1
    assert isinstance(only, dict) and "text" not in only
    assert kwargs == {"prompt": QWEN_QUERY_INSTRUCTION}


def test_gemma_chunk_text_has_document_prefix_and_three_tags() -> None:
    with_text = _gemma().build_chunk_input("hola", _frames())
    without_text = _gemma().build_chunk_input(None, _frames())
    assert isinstance(with_text, dict) and isinstance(without_text, dict)
    assert with_text["text"] == "title: none | text: <|image|><|image|><|image|> hola"
    assert without_text["text"] == "title: none | text: <|image|><|image|><|image|>"
    assert len(with_text["image"]) == 3


def test_gemma_text_query_uses_search_query_prompt_name() -> None:
    inputs, kwargs = _gemma().build_query_input("que es una variable")
    assert inputs == "que es una variable"
    assert kwargs == {"prompt_name": "SearchQuery"}


def test_gemma_image_query_has_prefix_and_one_tag_per_image() -> None:
    inputs, kwargs = _gemma().build_query_input(
        MultimodalQuery(text="rojo", images=(_png(_frame()), _png(_frame())))
    )
    assert isinstance(inputs, dict)
    assert inputs["text"] == "task: search result | query: <|image|><|image|> rojo"
    assert len(inputs["image"]) == 2
    assert kwargs == {}


def test_gemma_image_only_query_has_tags_without_prefix() -> None:
    inputs, kwargs = _gemma().build_query_input(
        MultimodalQuery(images=(_png(_frame()), _png(_frame()), _png(_frame())))
    )
    assert isinstance(inputs, dict)
    assert inputs["text"] == IMAGE_TAG * 3
    assert kwargs == {}


def test_embedding_is_truncated_and_l2_normalized() -> None:
    vec = _gemma(FakeModel(GEMMA_PROMPTS, size=768), dim=256)
    result = asyncio.run(vec.embed_text("hola"))
    assert len(result) == 256
    assert float(np.linalg.norm(result)) == pytest.approx(1.0, rel=1e-6)
    expected = np.arange(1, 257, dtype=np.float64)
    assert np.allclose(result, expected / np.linalg.norm(expected), atol=1e-6)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_output_raises_vectorization_error(bad: float) -> None:
    result = np.ones(768, dtype=np.float32)
    result[5] = bad
    vec = _gemma(FakeModel(GEMMA_PROMPTS, result=result))
    with pytest.raises(VectorizationError):
        asyncio.run(vec.embed_text("hola"))


def test_encode_calls_are_serialized() -> None:
    model = FakeModel(GEMMA_PROMPTS, delay=0.01)
    vec = _gemma(model)

    async def run() -> None:
        await asyncio.gather(*(vec.embed_text(f"q{i}") for i in range(4)))

    asyncio.run(run())
    assert len(model.calls) == 4
    assert model.peak == 1


def test_empty_query_raises_value_error() -> None:
    vec = _gemma()
    with pytest.raises(ValueError):
        asyncio.run(vec.embed_query(MultimodalQuery(text="  ")))
    with pytest.raises(ValueError):
        asyncio.run(vec.embed_text(""))


def test_corrupt_query_image_raises_value_error() -> None:
    vec = _gemma()
    with pytest.raises(ValueError, match="corrupta"):
        asyncio.run(vec.embed_query(MultimodalQuery(images=(b"not-an-image",))))


def test_vectorize_chunk_returns_vectorized_chunk_with_metadata() -> None:
    model = FakeModel(GEMMA_PROMPTS)
    result = asyncio.run(_gemma(model, dim=512).vectorize_chunk(CHUNK, _frames()))
    assert isinstance(result, VectorizedChunk)
    assert (result.index, result.start_time, result.end_time) == (3, 1.5, 4.0)
    assert result.transcript == "hola"
    assert len(result.embedding) == 512
    assert len(model.calls) == 1


def test_vectorize_chunk_without_frames_raises_value_error() -> None:
    with pytest.raises(ValueError):
        asyncio.run(_qwen().vectorize_chunk(CHUNK, []))
