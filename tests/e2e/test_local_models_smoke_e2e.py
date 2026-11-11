"""Prueba de humo con los pesos reales de Qwen3-VL-Embedding-2B y EmbeddingGemma 2 en MPS."""

from __future__ import annotations

import gc
import importlib
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image

pytestmark = pytest.mark.e2e_local

GEMMA_ID = "google/embeddinggemma-2"
QWEN_ID = "Qwen/Qwen3-VL-Embedding-2B"
IMAGE_TAG = "<|image|>"
GEMMA_DOCUMENT_PREFIX = "title: none | text: "
GEMMA_QUERY_PREFIX = "task: search result | query: "
QWEN_QUERY_INSTRUCTION = "Retrieve video segments relevant to the query."


def _torch() -> Any:
    return importlib.import_module("torch")


def _load(model_id: str) -> Iterator[Any]:
    torch = _torch()
    if not torch.backends.mps.is_available():
        pytest.skip("MPS no disponible.")
    sentence_transformers = importlib.import_module("sentence_transformers")
    model = sentence_transformers.SentenceTransformer(
        model_id, device="mps", model_kwargs={"torch_dtype": torch.bfloat16}
    )
    yield model
    del model
    gc.collect()
    torch.mps.empty_cache()


def _gradient(channel: int) -> Image.Image:
    """Degradado de 512x288 en un solo canal RGB para obtener imágenes distinguibles."""
    ramp = np.tile(np.linspace(0, 255, 512, dtype=np.uint8), (288, 1))
    pixels = np.zeros((288, 512, 3), dtype=np.uint8)
    pixels[:, :, channel] = ramp
    return Image.fromarray(pixels)


def _cosine(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _assert_finite(vector: NDArray[np.float32], size: int) -> None:
    assert vector.shape == (size,)
    assert bool(np.isfinite(vector).all())


@pytest.fixture(scope="module")
def frames() -> list[Image.Image]:
    return [_gradient(0), _gradient(1), _gradient(2)]


@pytest.fixture(scope="module")
def gemma() -> Iterator[Any]:
    yield from _load(GEMMA_ID)


@pytest.fixture(scope="module")
def qwen() -> Iterator[Any]:
    yield from _load(QWEN_ID)


def test_gemma_three_tags_three_images_finite_unit_norm(
    gemma: Any, frames: list[Image.Image]
) -> None:
    text = f"{GEMMA_DOCUMENT_PREFIX}{IMAGE_TAG * 3} hola"
    vector = gemma.encode({"text": text, "image": frames}, convert_to_numpy=True)
    _assert_finite(vector, 768)
    assert 0.98 <= float(np.linalg.norm(vector)) <= 1.02


def test_gemma_image_order_changes_embedding(gemma: Any, frames: list[Image.Image]) -> None:
    text = f"{GEMMA_DOCUMENT_PREFIX}{IMAGE_TAG * 3}"
    forward = gemma.encode({"text": text, "image": frames}, convert_to_numpy=True)
    backward = gemma.encode({"text": text, "image": frames[::-1]}, convert_to_numpy=True)
    assert _cosine(forward, backward) < 0.9999


def test_gemma_image_only_query_is_closest_to_its_own_frame(
    gemma: Any, frames: list[Image.Image]
) -> None:
    query = gemma.encode({"text": IMAGE_TAG, "image": [frames[0]]}, convert_to_numpy=True)
    documents = [
        gemma.encode(
            {"text": f"{GEMMA_DOCUMENT_PREFIX}{IMAGE_TAG}", "image": [frame]},
            convert_to_numpy=True,
        )
        for frame in frames
    ]
    scores = [_cosine(query, doc) for doc in documents]
    assert int(np.argmax(scores)) == 0


def test_gemma_image_query_with_prefix_and_text_is_finite(
    gemma: Any, frames: list[Image.Image]
) -> None:
    text = f"{GEMMA_QUERY_PREFIX}{IMAGE_TAG} color rojo"
    vector = gemma.encode({"text": text, "image": [frames[0]]}, convert_to_numpy=True)
    _assert_finite(vector, 768)


def test_qwen_three_images_and_text_finite(qwen: Any, frames: list[Image.Image]) -> None:
    vector = qwen.encode({"text": "hola", "image": frames}, convert_to_numpy=True)
    _assert_finite(vector, 2048)


def test_qwen_image_only_and_instructed_text_query_finite(
    qwen: Any, frames: list[Image.Image]
) -> None:
    image_only = qwen.encode({"image": [frames[0]]}, convert_to_numpy=True)
    text_query = qwen.encode(
        "variable", prompt=QWEN_QUERY_INSTRUCTION, convert_to_numpy=True
    )
    _assert_finite(image_only, 2048)
    _assert_finite(text_query, 2048)
