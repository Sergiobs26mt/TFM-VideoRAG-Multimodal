"""Vectorización multimodal local con sentence-transformers (Qwen3-VL y EmbeddingGemma 2)."""

from __future__ import annotations

import asyncio
import importlib
import logging
import threading
from collections.abc import Mapping
from typing import Any, cast

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from src.config import EMBEDDING_BACKENDS, validate_embedding_choice
from src.embeddings.mrl import truncate_and_normalize_mrl
from src.models import MultimodalQuery, VectorizationError, VectorizedChunk, VideoChunk
from src.processing.video_chunker import resize_frame

logger = logging.getLogger(__name__)
__all__ = [
    "IMAGE_TAG",
    "QWEN_QUERY_INSTRUCTION",
    "LocalMultimodalVectorizer",
    "load_sentence_transformer",
    "select_device",
    "torch_dtype_for",
]

QWEN_QUERY_INSTRUCTION = "Retrieve video segments relevant to the query."
IMAGE_TAG = "<|image|>"
_LOCAL_BACKENDS = ("qwen3-vl-2b", "gemma2")


def _import_torch() -> Any:
    return importlib.import_module("torch")


def select_device() -> str:
    """Devuelve ``mps`` si Metal está disponible y ``cpu`` en caso contrario."""
    torch = _import_torch()
    return "mps" if torch.backends.mps.is_available() else "cpu"


def torch_dtype_for(device: str, torch_module: Any) -> Any:
    """``bfloat16`` en MPS y ``float32`` en CPU (Gemma 2 produce NaN en ``float16``)."""
    return torch_module.bfloat16 if device == "mps" else torch_module.float32


def load_sentence_transformer(model_id: str, device: str) -> Any:
    """Carga un ``SentenceTransformer`` con el dtype que corresponde al dispositivo."""
    try:
        sentence_transformers = importlib.import_module("sentence_transformers")
    except ImportError as exc:
        raise VectorizationError(
            "Faltan dependencias de los embeddings locales. "
            'Ejecuta: pip install -U "sentence-transformers[image]" transformers'
        ) from exc
    torch = _import_torch()
    try:
        return sentence_transformers.SentenceTransformer(
            model_id,
            device=device,
            model_kwargs={"torch_dtype": torch_dtype_for(device, torch)},
        )
    except Exception as exc:  # pesos ausentes, sin red, disco lleno, memoria
        raise VectorizationError(
            f"No se pudo cargar el modelo local '{model_id}': {exc}"
        ) from exc


def _qwen_document(text: str, images: list[Image.Image]) -> str | dict[str, Any]:
    payload: dict[str, Any] = {"image": images}
    if text:
        payload["text"] = text
    return payload


def _qwen_query(
    text: str, images: list[Image.Image]
) -> tuple[str | dict[str, Any], dict[str, Any]]:
    kwargs = {"prompt": QWEN_QUERY_INSTRUCTION}
    if not images:
        return text, kwargs
    payload: dict[str, Any] = {"image": images}
    if text:
        payload["text"] = text
    return payload, kwargs


def _gemma_document(
    text: str, images: list[Image.Image], prompts: Mapping[str, str]
) -> str | dict[str, Any]:
    tagged = f"{prompts['Document']}{IMAGE_TAG * len(images)}"
    return {"text": f"{tagged} {text}" if text else tagged, "image": images}


def _gemma_query(
    text: str, images: list[Image.Image], prompts: Mapping[str, str]
) -> tuple[str | dict[str, Any], dict[str, Any]]:
    if not images:
        return text, {"prompt_name": "SearchQuery"}
    tags = IMAGE_TAG * len(images)
    if not text:
        return {"text": tags, "image": images}, {}
    return {"text": f"{prompts['SearchQuery']}{tags} {text}", "image": images}, {}


class LocalMultimodalVectorizer:
    """Genera un embedding por chunk o consulta con un modelo local ya cargado."""

    def __init__(
        self,
        backend: str,
        model: Any,
        device: str,
        output_dimensionality: int,
        target_width: int = 512,
    ) -> None:
        if backend not in _LOCAL_BACKENDS:
            raise ValueError(f"Backend local desconocido: {backend!r}.")
        validate_embedding_choice(backend, output_dimensionality)
        self._backend = backend
        self._model = model
        self._device = device
        self._dimensionality = output_dimensionality
        self._target_width = target_width
        self._model_id = EMBEDDING_BACKENDS[backend].model_id
        # Un solo hilo usa el modelo a la vez, también entre bucles de eventos distintos
        self._lock = threading.Lock()

    @property
    def backend(self) -> str:
        """Clave del backend en ``EMBEDDING_BACKENDS``."""
        return self._backend

    @property
    def device_label(self) -> str:
        """Etiqueta legible del dispositivo de ejecución."""
        return f"Local {self._device.upper()}"

    def _prompts(self) -> Mapping[str, str]:
        return cast("Mapping[str, str]", getattr(self._model, "prompts", {}) or {})

    def _to_pil(self, frame: NDArray[np.uint8]) -> Image.Image:
        resized = resize_frame(frame, self._target_width)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return Image.fromarray(cast("NDArray[np.uint8]", rgb))

    def build_chunk_input(
        self, transcript: str | None, key_frames: list[NDArray[np.uint8]]
    ) -> str | dict[str, Any]:
        """Empaqueta fotogramas y transcripción de un chunk en el formato del modelo."""
        if not key_frames:
            raise ValueError("Se necesita al menos un fotograma clave.")
        images = [self._to_pil(frame) for frame in key_frames]
        text = transcript.strip() if transcript else ""
        if self._backend == "gemma2":
            return _gemma_document(text, images, self._prompts())
        return _qwen_document(text, images)

    def build_query_input(
        self, query: str | MultimodalQuery
    ) -> tuple[str | dict[str, Any], dict[str, Any]]:
        """Empaqueta una consulta y devuelve la entrada y los kwargs de ``encode``."""
        mquery = MultimodalQuery(text=query) if isinstance(query, str) else query
        images: list[Image.Image] = []
        for img_bytes in mquery.images:
            decoded = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if decoded is None:
                raise ValueError("Imagen de consulta no válida o corrupta.")
            images.append(self._to_pil(cast("NDArray[np.uint8]", decoded)))
        text = mquery.composed_text
        if not images and not text:
            raise ValueError(
                "La consulta no contiene texto ni imágenes válidas para vectorizar."
            )
        if self._backend == "gemma2":
            return _gemma_query(text, images, self._prompts())
        return _qwen_query(text, images)

    async def vectorize_chunk(
        self, chunk: VideoChunk, key_frames: list[NDArray[np.uint8]]
    ) -> VectorizedChunk:
        """Genera de forma asíncrona el embedding de un chunk."""
        inputs = self.build_chunk_input(chunk.transcript, key_frames)
        values = await self._embed(inputs, {}, f"chunk {chunk.index}")
        return VectorizedChunk(
            index=chunk.index,
            start_time=chunk.start_time,
            end_time=chunk.end_time,
            embedding=values,
            transcript=chunk.transcript,
        )

    async def embed_query(self, query: str | MultimodalQuery) -> list[float]:
        """Genera de forma asíncrona el embedding unificado de una consulta multimodal."""
        inputs, kwargs = self.build_query_input(query)
        return await self._embed(inputs, kwargs, "consulta multimodal")

    async def embed_text(self, text: str) -> list[float]:
        """Genera de forma asíncrona el embedding de una consulta de solo texto."""
        return await self.embed_query(text)

    def _encode(self, inputs: str | dict[str, Any], kwargs: dict[str, Any]) -> Any:
        with self._lock:
            return self._model.encode(inputs, convert_to_numpy=True, **kwargs)

    async def _embed(
        self, inputs: str | dict[str, Any], kwargs: dict[str, Any], label: str
    ) -> list[float]:
        """Codifica en un hilo, valida que el vector sea finito y lo recorta a la dimensión."""
        try:
            raw = await asyncio.to_thread(self._encode, inputs, kwargs)
        except Exception as exc:  # cualquier fallo del modelo se reporta como del vectorizador
            raise VectorizationError(
                f"Fallo del modelo local '{self._model_id}' en {label}: {exc}"
            ) from exc
        vector = np.asarray(raw, dtype=np.float32).reshape(-1)
        if not bool(np.isfinite(vector).all()):
            raise VectorizationError(f"Embedding no finito para {label}.")
        return truncate_and_normalize_mrl(vector.tolist(), self._dimensionality)
