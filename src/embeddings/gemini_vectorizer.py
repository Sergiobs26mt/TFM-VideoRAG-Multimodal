"""Vectorización multimodal de fragmentos de vídeo con Gemini Embedding."""

from __future__ import annotations

import logging

import numpy as np
from google.genai import types
from numpy.typing import NDArray

from src.config import EMBEDDING_MODEL, VECTOR_SIZE, VectorizerConfig
from src.models import VectorizationError, VectorizedChunk, VideoChunk
from src.processing.video_chunker import encode_frame_jpeg
from src.utils.gemini_client import call_with_retry, create_gemini_client

logger = logging.getLogger(__name__)


class GeminiMultimodalVectorizer:
    """Genera un embedding único por chunk a partir de sus fotogramas clave."""

    def __init__(
        self,
        model: str = EMBEDDING_MODEL,
        output_dimensionality: int = VECTOR_SIZE,
        max_retries: int = 4,
        retry_base_delay: float = 4.0,
        jpeg_quality: int = 80,
        target_width: int = 512,
    ) -> None:
        cfg = VectorizerConfig(
            model=model,
            output_dimensionality=output_dimensionality,
            max_retries=max_retries,
            retry_base_delay=retry_base_delay,
            jpeg_quality=jpeg_quality,
            target_width=target_width,
        )
        self._client = create_gemini_client()
        self._model = cfg.model
        self._dimensionality = cfg.output_dimensionality
        self._max_retries = cfg.max_retries
        self._retry_base_delay = cfg.retry_base_delay
        self._jpeg_quality = cfg.jpeg_quality
        self._target_width = cfg.target_width

    def build_payload(
        self, key_frames: list[NDArray[np.uint8]]
    ) -> types.Content:
        """Codifica fotogramas clave a JPEG en un único ``Content``."""
        if not key_frames:
            raise ValueError("Se necesita al menos un fotograma clave.")

        parts = [
            types.Part.from_bytes(
                data=encode_frame_jpeg(
                    frame,
                    jpeg_quality=self._jpeg_quality,
                    target_width=self._target_width,
                ),
                mime_type="image/jpeg",
            )
            for frame in key_frames
        ]
        return types.Content(parts=parts)

    async def vectorize_chunk(
        self, chunk: VideoChunk, key_frames: list[NDArray[np.uint8]]
    ) -> VectorizedChunk:
        """Genera de forma asíncrona el embedding de un chunk."""
        payload = self.build_payload(key_frames)
        values = await self._embed(payload, f"chunk {chunk.index}")
        return VectorizedChunk(
            index=chunk.index,
            start_time=chunk.start_time,
            end_time=chunk.end_time,
            embedding=values,
        )

    async def embed_text(self, text: str) -> list[float]:
        """Genera de forma asíncrona el embedding de una consulta de solo texto."""
        if not text.strip():
            raise ValueError("La consulta de texto no puede estar vacía.")
        return await self._embed(text, "consulta de texto")

    async def _embed(self, contents: types.Content | str, label: str) -> list[float]:
        """Llama a la API de embeddings con reintentos y backoff exponencial."""
        config = types.EmbedContentConfig(output_dimensionality=self._dimensionality)
        response = await call_with_retry(
            lambda: self._client.aio.models.embed_content(
                model=self._model, contents=contents, config=config
            ),
            label=label,
            max_retries=self._max_retries,
            retry_base_delay=self._retry_base_delay,
        )
        return self._extract_values(response, label)

    def _extract_values(
        self, response: types.EmbedContentResponse, label: str
    ) -> list[float]:
        """Extrae y valida el vector de la respuesta de la API (normaliza L2 si dim < 3072)."""
        if not response.embeddings or not response.embeddings[0].values:
            raise VectorizationError(f"Respuesta sin embedding para {label}.")

        values = [float(v) for v in response.embeddings[0].values]
        if self._dimensionality < 3072:
            norm = float(np.linalg.norm(values))
            if norm > 0:
                values = [v / norm for v in values]
        return values
