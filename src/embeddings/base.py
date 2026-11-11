"""Interfaz común de los vectorizadores multimodales (Gemini y modelos locales)."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from src.models import MultimodalQuery, VectorizedChunk, VideoChunk

__all__ = ["MultimodalVectorizer"]


class MultimodalVectorizer(Protocol):
    """Motor de embeddings intercambiable para chunks de vídeo y consultas."""

    @property
    def backend(self) -> str:
        """Clave del backend en ``EMBEDDING_BACKENDS``."""
        ...

    @property
    def device_label(self) -> str:
        """Etiqueta legible del dispositivo de ejecución."""
        ...

    async def vectorize_chunk(
        self, chunk: VideoChunk, key_frames: list[NDArray[np.uint8]]
    ) -> VectorizedChunk:
        """Genera el embedding de un chunk a partir de sus fotogramas y su transcripción."""
        ...

    async def embed_query(self, query: str | MultimodalQuery) -> list[float]:
        """Genera el embedding unificado de una consulta multimodal."""
        ...

    async def embed_text(self, text: str) -> list[float]:
        """Genera el embedding de una consulta de solo texto."""
        ...
