"""Factoría de vectorizadores y gestión del único modelo local en memoria."""

from __future__ import annotations

import gc
import importlib
import logging
from typing import Any

from src.config import EMBEDDING_BACKENDS, validate_embedding_choice
from src.embeddings.base import MultimodalVectorizer
from src.embeddings.gemini_vectorizer import GeminiMultimodalVectorizer
from src.embeddings.local_vectorizer import (
    LocalMultimodalVectorizer,
    load_sentence_transformer,
    select_device,
)

logger = logging.getLogger(__name__)
__all__ = ["create_vectorizer", "device_label_for", "unload_active_vectorizer"]

# (model_id, modelo): solo un modelo local vive en memoria a la vez
_active: tuple[str, Any] | None = None


def _release_memory() -> None:
    """Fuerza la recogida de basura y vacía la caché de la GPU integrada."""
    gc.collect()
    torch = importlib.import_module("torch")
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def unload_active_vectorizer() -> None:
    """Descarta el modelo local cargado y libera su memoria."""
    global _active
    if _active is None:
        return
    logger.info("Descargando modelo local '%s'", _active[0])
    _active = None
    _release_memory()


def device_label_for(backend: str) -> str:
    """Etiqueta del dispositivo que usaría el backend (sin cargar ningún modelo)."""
    if not EMBEDDING_BACKENDS[backend].is_local:
        return "Cloud API"
    return f"Local {select_device().upper()}"


def create_vectorizer(
    backend: str,
    output_dimensionality: int,
    jpeg_quality: int = 80,
    target_width: int = 512,
) -> MultimodalVectorizer:
    """Crea el vectorizador del backend; un cambio de modelo local descarga el anterior."""
    global _active
    validate_embedding_choice(backend, output_dimensionality)
    spec = EMBEDDING_BACKENDS[backend]
    if not spec.is_local:
        unload_active_vectorizer()
        return GeminiMultimodalVectorizer(
            output_dimensionality=output_dimensionality,
            jpeg_quality=jpeg_quality,
            target_width=target_width,
        )

    device = select_device()
    if _active is None or _active[0] != spec.model_id:
        unload_active_vectorizer()
        _active = (spec.model_id, load_sentence_transformer(spec.model_id, device))
    return LocalMultimodalVectorizer(
        backend, _active[1], device, output_dimensionality, target_width
    )
