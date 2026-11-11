"""Configuración centralizada, rutas y parámetros por defecto del sistema."""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.models import IndexStats

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
RAW_DIR: Path = Path("data/raw")
PROCESSED_DIR: Path = Path("data/processed")
DB_PATH: str = "./data/vector_db"
AUDIT_DIR: Path = PROJECT_ROOT / "audit"

COLLECTION_NAME: str = "video_chunks"
VECTOR_SIZE: int = 3072
MAX_CHUNKS_PER_VIDEO: int = 800
EMBEDDING_MODEL: str = "gemini-embedding-2-preview"
RERANKER_MODEL: str = "gemini-3.5-flash-lite"
WHISPER_MODEL: str = "base"
AUDIO_LANGUAGE: str = "es"
DEFAULT_EMBEDDING_BACKEND: str = "gemini"


@dataclass(frozen=True)
class EmbeddingBackendSpec:
    """Motor de embeddings disponible: checkpoint, dimensiones MRL válidas y ubicación."""

    name: str
    model_id: str
    dimensions: tuple[int, ...]
    is_local: bool


EMBEDDING_BACKENDS: dict[str, EmbeddingBackendSpec] = {
    "gemini": EmbeddingBackendSpec("gemini", EMBEDDING_MODEL, (3072, 1536, 768), False),
    "qwen3-vl-2b": EmbeddingBackendSpec(
        "qwen3-vl-2b", "Qwen/Qwen3-VL-Embedding-2B", (2048, 1536, 768), True
    ),
    "gemma2": EmbeddingBackendSpec(
        "gemma2", "google/embeddinggemma-2", (768, 512, 256), True
    ),
}


def default_dimension(backend: str) -> int:
    """Devuelve la dimensión nativa (la mayor) del backend."""
    return EMBEDDING_BACKENDS[backend].dimensions[0]


def validate_embedding_choice(backend: str, dim: int) -> None:
    """Lanza ``ValueError`` si el backend no existe o no admite la dimensión."""
    spec = EMBEDDING_BACKENDS.get(backend)
    if spec is None:
        raise ValueError(
            f"Backend de embeddings desconocido: {backend!r}. "
            f"Opciones: {', '.join(EMBEDDING_BACKENDS)}."
        )
    if dim not in spec.dimensions:
        raise ValueError(
            f"Dimensión {dim} no válida para '{backend}'. "
            f"Opciones: {', '.join(map(str, spec.dimensions))}."
        )


def collection_name_for(backend: str, dim: int) -> str:
    """Nombre de la colección Qdrant de un backend y dimensión (Gemini 3072 conserva el original)."""
    validate_embedding_choice(backend, dim)
    if backend == DEFAULT_EMBEDDING_BACKEND and dim == VECTOR_SIZE:
        return COLLECTION_NAME
    return f"{COLLECTION_NAME}__{backend}__{dim}"


@dataclass(frozen=True)
class ChunkingConfig:
    """Parámetros de segmentación de planos por SSIM y muestreo de entropía."""

    threshold: float = 0.5
    min_shot_frames: int = 12
    max_shot_duration: float = 8.0
    resize_width: int = 160
    entropy_stride: int = 1

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold < 1.0:
            raise ValueError(f"threshold debe estar en el intervalo (0, 1): {self.threshold}")
        if self.min_shot_frames < 1:
            raise ValueError(f"min_shot_frames debe ser >= 1: {self.min_shot_frames}")
        if self.max_shot_duration <= 0.0:
            raise ValueError(f"max_shot_duration debe ser > 0: {self.max_shot_duration}")
        if self.resize_width < 8:
            raise ValueError(f"resize_width debe ser >= 8: {self.resize_width}")
        if self.entropy_stride < 1:
            raise ValueError(f"entropy_stride debe ser >= 1: {self.entropy_stride}")


@dataclass(frozen=True)
class VectorizerConfig:
    """Parámetros de codificación de imagen y cliente de embeddings."""

    model: str = EMBEDDING_MODEL
    output_dimensionality: int = VECTOR_SIZE
    max_retries: int = 4
    retry_base_delay: float = 4.0
    jpeg_quality: int = 80
    target_width: int = 512

    def __post_init__(self) -> None:
        if (
            not 128 <= self.output_dimensionality <= 3072
            or self.max_retries < 0
            or self.retry_base_delay < 0.0
            or not 1 <= self.jpeg_quality <= 100
            or self.target_width < 0
        ):
            raise ValueError("Parámetros de configuración no válidos.")


@dataclass(frozen=True)
class SearchConfig:
    """Parámetros de recuperación vectorial y reordenación multimodal."""

    top_k: int = 5
    rerank: bool = True
    reranker_model: str = RERANKER_MODEL

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k debe ser >= 1.")


@dataclass(frozen=True)
class AudioConfig:
    """Parámetros de transcripción ASR y alineación temporal de audio."""

    enable_audio: bool = True
    whisper_model: str = WHISPER_MODEL
    language: str | None = AUDIO_LANGUAGE

    def __post_init__(self) -> None:
        if not self.whisper_model.strip():
            raise ValueError("whisper_model no puede estar vacío.")


def build_run_config(
    mode: str | None = None,
    cli_args: dict[str, Any] | None = None,
    *,
    raw_dir: Path = RAW_DIR,
    db_path: str = DB_PATH,
    vector_size: int = VECTOR_SIZE,
    max_chunks_per_video: int = MAX_CHUNKS_PER_VIDEO,
    embedding_backend: str = DEFAULT_EMBEDDING_BACKEND,
    index_stats: IndexStats | None = None,
) -> dict[str, Any]:
    """Construye el diccionario de configuración de sesión para auditoría."""
    payload: dict[str, Any] = {}
    if mode is not None:
        payload["mode"] = mode
    if cli_args is not None:
        payload["cli_args"] = cli_args

    payload.update(
        {
            "vector_size": vector_size,
            "embedding_backend": embedding_backend,
            "embedding_dimension": vector_size,
            "device": index_stats.device_label if index_stats else None,
            "index_seconds": index_stats.seconds if index_stats else None,
            "db_path": db_path,
            "raw_dir": str(raw_dir),
            "max_chunks_per_video": max_chunks_per_video,
            "chunking_defaults": asdict(ChunkingConfig()),
            "vectorizer_defaults": asdict(VectorizerConfig()),
            "search_defaults": asdict(SearchConfig()),
            "audio_defaults": asdict(AudioConfig()),
            "python": sys.version,
        }
    )
    return payload

