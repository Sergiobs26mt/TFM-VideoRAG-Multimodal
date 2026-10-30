"""Configuración centralizada, rutas y parámetros por defecto del sistema."""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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
            self.output_dimensionality < 1
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


def build_run_config(
    mode: str | None = None,
    cli_args: dict[str, Any] | None = None,
    *,
    raw_dir: Path = RAW_DIR,
    db_path: str = DB_PATH,
    vector_size: int = VECTOR_SIZE,
    max_chunks_per_video: int = MAX_CHUNKS_PER_VIDEO,
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
            "db_path": db_path,
            "raw_dir": str(raw_dir),
            "max_chunks_per_video": max_chunks_per_video,
            "chunking_defaults": asdict(ChunkingConfig()),
            "vectorizer_defaults": asdict(VectorizerConfig()),
            "search_defaults": asdict(SearchConfig()),
            "python": sys.version,
        }
    )
    return payload

