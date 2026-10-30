"""Caché persistente en disco (``data/processed``) de fotogramas clave por chunk."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.config import PROCESSED_DIR
from src.processing.video_chunker import encode_frame_jpeg


def save_video_keyframes(
    video_name: str,
    frames_by_index: dict[int, list[NDArray[np.uint8]]],
    processed_dir: Path | str = PROCESSED_DIR,
    jpeg_quality: int = 80,
    target_width: int = 512,
) -> Path:
    """Guarda en disco los fotogramas clave JPEG de cada chunk reemplazando cachés previas."""
    video_dir = Path(processed_dir) / video_name
    if video_dir.exists():
        shutil.rmtree(video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

    for chunk_idx, frames in frames_by_index.items():
        jpeg_frames = [
            encode_frame_jpeg(
                frame, jpeg_quality=jpeg_quality, target_width=target_width
            )
            for frame in frames
        ]
        save_chunk_keyframes_jpeg(
            video_name, chunk_idx, jpeg_frames, processed_dir=processed_dir
        )

    return video_dir


def save_chunk_keyframes_jpeg(
    video_name: str,
    chunk_index: int,
    jpeg_frames: list[bytes],
    processed_dir: Path | str = PROCESSED_DIR,
) -> None:
    """Guarda en disco los fotogramas JPEG de un único chunk (escritura en caché bajo demanda)."""
    if not jpeg_frames or chunk_index < 0:
        return
    video_dir = Path(processed_dir) / video_name
    video_dir.mkdir(parents=True, exist_ok=True)
    for frame_pos, jpeg_bytes in enumerate(jpeg_frames):
        frame_file = video_dir / f"chunk_{chunk_index:04d}_frame_{frame_pos}.jpg"
        frame_file.write_bytes(jpeg_bytes)


def load_chunk_keyframes_jpeg(
    video_name: str,
    chunk_index: int,
    processed_dir: Path | str = PROCESSED_DIR,
    max_frames: int | None = None,
) -> list[bytes]:
    """Lee desde disco los fotogramas JPEG cacheados de un chunk."""
    video_dir = Path(processed_dir) / video_name
    if not video_dir.is_dir():
        return []

    files = sorted(video_dir.glob(f"chunk_{chunk_index:04d}_frame_*.jpg"))
    if max_frames is not None and max_frames > 0:
        if max_frames == 1 and len(files) >= 3:
            files = [files[len(files) // 2]]
        else:
            files = files[:max_frames]
    return [f.read_bytes() for f in files]


def delete_video_keyframes(
    video_name: str,
    processed_dir: Path | str = PROCESSED_DIR,
) -> None:
    """Elimina la carpeta de fotogramas cacheados de un vídeo."""
    video_dir = Path(processed_dir) / video_name
    if video_dir.exists():
        shutil.rmtree(video_dir)


def clear_all_keyframes(processed_dir: Path | str = PROCESSED_DIR) -> None:
    """Elimina todas las subcarpetas de fotogramas cacheados en ``processed_dir``."""
    base = Path(processed_dir)
    if not base.is_dir():
        return
    for entry in base.iterdir():
        if entry.is_dir():
            shutil.rmtree(entry)

