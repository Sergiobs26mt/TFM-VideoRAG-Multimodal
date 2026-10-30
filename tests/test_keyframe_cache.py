"""Pruebas unitarias de la caché en disco de fotogramas clave (``data/processed``)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.processing.keyframe_cache import (
    clear_all_keyframes,
    delete_video_keyframes,
    load_chunk_keyframes_jpeg,
    save_chunk_keyframes_jpeg,
    save_video_keyframes,
)


def _dummy_frames(count: int = 3) -> list[np.ndarray]:
    rng = np.random.default_rng(7)
    return [rng.integers(0, 256, (60, 80, 3), dtype=np.uint8) for _ in range(count)]


def test_save_and_load_chunk_keyframes_jpeg(tmp_path: Path) -> None:
    frames_map = {0: _dummy_frames(3), 1: _dummy_frames(3)}
    video_dir = save_video_keyframes(
        "demo.mp4", frames_map, processed_dir=tmp_path, jpeg_quality=85
    )
    assert video_dir == tmp_path / "demo.mp4"

    loaded_all = load_chunk_keyframes_jpeg("demo.mp4", 0, processed_dir=tmp_path)
    assert len(loaded_all) == 3
    decoded = cv2.imdecode(np.frombuffer(loaded_all[0], dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None and decoded.shape == (60, 80, 3)

    loaded_two = load_chunk_keyframes_jpeg(
        "demo.mp4", 0, processed_dir=tmp_path, max_frames=2
    )
    assert len(loaded_two) == 2

    assert load_chunk_keyframes_jpeg("demo.mp4", 99, processed_dir=tmp_path) == []
    assert load_chunk_keyframes_jpeg("missing.mp4", 0, processed_dir=tmp_path) == []


def test_save_video_keyframes_overwrites_stale_chunks(tmp_path: Path) -> None:
    save_video_keyframes(
        "v.mp4", {0: _dummy_frames(3), 1: _dummy_frames(3)}, processed_dir=tmp_path
    )
    assert len(load_chunk_keyframes_jpeg("v.mp4", 1, processed_dir=tmp_path)) == 3

    save_video_keyframes("v.mp4", {0: _dummy_frames(3)}, processed_dir=tmp_path)
    assert len(load_chunk_keyframes_jpeg("v.mp4", 0, processed_dir=tmp_path)) == 3
    assert load_chunk_keyframes_jpeg("v.mp4", 1, processed_dir=tmp_path) == []


def test_delete_and_clear_keyframes(tmp_path: Path) -> None:
    save_video_keyframes("a.mp4", {0: _dummy_frames(3)}, processed_dir=tmp_path)
    save_video_keyframes("b.mp4", {0: _dummy_frames(3)}, processed_dir=tmp_path)

    delete_video_keyframes("a.mp4", processed_dir=tmp_path)
    assert load_chunk_keyframes_jpeg("a.mp4", 0, processed_dir=tmp_path) == []
    assert len(load_chunk_keyframes_jpeg("b.mp4", 0, processed_dir=tmp_path)) == 3

    clear_all_keyframes(processed_dir=tmp_path)
    assert load_chunk_keyframes_jpeg("b.mp4", 0, processed_dir=tmp_path) == []


def test_save_chunk_keyframes_jpeg_populates_single_chunk(tmp_path: Path) -> None:
    save_chunk_keyframes_jpeg(
        "legacy.mp4", 8, [b"jpeg-0", b"jpeg-1", b"jpeg-2"], processed_dir=tmp_path
    )
    assert load_chunk_keyframes_jpeg("legacy.mp4", 8, processed_dir=tmp_path) == [
        b"jpeg-0",
        b"jpeg-1",
        b"jpeg-2",
    ]
    assert load_chunk_keyframes_jpeg(
        "legacy.mp4", 8, processed_dir=tmp_path, max_frames=1
    ) == [b"jpeg-1"]


