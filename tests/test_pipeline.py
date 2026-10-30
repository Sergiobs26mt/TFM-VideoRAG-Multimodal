"""Pruebas unitarias del pipeline (vectorizador simulado, sin red)."""

import asyncio
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from src.config import MAX_CHUNKS_PER_VIDEO
from src.models import (
    QuotaSafetyError,
    VectorizationError,
    VectorizedChunk,
    VideoChunk,
)
from src.processing import pipeline


def _make_video(path: Path, shots: list[int]) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), 10, (160, 120))
    rng = np.random.default_rng(0)
    for n in shots:
        blocks = rng.integers(0, 2, (12, 16, 3), dtype=np.uint8) * 255
        img = cv2.resize(blocks, (160, 120), interpolation=cv2.INTER_NEAREST)
        for _ in range(n):
            writer.write(img)
    writer.release()


class _FakeVectorizer:
    """Simula el vectorizador midiendo la concurrencia máxima alcanzada."""

    active = 0
    peak = 0
    fail_index: int | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def vectorize_chunk(self, chunk: VideoChunk, key_frames: list[Any]) -> VectorizedChunk:
        assert len(key_frames) == 3
        cls = _FakeVectorizer
        cls.active += 1
        cls.peak = max(cls.peak, cls.active)
        await asyncio.sleep(0.01)
        cls.active -= 1
        if chunk.index == cls.fail_index:
            raise VectorizationError("fallo simulado")
        return VectorizedChunk(chunk.index, chunk.start_time, chunk.end_time, [float(chunk.index)])


@pytest.fixture(autouse=True)
def _patch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _FakeVectorizer.active = _FakeVectorizer.peak = 0
    _FakeVectorizer.fail_index = None
    monkeypatch.setattr(pipeline, "GeminiMultimodalVectorizer", _FakeVectorizer)
    monkeypatch.setattr(pipeline, "PROCESSED_DIR", tmp_path / "processed")


def test_pipeline_orders_and_limits_concurrency(tmp_path: Path) -> None:
    from src.processing.keyframe_cache import load_chunk_keyframes_jpeg

    video = tmp_path / "v.mp4"
    _make_video(video, [20, 20, 20, 20])
    result = asyncio.run(pipeline.process_video_pipeline(str(video), max_concurrent_requests=2))
    assert [r.index for r in result] == [0, 1, 2, 3]
    assert _FakeVectorizer.peak <= 2
    for idx in range(4):
        cached = load_chunk_keyframes_jpeg(
            "v.mp4", idx, processed_dir=tmp_path / "processed"
        )
        assert len(cached) == 3


def test_pipeline_skips_failed_chunk(tmp_path: Path) -> None:
    video = tmp_path / "v.mp4"
    _make_video(video, [20, 20, 20])
    _FakeVectorizer.fail_index = 1
    result = asyncio.run(pipeline.process_video_pipeline(str(video)))
    assert [r.index for r in result] == [0, 2]


def test_pipeline_invalid_concurrency() -> None:
    with pytest.raises(ValueError):
        asyncio.run(pipeline.process_video_pipeline("x.mp4", max_concurrent_requests=0))


def test_pipeline_missing_video() -> None:
    with pytest.raises(FileNotFoundError):
        asyncio.run(pipeline.process_video_pipeline("no_existe.mp4"))


def test_pipeline_aborts_over_quota_limit(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fake_chunks = [
        VideoChunk(i, i, i, float(i), float(i + 1))
        for i in range(MAX_CHUNKS_PER_VIDEO + 1)
    ]
    fake_frames = {c.index: [np.zeros((2, 2, 3), np.uint8)] * 3 for c in fake_chunks}
    monkeypatch.setattr(
        pipeline, "process_video_single_pass", lambda *_: (fake_chunks, fake_frames)
    )

    with caplog.at_level("ERROR"), pytest.raises(QuotaSafetyError, match="801 chunks"):
        asyncio.run(pipeline.process_video_pipeline("x.mp4"))
    assert "excede el límite de seguridad de 800" in caplog.text


def test_pipeline_allows_exactly_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_chunks = [
        VideoChunk(i, i, i, float(i), float(i + 1))
        for i in range(MAX_CHUNKS_PER_VIDEO)
    ]
    fake_frames = {c.index: [np.zeros((2, 2, 3), np.uint8)] * 3 for c in fake_chunks}
    monkeypatch.setattr(
        pipeline, "process_video_single_pass", lambda *_: (fake_chunks, fake_frames)
    )
    result = asyncio.run(pipeline.process_video_pipeline("x.mp4", max_concurrent_requests=50))
    assert len(result) == MAX_CHUNKS_PER_VIDEO
