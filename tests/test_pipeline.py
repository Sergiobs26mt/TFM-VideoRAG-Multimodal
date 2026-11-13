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
    TranscriptSegment,
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
        return VectorizedChunk(
            chunk.index,
            chunk.start_time,
            chunk.end_time,
            [float(chunk.index)],
            transcript=chunk.transcript,
        )


@pytest.fixture(autouse=True)
def _patch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _FakeVectorizer.active = _FakeVectorizer.peak = 0
    _FakeVectorizer.fail_index = None
    monkeypatch.setattr(pipeline, "create_vectorizer", lambda *a, **k: _FakeVectorizer())
    monkeypatch.setattr(pipeline, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(
        pipeline, "transcribe_video_audio", lambda *_, **__: []
    )


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


def test_pipeline_enriches_chunks_with_aligned_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_chunks = [
        VideoChunk(0, 0, 9, 0.0, 2.0),
        VideoChunk(1, 10, 19, 2.0, 4.0),
    ]
    fake_frames = {0: [np.zeros((2, 2, 3), np.uint8)] * 3, 1: [np.zeros((2, 2, 3), np.uint8)] * 3}
    monkeypatch.setattr(
        pipeline, "process_video_single_pass", lambda *_: (fake_chunks, fake_frames)
    )
    monkeypatch.setattr(
        pipeline,
        "transcribe_video_audio",
        lambda path, model: [
            TranscriptSegment(start_time=0.2, end_time=1.5, text="Explicación inicial")
        ],
    )

    result = asyncio.run(
        pipeline.process_video_pipeline("x.mp4", enable_audio=True, whisper_model="tiny")
    )
    assert len(result) == 2
    assert result[0].transcript == "Explicación inicial"
    assert result[1].transcript is None


def test_pipeline_degrades_gracefully_when_audio_fails(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fake_chunks = [VideoChunk(0, 0, 9, 0.0, 2.0)]
    fake_frames = {0: [np.zeros((2, 2, 3), np.uint8)] * 3}
    monkeypatch.setattr(
        pipeline, "process_video_single_pass", lambda *_: (fake_chunks, fake_frames)
    )

    def broken_audio(*_: Any, **__: Any) -> list[TranscriptSegment]:
        raise RuntimeError("ffmpeg no disponible")

    monkeypatch.setattr(pipeline, "transcribe_video_audio", broken_audio)

    with caplog.at_level("WARNING"):
        result = asyncio.run(pipeline.process_video_pipeline("x.mp4", enable_audio=True))

    assert len(result) == 1
    assert result[0].transcript is None
    assert "continuando solo con modalidad visual" in caplog.text


def test_pipeline_skips_audio_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_chunks = [VideoChunk(0, 0, 9, 0.0, 2.0)]
    fake_frames = {0: [np.zeros((2, 2, 3), np.uint8)] * 3}
    monkeypatch.setattr(
        pipeline, "process_video_single_pass", lambda *_: (fake_chunks, fake_frames)
    )
    called = {"audio": False}

    def track_audio(*_: Any, **__: Any) -> list[TranscriptSegment]:
        called["audio"] = True
        return []

    monkeypatch.setattr(pipeline, "transcribe_video_audio", track_audio)

    result = asyncio.run(pipeline.process_video_pipeline("x.mp4", enable_audio=False))
    assert len(result) == 1
    assert called["audio"] is False


def _patch_two_fake_chunks(monkeypatch: pytest.MonkeyPatch, count: int = 2) -> None:
    fake_chunks = [VideoChunk(i, i, i, float(i), float(i + 1)) for i in range(count)]
    fake_frames = {c.index: [np.zeros((2, 2, 3), np.uint8)] * 3 for c in fake_chunks}
    monkeypatch.setattr(
        pipeline, "process_video_single_pass", lambda *_: (fake_chunks, fake_frames)
    )


def test_pipeline_forces_single_concurrency_for_local_backend(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    video = tmp_path / "v.mp4"
    _make_video(video, [20, 20, 20, 20])
    with caplog.at_level("WARNING"):
        result = asyncio.run(
            pipeline.process_video_pipeline(
                str(video),
                max_concurrent_requests=3,
                embedding_backend="qwen3-vl-2b",
                output_dimensionality=768,
            )
        )
    assert len(result) == 4
    assert _FakeVectorizer.peak == 1
    assert "concurrencia 1" in caplog.text


def test_pipeline_unloads_whisper_for_local_backend_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_two_fake_chunks(monkeypatch)
    events: list[str] = []

    def fake_create(*_: Any, **__: Any) -> _FakeVectorizer:
        events.append("create")
        return _FakeVectorizer()

    monkeypatch.setattr(pipeline, "unload_whisper_models", lambda: events.append("unload"))
    monkeypatch.setattr(pipeline, "create_vectorizer", fake_create)

    asyncio.run(
        pipeline.process_video_pipeline(
            "x.mp4", embedding_backend="gemma2", output_dimensionality=768
        )
    )
    assert events == ["unload", "create"]

    events.clear()
    asyncio.run(pipeline.process_video_pipeline("x.mp4", embedding_backend="gemini"))
    assert events == ["create"]


def test_pipeline_skips_chunk_limit_for_local_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_two_fake_chunks(monkeypatch, count=3)
    monkeypatch.setattr(pipeline, "MAX_CHUNKS_PER_VIDEO", 2)

    with pytest.raises(QuotaSafetyError):
        asyncio.run(pipeline.process_video_pipeline("x.mp4", embedding_backend="gemini"))
    result = asyncio.run(
        pipeline.process_video_pipeline(
            "x.mp4", embedding_backend="gemma2", output_dimensionality=768
        )
    )
    assert len(result) == 3


def test_pipeline_reports_local_model_loading_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_two_fake_chunks(monkeypatch)
    local: list[str] = []
    cloud: list[str] = []

    asyncio.run(
        pipeline.process_video_pipeline(
            "x.mp4",
            embedding_backend="gemma2",
            output_dimensionality=768,
            progress_callback=lambda msg, _: local.append(msg),
        )
    )
    asyncio.run(
        pipeline.process_video_pipeline(
            "x.mp4",
            embedding_backend="gemini",
            progress_callback=lambda msg, _: cloud.append(msg),
        )
    )
    assert any("Cargando modelo local" in m for m in local)
    assert any("gemma2" in m for m in local)
    assert not any("Cargando modelo local" in m for m in cloud)
    assert not any("Gemini" in m for m in local)


def test_pipeline_passes_backend_and_dimension_to_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_two_fake_chunks(monkeypatch)
    calls: list[tuple[Any, ...]] = []

    def fake_create(*args: Any) -> _FakeVectorizer:
        calls.append(args)
        return _FakeVectorizer()

    monkeypatch.setattr(pipeline, "create_vectorizer", fake_create)
    asyncio.run(
        pipeline.process_video_pipeline(
            "x.mp4",
            embedding_backend="gemma2",
            output_dimensionality=768,
            jpeg_quality=70,
            target_width=640,
        )
    )
    assert calls == [("gemma2", 768, 70, 640)]


def test_pipeline_rejects_invalid_backend_dimension_before_processing() -> None:
    with pytest.raises(ValueError):
        asyncio.run(
            pipeline.process_video_pipeline(
                "no_existe.mp4", embedding_backend="gemma2", output_dimensionality=3072
            )
        )


def test_pipeline_resolves_default_dimension_from_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_two_fake_chunks(monkeypatch)
    calls: list[tuple[Any, ...]] = []

    def fake_create(*args: Any) -> _FakeVectorizer:
        calls.append(args)
        return _FakeVectorizer()

    monkeypatch.setattr(pipeline, "create_vectorizer", fake_create)
    asyncio.run(pipeline.process_video_pipeline("x.mp4", embedding_backend="gemma2"))
    assert calls[0][0:2] == ("gemma2", 768)
