"""Pruebas unitarias de ``video_chunker``."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.processing.video_chunker import (
    compute_image_entropy,
    compute_ssim,
    encode_frame_jpeg,
    extract_interval_frames,
    process_video_single_pass,
    resize_frame,
)


def _make_video(
    path: Path,
    shots: list[int],
    fps: int = 10,
    size: tuple[int, int] = (160, 120),
) -> None:
    """Genera un vídeo sintético con un patrón distinto por plano."""
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter.fourcc(*"mp4v"), fps, size
    )
    rng = np.random.default_rng(0)
    for n_frames in shots:
        blocks = rng.integers(0, 2, (12, 16, 3), dtype=np.uint8) * 255
        base = cv2.resize(blocks, size, interpolation=cv2.INTER_NEAREST)
        for _ in range(n_frames):
            writer.write(base)
    writer.release()


def test_ssim_identical_is_one() -> None:
    img = np.random.default_rng(1).integers(0, 255, (60, 80), dtype=np.uint8)
    assert compute_ssim(img, img) == pytest.approx(1.0)


def test_ssim_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        compute_ssim(np.zeros((10, 10), np.uint8), np.zeros((12, 10), np.uint8))


def test_compute_image_entropy() -> None:
    constant_img = np.zeros((50, 50), dtype=np.uint8)
    assert compute_image_entropy(constant_img) == pytest.approx(0.0)

    binary_img = np.zeros((50, 50), dtype=np.uint8)
    binary_img[:25, :] = 255
    assert compute_image_entropy(binary_img) == pytest.approx(1.0)


def test_process_video_single_pass_selects_highest_entropy_frame(tmp_path: Path) -> None:
    video = tmp_path / "video_frames.mp4"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter.fourcc(*"mp4v"), 10, (160, 160)
    )

    f0 = np.zeros((160, 160, 3), dtype=np.uint8)
    rng = np.random.default_rng(42)
    f1 = rng.integers(0, 256, (160, 160, 3), dtype=np.uint8)
    f2 = np.full((160, 160, 3), 255, dtype=np.uint8)

    writer.write(f0)
    writer.write(f1)
    writer.write(f2)
    writer.release()

    chunks, frames_map = process_video_single_pass(video, min_shot_frames=10)
    assert len(chunks) == 1
    frames = frames_map[0]
    assert len(frames) == 3
    for f in frames:
        assert isinstance(f, np.ndarray)
        assert f.dtype == np.uint8
        assert f.shape == (160, 160, 3)

    assert np.allclose(frames[0], f0, atol=15)
    assert np.allclose(frames[2], f2, atol=15)
    assert compute_image_entropy(frames[1]) > compute_image_entropy(frames[0])


def test_process_video_single_pass_skips_boundary_transitions_and_samples_three_thirds(
    tmp_path: Path,
) -> None:
    video = tmp_path / "video_thirds.mp4"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter.fourcc(*"mp4v"), 10, (160, 160)
    )
    rng = np.random.default_rng(7)
    # 20 frames in one shot (min_shot_frames=25 keeps it as a single chunk).
    # Frames 0 and 19 are solid white boundary transitions (entropy ~ 0).
    # Interior frames are black except frames 4, 10, 15 in each third which have high entropy.
    target_frames = {
        4: rng.integers(0, 256, (160, 160, 3), dtype=np.uint8),
        10: rng.integers(0, 256, (160, 160, 3), dtype=np.uint8),
        15: rng.integers(0, 256, (160, 160, 3), dtype=np.uint8),
    }
    for idx in range(20):
        if idx in (0, 19):
            writer.write(np.full((160, 160, 3), 255, dtype=np.uint8))
        elif idx in target_frames:
            writer.write(target_frames[idx])
        else:
            writer.write(np.zeros((160, 160, 3), dtype=np.uint8))
    writer.release()

    chunks, frames_map = process_video_single_pass(video, min_shot_frames=25)
    assert len(chunks) == 1
    keyframes = frames_map[0]
    assert len(keyframes) == 3
    for kf in keyframes:
        assert compute_image_entropy(kf) > 5.0


def test_process_video_single_pass(tmp_path: Path) -> None:
    video = tmp_path / "v_single.mp4"
    _make_video(video, [20, 25, 30])
    chunks, frames_map = process_video_single_pass(video)

    assert len(chunks) == 3
    assert len(frames_map) == 3
    assert chunks[0].start_frame == 0
    assert chunks[1].start_frame == 20
    assert chunks[-1].end_frame == 74

    for c in chunks:
        assert c.index in frames_map
        frames = frames_map[c.index]
        assert len(frames) == 3
        for f in frames:
            assert f.shape == (120, 160, 3)
            assert f.dtype == np.uint8


def test_process_video_single_pass_splits_long_static_shots(tmp_path: Path) -> None:
    video = tmp_path / "v_static_long.mp4"
    _make_video(video, [60], fps=10)
    chunks, frames_map = process_video_single_pass(video, max_shot_duration=2.0)

    assert len(chunks) == 3
    assert [(c.start_frame, c.end_frame) for c in chunks] == [
        (0, 19),
        (20, 39),
        (40, 59),
    ]
    assert len(frames_map) == 3


def test_process_video_single_pass_resizes_to_target_width(tmp_path: Path) -> None:
    video = tmp_path / "v_hd.mp4"
    _make_video(video, [15, 15], size=(640, 360))
    chunks, frames_map = process_video_single_pass(video, target_width=320)

    assert len(chunks) == 2
    for c in chunks:
        for frame in frames_map[c.index]:
            assert frame.shape == (180, 320, 3)


def test_process_video_single_pass_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        process_video_single_pass("no_existe.mp4")

    video = tmp_path / "v_single_err.mp4"
    _make_video(video, [10])

    with pytest.raises(ValueError):
        process_video_single_pass(video, threshold=1.5)

    with pytest.raises(ValueError):
        process_video_single_pass(video, min_shot_frames=0)

    with pytest.raises(ValueError):
        process_video_single_pass(video, resize_width=4)

    with pytest.raises(ValueError):
        process_video_single_pass(video, entropy_stride=0)

    with pytest.raises(ValueError):
        process_video_single_pass(video, target_width=-1)

    with pytest.raises(ValueError):
        process_video_single_pass(video, max_shot_duration=0.0)


def test_resize_and_encode_frame_jpeg() -> None:
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    resized = resize_frame(frame, target_width=320)
    assert resized.shape == (180, 320, 3)

    unchanged = resize_frame(frame, target_width=800)
    assert unchanged.shape == (360, 640, 3)

    jpeg_bytes = encode_frame_jpeg(frame, jpeg_quality=80, target_width=320)
    decoded = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None and decoded.shape == (180, 320, 3)

    with pytest.raises(ValueError):
        encode_frame_jpeg(frame, jpeg_quality=0)


def test_extract_interval_frames(tmp_path: Path) -> None:
    video = tmp_path / "interval.mp4"
    _make_video(video, [30], fps=10, size=(640, 360))

    frames = extract_interval_frames(
        video, start_time=0.5, end_time=2.5, num_frames=2, target_width=320
    )
    assert len(frames) == 2
    for f in frames:
        assert f.shape == (180, 320, 3)

    assert extract_interval_frames(tmp_path / "missing.mp4", 0.0, 1.0) == []

