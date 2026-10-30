"""Fragmentación de vídeo, extracción de fotogramas y codificación de imagen."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from src.config import ChunkingConfig
from src.models import VideoChunk

logger = logging.getLogger(__name__)

# Constantes de estabilidad numérica del SSIM (rango dinámico L = 255)
_C1: float = (0.01 * 255) ** 2
_C2: float = (0.03 * 255) ** 2


def compute_ssim(
    frame_a: NDArray[np.uint8], frame_b: NDArray[np.uint8], window_size: int = 7
) -> float:
    """Calcula el SSIM medio entre dos imágenes en escala de grises."""
    if frame_a.shape != frame_b.shape:
        raise ValueError("Los fotogramas deben tener el mismo tamaño.")

    a = frame_a.astype(np.float64)
    b = frame_b.astype(np.float64)
    ksize = (window_size, window_size)

    mu_a = cv2.GaussianBlur(a, ksize, 1.5)
    mu_b = cv2.GaussianBlur(b, ksize, 1.5)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b

    sigma_a2 = cv2.GaussianBlur(a * a, ksize, 1.5) - mu_a2
    sigma_b2 = cv2.GaussianBlur(b * b, ksize, 1.5) - mu_b2
    sigma_ab = cv2.GaussianBlur(a * b, ksize, 1.5) - mu_ab

    ssim_map = ((2 * mu_ab + _C1) * (2 * sigma_ab + _C2)) / (
        (mu_a2 + mu_b2 + _C1) * (sigma_a2 + sigma_b2 + _C2)
    )
    return float(ssim_map.mean())


def compute_image_entropy(frame: NDArray[np.uint8]) -> float:
    """Calcula la entropía visual de Shannon en bits para un fotograma."""
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    total_pixels = float(gray.shape[0] * gray.shape[1])
    if total_pixels == 0:
        return 0.0

    prob = hist.flatten() / total_pixels
    prob_non_zero = prob[prob > 0]
    return -float(np.sum(prob_non_zero * np.log2(prob_non_zero)))


def resize_frame(frame: NDArray[np.uint8], target_width: int) -> NDArray[np.uint8]:
    """Redimensiona el fotograma si supera ``target_width`` conservando relación de aspecto."""
    height, width = frame.shape[:2]
    if target_width > 0 and width > target_width:
        new_h = max(8, int(height * target_width / width))
        return cast(
            "NDArray[np.uint8]",
            cv2.resize(frame, (target_width, new_h), interpolation=cv2.INTER_AREA),
        )
    return frame


def encode_frame_jpeg(
    frame: NDArray[np.uint8],
    jpeg_quality: int = 80,
    target_width: int = 0,
) -> bytes:
    """Redimensiona opcionalmente y codifica un fotograma BGR a bytes JPEG."""
    if not 1 <= jpeg_quality <= 100:
        raise ValueError(f"jpeg_quality debe estar entre 1 y 100: {jpeg_quality}")
    target = resize_frame(frame, target_width)
    ok, buffer = cv2.imencode(".jpg", target, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        raise ValueError("No se pudo codificar el fotograma a JPEG.")
    return buffer.tobytes()


def extract_interval_frames(
    video_path: str | Path,
    start_time: float,
    end_time: float,
    num_frames: int = 2,
    target_width: int = 512,
) -> list[NDArray[np.uint8]]:
    """Extrae fotogramas equiespaciados y redimensionados de un intervalo temporal."""
    path = Path(video_path)
    if not path.is_file():
        logger.warning("Vídeo no encontrado para extracción de fotogramas: %s", path)
        return []

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        logger.warning("No se pudo abrir el vídeo con OpenCV: %s", path)
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames: list[NDArray[np.uint8]] = []

    try:
        duration = max(0.1, end_time - start_time)
        timestamps = [
            start_time + duration * ((i + 0.5) / num_frames) for i in range(num_frames)
        ]

        for ts in timestamps:
            frame_idx = max(0, int(ts * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, raw_frame = cap.read()
            if ok and raw_frame is not None:
                frame = cast("NDArray[np.uint8]", raw_frame)
                frames.append(resize_frame(frame, target_width))
    finally:
        cap.release()

    return frames


def _select_chunk_keyframes(
    chunk_frames: list[tuple[NDArray[np.uint8], float]],
    target_width: int,
) -> list[NDArray[np.uint8]]:
    """Selecciona 3 fotogramas de los tercios interiores evitando bordes de transición."""
    n = len(chunk_frames)
    if n == 1:
        only = resize_frame(chunk_frames[0][0], target_width)
        return [only, only, only]
    if n == 2:
        (f0, e0), (f1, e1) = chunk_frames[0], chunk_frames[1]
        mid = f0 if e0 >= e1 else f1
        return [
            resize_frame(f0, target_width),
            resize_frame(mid, target_width),
            resize_frame(f1, target_width),
        ]

    margin = n // 10
    interior = chunk_frames[margin : n - margin]
    m = len(interior)
    t1 = max(1, m // 3)
    t2 = max(t1 + 1, (2 * m) // 3)
    thirds = (interior[:t1], interior[t1:t2], interior[t2:])
    return [
        resize_frame(max(seg, key=lambda item: item[1])[0], target_width)
        for seg in thirds
    ]


def process_video_single_pass(
    video_path: str | Path,
    threshold: float = 0.5,
    min_shot_frames: int = 12,
    resize_width: int = 160,
    entropy_stride: int = 1,
    target_width: int = 512,
    max_shot_duration: float = 8.0,
) -> tuple[list[VideoChunk], dict[int, list[NDArray[np.uint8]]]]:
    """Segmenta el vídeo en planos (SSIM + duración máxima) y extrae 3 fotogramas clave en una pasada."""
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"No existe el archivo de vídeo: {path}")
    ChunkingConfig(
        threshold=threshold,
        min_shot_frames=min_shot_frames,
        max_shot_duration=max_shot_duration,
        resize_width=resize_width,
        entropy_stride=entropy_stride,
    )
    if target_width < 0:
        raise ValueError(f"target_width debe ser >= 0: {target_width}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV no pudo abrir el vídeo: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    max_shot_frames = max(min_shot_frames, round(max_shot_duration * fps))

    chunks: list[VideoChunk] = []
    frames_by_index: dict[int, list[NDArray[np.uint8]]] = {}

    current_chunk_idx = 0
    chunk_start_frame = 0
    prev_gray: NDArray[np.uint8] | None = None
    chunk_frames: list[tuple[NDArray[np.uint8], float]] = []
    frame_idx = 0

    def _finalize_chunk(end_idx: int, next_frame_idx: int) -> None:
        chunks.append(
            VideoChunk(
                index=current_chunk_idx,
                start_frame=chunk_start_frame,
                end_frame=end_idx,
                start_time=chunk_start_frame / fps,
                end_time=float(next_frame_idx) / fps,
            )
        )
        frames_by_index[current_chunk_idx] = _select_chunk_keyframes(
            chunk_frames, target_width
        )
        chunk_frames.clear()

    try:
        while True:
            ok, raw_frame = cap.read()
            if not ok or raw_frame is None:
                break
            frame = cast("NDArray[np.uint8]", raw_frame)

            height, width = frame.shape[:2]
            new_h = max(8, int(height * resize_width / width))
            small = cv2.resize(frame, (resize_width, new_h), interpolation=cv2.INTER_AREA)
            gray = cast("NDArray[np.uint8]", cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))

            is_boundary = False
            shot_len = frame_idx - chunk_start_frame
            if shot_len >= max_shot_frames:
                is_boundary = True
            elif prev_gray is not None:
                score = compute_ssim(prev_gray, gray)
                if score < threshold and shot_len >= min_shot_frames:
                    is_boundary = True

            if is_boundary and chunk_frames:
                _finalize_chunk(frame_idx - 1, frame_idx)
                current_chunk_idx += 1
                chunk_start_frame = frame_idx

            offset = frame_idx - chunk_start_frame
            ent = (
                compute_image_entropy(gray)
                if offset % entropy_stride == 0
                else -1.0
            )
            chunk_frames.append((frame, ent))
            prev_gray = gray
            frame_idx += 1

        if chunk_frames:
            _finalize_chunk(frame_idx - 1, frame_idx)

    finally:
        cap.release()

    return chunks, frames_by_index
