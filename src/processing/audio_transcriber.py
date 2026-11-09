"""Extracción de pista de audio, transcripción ASR con Whisper y alineación temporal."""

from __future__ import annotations

import importlib
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from src.config import AUDIO_LANGUAGE, WHISPER_MODEL
from src.models import AudioTranscriptionError, TranscriptSegment, VideoChunk

logger = logging.getLogger(__name__)

whisper: Any = None
_MODEL_CACHE: dict[str, Any] = {}

__all__ = [
    "AudioTranscriptionError",
    "TranscriptSegment",
    "align_transcript_with_chunks",
    "extract_audio_track",
    "transcribe_audio",
    "transcribe_video_audio",
]


def _resolve_ffmpeg_executable() -> str:
    """Garantiza que el directorio del entorno virtual esté en ``PATH`` y localiza ``ffmpeg``."""
    venv_bin = str(Path(sys.executable).parent)
    current_path = os.environ.get("PATH", "")
    if venv_bin not in current_path.split(os.pathsep):
        os.environ["PATH"] = f"{venv_bin}{os.pathsep}{current_path}"

    if shutil.which("ffmpeg") is not None:
        return "ffmpeg"

    try:
        imageio_ffmpeg: Any = importlib.import_module("imageio_ffmpeg")
        exe = str(imageio_ffmpeg.get_ffmpeg_exe())
        if Path(exe).is_file():
            return exe
    except (ImportError, RuntimeError, OSError):
        logger.debug("No se pudo resolver ffmpeg desde imageio-ffmpeg", exc_info=True)
    return "ffmpeg"


def extract_audio_track(
    video_path: str | Path,
    output_wav: Path | None = None,
) -> Path | None:
    """Extrae el audio de un vídeo en un WAV mono a 16 kHz o devuelve ``None`` si no hay audio."""
    source = Path(video_path)
    if not source.is_file():
        raise FileNotFoundError(f"No existe el archivo de vídeo: {source}")

    if output_wav is not None:
        wav_path = Path(output_wav)
    else:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = Path(tmp.name)

    ffmpeg_bin = _resolve_ffmpeg_executable()
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(source),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(wav_path),
    ]

    try:
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        wav_path.unlink(missing_ok=True)
        raise AudioTranscriptionError(
            f"No se pudo ejecutar ffmpeg para extraer el audio de '{source}': {exc}"
        ) from exc

    if (
        completed.returncode != 0
        or not wav_path.is_file()
        or wav_path.stat().st_size == 0
    ):
        wav_path.unlink(missing_ok=True)
        stderr_tail = (completed.stderr or "").strip().splitlines()
        detail = stderr_tail[-1] if stderr_tail else f"código {completed.returncode}"
        logger.warning(
            "Sin pista de audio utilizable en '%s' (%s); omitiendo transcripción.",
            source,
            detail,
        )
        return None

    return wav_path


def _load_whisper_model(model_name: str) -> Any:
    """Carga y almacena en caché de módulo el modelo Whisper solicitado."""
    global whisper
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    if whisper is None:
        try:
            whisper = importlib.import_module("whisper")
        except ImportError as exc:
            raise AudioTranscriptionError(
                "El paquete 'openai-whisper' no está instalado."
            ) from exc

    try:
        model = whisper.load_model(model_name)
    except Exception as exc:
        raise AudioTranscriptionError(
            f"Error al cargar el modelo Whisper '{model_name}': {exc}"
        ) from exc

    _MODEL_CACHE[model_name] = model
    return model


def _default_transcriber(
    audio_path: str,
    model_name: str,
    language: str | None,
) -> list[dict[str, Any]]:
    """Ejecuta la transcripción con ``openai-whisper`` en CPU/FP32."""
    _resolve_ffmpeg_executable()
    model = _load_whisper_model(model_name)
    try:
        result: dict[str, Any] = model.transcribe(
            audio_path,
            language=language,
            fp16=False,
        )
    except Exception as exc:
        raise AudioTranscriptionError(
            f"Error al transcribir '{audio_path}' con Whisper: {exc}"
        ) from exc

    raw_segments = result.get("segments", [])
    if not isinstance(raw_segments, list):
        return []
    return [seg for seg in raw_segments if isinstance(seg, dict)]


def transcribe_audio(
    audio_path: str | Path,
    model_name: str = WHISPER_MODEL,
    language: str | None = AUDIO_LANGUAGE,
    transcriber_fn: Callable[[str, str, str | None], list[dict[str, Any]]] | None = None,
) -> list[TranscriptSegment]:
    """Transcribe un archivo de audio y devuelve los segmentos no vacíos ordenados."""
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"No existe el archivo de audio: {path}")

    runner = transcriber_fn if transcriber_fn is not None else _default_transcriber
    raw_segments = runner(str(path), model_name, language)

    segments: list[TranscriptSegment] = []
    for item in raw_segments:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        start = float(item.get("start", item.get("start_time", 0.0)))
        end = float(item.get("end", item.get("end_time", start)))
        segments.append(
            TranscriptSegment(
                start_time=start,
                end_time=max(start, end),
                text=text,
            )
        )

    segments.sort(key=lambda s: (s.start_time, s.end_time))
    return segments


def transcribe_video_audio(
    video_path: str | Path,
    model_name: str = WHISPER_MODEL,
    language: str | None = AUDIO_LANGUAGE,
    transcriber_fn: Callable[[str, str, str | None], list[dict[str, Any]]] | None = None,
) -> list[TranscriptSegment]:
    """Extrae el audio temporal de un vídeo, lo transcribe y limpia el archivo WAV."""
    wav_path = extract_audio_track(video_path)
    if wav_path is None:
        return []

    try:
        return transcribe_audio(
            wav_path,
            model_name=model_name,
            language=language,
            transcriber_fn=transcriber_fn,
        )
    finally:
        wav_path.unlink(missing_ok=True)


def align_transcript_with_chunks(
    chunks: list[VideoChunk],
    segments: list[TranscriptSegment],
    min_overlap: float = 0.2,
) -> list[VideoChunk]:
    """Asigna y concatena segmentos de locución a cada ``VideoChunk`` por solapamiento temporal."""
    if not chunks:
        return []
    if not segments:
        return [replace(chunk, transcript=None) for chunk in chunks]

    ordered_segments = sorted(segments, key=lambda s: (s.start_time, s.end_time))
    aligned: list[VideoChunk] = []

    for chunk in chunks:
        matched: list[str] = []
        for seg in ordered_segments:
            cleaned = seg.text.strip()
            if not cleaned:
                continue
            overlap = min(chunk.end_time, seg.end_time) - max(
                chunk.start_time, seg.start_time
            )
            is_contained = (
                seg.start_time >= chunk.start_time
                and seg.end_time <= chunk.end_time
                and overlap > 0.0
            )
            if overlap >= min_overlap or is_contained:
                matched.append(cleaned)

        transcript = " ".join(matched) if matched else None
        aligned.append(replace(chunk, transcript=transcript))

    return aligned
