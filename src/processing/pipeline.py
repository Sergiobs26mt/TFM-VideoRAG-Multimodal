"""Orquestador del flujo completo: vídeo crudo -> chunks -> fotogramas -> embeddings."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path

from src.config import (
    DEFAULT_EMBEDDING_BACKEND,
    EMBEDDING_BACKENDS,
    MAX_CHUNKS_PER_VIDEO,
    PROCESSED_DIR,
    VECTOR_SIZE,
    WHISPER_MODEL,
    validate_embedding_choice,
)
from src.embeddings.factory import create_vectorizer
from src.models import (
    AudioTranscriptionError,
    QuotaSafetyError,
    VectorizationError,
    VectorizedChunk,
    VideoChunk,
)
from src.processing.audio_transcriber import (
    align_transcript_with_chunks,
    transcribe_video_audio,
    unload_whisper_models,
)
from src.processing.keyframe_cache import save_video_keyframes
from src.processing.video_chunker import process_video_single_pass

logger = logging.getLogger(__name__)


async def process_video_pipeline(
    video_path: str,
    max_concurrent_requests: int = 3,
    progress_callback: Callable[[str, float], None] | None = None,
    threshold: float = 0.5,
    min_shot_frames: int = 12,
    resize_width: int = 160,
    entropy_stride: int = 1,
    jpeg_quality: int = 80,
    output_dimensionality: int = VECTOR_SIZE,
    target_width: int = 512,
    max_shot_duration: float = 8.0,
    enable_audio: bool = True,
    whisper_model: str = WHISPER_MODEL,
    embedding_backend: str = DEFAULT_EMBEDDING_BACKEND,
) -> list[VectorizedChunk]:
    """Procesa un vídeo de principio a fin y devuelve sus chunks vectorizados.

    Pasos:
        1. (Síncrono) Segmentación y extracción de fotogramas clave en UNA SOLA pasada
           secuencial del vídeo (SSIM + entropía) -> ``VideoChunk`` y fotogramas.
        2. (Asíncrono) Vectorización con el backend ``embedding_backend``, redimensionando
           fotogramas a ``target_width`` y limitando la concurrencia con un ``asyncio.Semaphore``.
           Los backends locales usan siempre concurrencia 1 y descargan Whisper antes de cargar el modelo.

    Los chunks cuya vectorización falla de forma definitiva se registran en el
    log y se omiten, de modo que un fallo puntual no descarte todo el vídeo.

    Args:
        video_path: Ruta al vídeo (normalmente dentro de ``data/raw``).
        max_concurrent_requests: Máximo de peticiones simultáneas a la API (1 en backends locales).
        progress_callback: Función opcional ``callback(mensaje, progreso_0_a_1)``
            para reportar el avance en tiempo real (útil en interfaces gráficas).
        threshold: Umbral SSIM para detección de planos (menor = menos cortes).
        min_shot_frames: Mínimo de fotogramas por plano para evitar microcortes.
        resize_width: Ancho de redimensionado para cálculo de SSIM y entropía rápida.
        entropy_stride: Paso de muestreo de fotogramas para cálculo de entropía.
        jpeg_quality: Calidad de compresión JPEG de las imágenes para Gemini (1-100).
        output_dimensionality: Dimensión de los vectores de embedding generados.
        target_width: Ancho máximo de redimensionado de fotogramas previo a la vectorización.
        embedding_backend: Clave del backend en ``EMBEDDING_BACKENDS``.

    Returns:
        Lista de ``VectorizedChunk`` ordenada por índice temporal.

    Raises:
        ValueError: Si ``max_concurrent_requests`` es menor que 1 o el backend y la
            dimensión no son compatibles.
        FileNotFoundError: Si el vídeo no existe.
        QuotaSafetyError: Si el vídeo genera más de ``MAX_CHUNKS_PER_VIDEO`` chunks.
    """
    if max_concurrent_requests < 1:
        raise ValueError("max_concurrent_requests debe ser >= 1.")
    validate_embedding_choice(embedding_backend, output_dimensionality)
    is_local = EMBEDDING_BACKENDS[embedding_backend].is_local

    if progress_callback:
        progress_callback("Analizando vídeo y extrayendo fotogramas (pasada única)...", 0.05)

    # Paso 1: segmentación y fotogramas en una única pasada secuencial
    logger.info("Procesando vídeo en pasada única (SSIM + fotogramas): %s", video_path)
    chunks, frames_by_index = await asyncio.to_thread(
        process_video_single_pass,
        video_path,
        threshold,
        min_shot_frames,
        resize_width,
        entropy_stride,
        target_width,
        max_shot_duration,
    )
    logger.info("Se han detectado %d chunks y extraído fotogramas", len(chunks))

    # Fail-safe: abortar antes de gastar ninguna petición de la API (solo backends en la nube)
    if not is_local and len(chunks) > MAX_CHUNKS_PER_VIDEO:
        message = (
            f"El vídeo genera {len(chunks)} chunks, lo cual excede el límite de "
            f"seguridad de {MAX_CHUNKS_PER_VIDEO} para proteger la cuota diaria de "
            "la API (1.000 RPD). Por favor, utiliza un vídeo más corto o ajusta los "
            "parámetros de fragmentación."
        )
        logger.error(message)
        raise QuotaSafetyError(message)

    if not chunks:
        if progress_callback:
            progress_callback("No se detectaron fotogramas en el vídeo.", 1.0)
        return []

    if enable_audio:
        if progress_callback:
            progress_callback(
                "Transcribiendo audio y alineando locución con los planos...", 0.25
            )
        try:
            segments = await asyncio.to_thread(
                transcribe_video_audio, video_path, whisper_model
            )
            if segments:
                chunks = align_transcript_with_chunks(chunks, segments)
                enriched = sum(1 for c in chunks if c.transcript)
                logger.info(
                    "Audio alineado: %d/%d chunks enriquecidos con transcripción",
                    enriched,
                    len(chunks),
                )
        except (AudioTranscriptionError, RuntimeError, OSError, ValueError) as exc:
            logger.warning(
                "Fallo en transcripción de audio; continuando solo con modalidad visual: %s",
                exc,
            )

    await asyncio.to_thread(
        save_video_keyframes,
        Path(video_path).name,
        frames_by_index,
        PROCESSED_DIR,
        jpeg_quality,
        target_width,
    )

    if progress_callback:
        progress_callback(
            f"Fotogramas listos. Vectorizando {len(chunks)} chunks con {embedding_backend}...",
            0.40,
        )

    if is_local:
        unload_whisper_models()
        if progress_callback:
            progress_callback(
                "Cargando modelo local (la primera vez descarga varios GB)...", 0.40
            )
        if max_concurrent_requests > 1:
            logger.warning(
                "Backend local '%s': se usa concurrencia 1 en lugar de %d",
                embedding_backend,
                max_concurrent_requests,
            )
        max_concurrent_requests = 1

    # Paso 2: vectorización asíncrona con concurrencia limitada
    vectorizer = await asyncio.to_thread(
        create_vectorizer,
        embedding_backend,
        output_dimensionality,
        jpeg_quality,
        target_width,
    )
    semaphore = asyncio.Semaphore(max_concurrent_requests)

    completed_count = 0

    async def _vectorize(chunk: VideoChunk) -> VectorizedChunk:
        nonlocal completed_count
        async with semaphore:
            logger.debug("Vectorizando chunk %d", chunk.index)
            res = await vectorizer.vectorize_chunk(chunk, frames_by_index[chunk.index])
            completed_count += 1
            if progress_callback:
                frac = 0.40 + 0.55 * (completed_count / len(chunks))
                progress_callback(
                    f"Vectorizando con {embedding_backend} ({completed_count}/{len(chunks)})...",
                    frac,
                )
            return res

    results = await asyncio.gather(
        *(_vectorize(c) for c in chunks), return_exceptions=True
    )

    vectorized: list[VectorizedChunk] = []
    for chunk, result in zip(chunks, results):
        if isinstance(result, VectorizedChunk):
            vectorized.append(result)
        elif isinstance(result, (VectorizationError, ValueError)):
            logger.error("Chunk %d omitido: %s", chunk.index, result)
        elif isinstance(result, BaseException):
            raise result  # error inesperado: no se silencia

    vectorized.sort(key=lambda v: v.index)
    logger.info("Pipeline completado: %d/%d chunks vectorizados", len(vectorized), len(chunks))

    if progress_callback:
        progress_callback(
            f"¡Completado! {len(vectorized)} chunks indexados con éxito.",
            1.0,
        )

    return vectorized

