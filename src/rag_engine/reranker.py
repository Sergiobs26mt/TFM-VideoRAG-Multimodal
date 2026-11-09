"""Módulo de reordenación y verificación multimodal (Etapa 2) con Gemini Flash Lite."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from google.genai import types
from numpy.typing import NDArray

from src.config import PROCESSED_DIR, RAW_DIR, RERANKER_MODEL
from src.models import RerankedResult, SearchResult, VectorizationError
from src.processing.keyframe_cache import (
    load_chunk_keyframes_jpeg,
    save_chunk_keyframes_jpeg,
)
from src.processing.video_chunker import (
    encode_frame_jpeg,
    extract_interval_frames,
    resize_frame,
)
from src.utils.gemini_client import call_with_retry, create_gemini_client

logger = logging.getLogger(__name__)


def _parse_evaluations_json(raw_json: str) -> list[dict[str, Any]]:
    """Parsea la lista JSON de evaluaciones o recupera los objetos completos si hubo truncamiento."""
    try:
        parsed = json.loads(raw_json)
        if not isinstance(parsed, list):
            raise TypeError("La respuesta de Gemini no es una lista JSON.")
        return [item for item in parsed if isinstance(item, dict)]
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        recovered: list[dict[str, Any]] = []
        pos = 0
        while True:
            brace = raw_json.find("{", pos)
            if brace == -1:
                break
            try:
                obj, end = decoder.raw_decode(raw_json, brace)
                if isinstance(obj, dict) and "candidate_index" in obj:
                    recovered.append(obj)
                pos = end
            except json.JSONDecodeError:
                pos = brace + 1
        if recovered:
            logger.warning(
                "Respuesta JSON truncada; recuperadas %d evaluaciones completas.",
                len(recovered),
            )
            return recovered
        raise


class MultimodalReranker:
    """Reordenador y verificador multimodal basado en Gemini Flash Lite."""

    def __init__(
        self,
        model: str = RERANKER_MODEL,
        api_key: str | None = None,
        max_retries: int = 4,
        retry_base_delay: float = 1.5,
        target_width: int = 384,
        jpeg_quality: int = 80,
    ) -> None:
        self._client = create_gemini_client(api_key=api_key)
        self._model = model
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay
        self._target_width = target_width
        self._jpeg_quality = jpeg_quality

    async def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        raw_dir: Path | str = RAW_DIR,
        processed_dir: Path | str = PROCESSED_DIR,
        frames_per_candidate: int = 1,
    ) -> list[RerankedResult]:
        """Verifica visualmente y reordena los candidatos en una única llamada por lotes."""
        if not candidates:
            return []

        base_raw = Path(raw_dir)
        base_processed = Path(processed_dir)
        contents: list[Any] = [
            (
                "Eres un juez experto para un sistema de Video RAG multimodal. "
                f"El usuario busca: '{query}'.\n"
                f"A continuación se presentan {len(candidates)} fragmentos de vídeo candidatos "
                "recuperados previamente mediante similitud vectorial.\n"
                "Para cada candidato se proporcionan sus fotogramas clave representativos "
                "(como imagen individual o tira vertical con inicio, mitad y fin del fragmento) "
                "y su transcripción hablada alineada temporalmente.\n\n"
                "Instrucciones de evaluación bimodal (Visual + Auditiva):\n"
                "1. Realiza un análisis bimodal: valida el candidato si el concepto buscado aparece "
                "visualmente en los fotogramas, si se menciona o explica en la transcripción hablada "
                "del segmento, o en ambos.\n"
                "2. No descartes un fragmento como Falso Positivo si el concepto buscado se explica "
                "en la transcripción hablada aunque el texto en pantalla sea pequeño o no se aprecie "
                "en el fotograma puntual, ni viceversa.\n"
                "3. Asigna 'is_relevant': true si el contenido visual o auditivo cumple la consulta. "
                "Si no guarda relación ni en imagen ni en voz, marca false.\n"
                "4. Asigna 'relevance_score' entre 0.0 y 1.0 (1.0 = certeza total, 0.0 = totalmente no relacionado).\n"
                "5. Proporciona en 'reasoning' una frase muy breve en español (máximo 12 palabras).\n\n"
                "Responde ESTRICTAMENTE con un array JSON con la siguiente estructura para cada candidato:\n"
                "[\n"
                "  {\n"
                '    "candidate_index": int,\n'
                '    "is_relevant": bool,\n'
                '    "relevance_score": float,\n'
                '    "reasoning": str\n'
                "  }\n"
                "]"
            )
        ]

        jpeg_map = await asyncio.to_thread(
            self._collect_candidate_jpegs,
            candidates,
            base_raw,
            base_processed,
            frames_per_candidate,
        )

        has_any_frame = False
        for idx, cand in enumerate(candidates):
            transcript_line = (
                f"Transcripción hablada en este segmento: {cand.transcript.strip()}"
                if cand.transcript and cand.transcript.strip()
                else "Transcripción hablada: [Sin locución detectada]"
            )
            header = (
                f"\n--- Candidato {idx} (Vídeo: {cand.video_name}, "
                f"Intervalo: {cand.start_time:.1f}s - {cand.end_time:.1f}s) ---\n"
                f"{transcript_line}"
            )
            contents.append(header)
            for jpeg_bytes in jpeg_map.get(idx, []):
                has_any_frame = True
                contents.append(
                    types.Part.from_bytes(
                        data=jpeg_bytes,
                        mime_type="image/jpeg",
                    )
                )

        if not has_any_frame:
            logger.warning(
                "No se pudieron extraer fotogramas para los candidatos. Omitiendo reranking."
            )
            return [
                RerankedResult(
                    result=c,
                    relevance_score=c.score,
                    is_relevant=True,
                    reasoning="Fotogramas no disponibles en disco; conservando orden vectorial.",
                )
                for c in candidates
            ]

        try:
            raw_json = await self._call_gemini_with_retry(contents)
            evaluations = _parse_evaluations_json(raw_json)
        except (
            VectorizationError,
            json.JSONDecodeError,
            ValueError,
            TypeError,
            OSError,
            RuntimeError,
        ) as exc:
            logger.error(
                "Error durante la reordenación multimodal: %s. Aplicando fallback.", exc
            )
            return [
                RerankedResult(
                    result=c,
                    relevance_score=c.score,
                    is_relevant=True,
                    reasoning=f"Reordenación visual no disponible ({exc}); conservando orden vectorial.",
                )
                for c in candidates
            ]

        eval_by_index: dict[int, dict[str, Any]] = {}
        for ev in evaluations:
            if isinstance(ev, dict) and "candidate_index" in ev:
                eval_by_index[int(ev["candidate_index"])] = ev

        reranked: list[RerankedResult] = []
        for idx, cand in enumerate(candidates):
            cand_eval = eval_by_index.get(idx)
            if cand_eval:
                rel_score = float(cand_eval.get("relevance_score", 0.0))
                is_rel = bool(cand_eval.get("is_relevant", rel_score >= 0.5))
                reason = str(cand_eval.get("reasoning", "Sin explicación proporcionada."))
            else:
                rel_score = cand.score
                is_rel = True
                reason = "No evaluado explícitamente por el modelo."

            reranked.append(
                RerankedResult(
                    result=cand,
                    relevance_score=max(0.0, min(1.0, rel_score)),
                    is_relevant=is_rel,
                    reasoning=reason,
                )
            )

        reranked.sort(
            key=lambda r: (1 if r.is_relevant else 0, r.relevance_score, r.result.score),
            reverse=True,
        )
        return reranked

    def _normalize_jpeg_bytes(self, jpeg_bytes: bytes) -> bytes:
        """Re-codifica el JPEG si su ancho supera ``_target_width`` para reducir el payload."""
        if self._target_width <= 0:
            return jpeg_bytes
        buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None or frame.shape[1] <= self._target_width:
            return jpeg_bytes
        return encode_frame_jpeg(
            cast("NDArray[np.uint8]", frame),
            jpeg_quality=self._jpeg_quality,
            target_width=self._target_width,
        )

    def _build_vertical_collage(self, jpeg_list: list[bytes]) -> bytes:
        """Apila verticalmente los fotogramas distintos de un candidato en un único JPEG."""
        distinct: list[NDArray[np.uint8]] = []
        for raw_bytes in jpeg_list:
            buf = np.frombuffer(raw_bytes, dtype=np.uint8)
            decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if decoded is None:
                continue
            resized = resize_frame(
                cast("NDArray[np.uint8]", decoded), self._target_width
            )
            if distinct and resized.shape[1] != distinct[0].shape[1]:
                target_w = distinct[0].shape[1]
                new_h = max(8, int(resized.shape[0] * target_w / resized.shape[1]))
                resized = cast(
                    "NDArray[np.uint8]",
                    cv2.resize(resized, (target_w, new_h), interpolation=cv2.INTER_AREA),
                )
            if not distinct or not np.array_equal(resized, distinct[-1]):
                distinct.append(resized)

        if not distinct:
            return jpeg_list[0]
        if len(distinct) == 1:
            return self._normalize_jpeg_bytes(jpeg_list[0])

        collage = cast("NDArray[np.uint8]", cv2.vconcat(distinct))
        return encode_frame_jpeg(
            collage,
            jpeg_quality=self._jpeg_quality,
            target_width=0,
        )

    def _collect_candidate_jpegs(
        self,
        candidates: list[SearchResult],
        base_raw: Path,
        base_processed: Path,
        frames_per_candidate: int,
    ) -> dict[int, list[bytes]]:
        """Obtiene los fotogramas JPEG de cada candidato desde caché o extrayéndolos del vídeo."""
        jpeg_map: dict[int, list[bytes]] = {}
        cache_limit = max(3, frames_per_candidate)
        for idx, cand in enumerate(candidates):
            cached = load_chunk_keyframes_jpeg(
                cand.video_name,
                cand.chunk_index,
                processed_dir=base_processed,
                max_frames=cache_limit,
            )
            if cached:
                if frames_per_candidate == 1 and len(cached) > 1:
                    jpeg_map[idx] = [self._build_vertical_collage(cached)]
                else:
                    jpeg_map[idx] = [
                        self._normalize_jpeg_bytes(b)
                        for b in cached[:frames_per_candidate]
                    ]
                continue

            vpath = base_raw / cand.video_name
            raw_frames = extract_interval_frames(
                vpath,
                cand.start_time,
                cand.end_time,
                num_frames=frames_per_candidate,
                target_width=self._target_width,
            )
            encoded = [
                encode_frame_jpeg(
                    frame,
                    jpeg_quality=self._jpeg_quality,
                    target_width=self._target_width,
                )
                for frame in raw_frames
            ]
            if encoded:
                save_chunk_keyframes_jpeg(
                    cand.video_name,
                    cand.chunk_index,
                    encoded,
                    processed_dir=base_processed,
                )
            jpeg_map[idx] = encoded
        return jpeg_map

    async def _call_gemini_with_retry(self, contents: list[Any]) -> str:
        """Invoca Gemini Flash Lite de forma asíncrona con nivel de pensamiento mínimo."""
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1,
            max_output_tokens=2048,
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.MINIMAL
            ),
        )
        response = await call_with_retry(
            lambda: self._client.aio.models.generate_content(
                model=self._model,
                contents=contents,
                config=config,
            ),
            label=f"rerank ({self._model})",
            max_retries=self._max_retries,
            retry_base_delay=self._retry_base_delay,
        )
        text = response.text or ""
        return text.strip()
