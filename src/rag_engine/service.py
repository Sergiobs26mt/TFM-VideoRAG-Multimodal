"""Fachada de aplicación que unifica indexación, búsqueda en dos etapas y gestión de BD."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from pathlib import Path

from src.config import (
    DB_PATH,
    PROCESSED_DIR,
    RAW_DIR,
    RERANKER_MODEL,
    VECTOR_SIZE,
    WHISPER_MODEL,
    SearchConfig,
)
from src.embeddings.gemini_vectorizer import GeminiMultimodalVectorizer
from src.models import RerankedResult, SearchResult, VectorizationError
from src.processing.pipeline import process_video_pipeline
from src.rag_engine.reranker import MultimodalReranker
from src.rag_engine.vector_db import QdrantSearchEngine
from src.utils.audit import RunAuditor

logger = logging.getLogger(__name__)


def format_timestamp(seconds: float) -> str:
    """Convierte segundos a formato ``MM:SS.s`` legible."""
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes):02d}:{secs:04.1f}"


def _merge_transcripts(first: str | None, second: str | None) -> str | None:
    """Combina dos transcripciones cronológicas sin duplicar frases fronterizas repetidas."""
    t1 = (first or "").strip()
    t2 = (second or "").strip()
    if not t1 and not t2:
        return None
    if not t1:
        return t2
    if not t2 or t1 == t2 or t2 in t1:
        return t1
    if t1 in t2:
        return t2

    w1 = t1.split()
    w2 = t2.split()
    max_overlap = min(len(w1), len(w2))
    overlap_len = 0
    for k in range(max_overlap, 0, -1):
        if w1[-k:] == w2[:k]:
            overlap_len = k
            break

    combined = " ".join(w1 + w2[overlap_len:])
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", combined) if s.strip()]
    deduped: list[str] = []
    for sentence in sentences:
        if not deduped or deduped[-1] != sentence:
            deduped.append(sentence)
    return " ".join(deduped)


def _deduplicate_temporal_candidates(
    candidates: list[SearchResult],
    *,
    micro_chunk_duration: float = 1.0,
    min_scene_separation: float = 4.0,
) -> list[SearchResult]:
    """Fusiona micro-fragmentos redundantes de la misma escena conservando el de mayor similitud."""
    selected: list[SearchResult] = []
    origins: list[tuple[float, float, str]] = []
    for cand in candidates:
        cand_dur = cand.end_time - cand.start_time
        cand_mid = (cand.start_time + cand.end_time) / 2.0
        cand_tr = (cand.transcript or "").strip()
        merged = False
        for idx, prev in enumerate(selected):
            if prev.video_name != cand.video_name:
                continue
            orig_dur, orig_mid, orig_tr = origins[idx]
            gap = max(
                0.0,
                max(cand.start_time, prev.start_time)
                - min(cand.end_time, prev.end_time),
            )
            is_micro_neighbor = (
                max(cand_dur, orig_dur) < micro_chunk_duration
                and abs(cand_mid - orig_mid) < min_scene_separation
            )
            is_same_transcript_neighbor = bool(
                cand_tr
                and orig_tr
                and cand_tr == orig_tr
                and gap <= min_scene_separation
            )
            if is_micro_neighbor or is_same_transcript_neighbor:
                earlier, later = (
                    (prev, cand) if prev.start_time <= cand.start_time else (cand, prev)
                )
                selected[idx] = SearchResult(
                    score=prev.score,
                    video_name=prev.video_name,
                    chunk_index=prev.chunk_index,
                    start_time=min(prev.start_time, cand.start_time),
                    end_time=max(prev.end_time, cand.end_time),
                    transcript=_merge_transcripts(
                        earlier.transcript, later.transcript
                    ),
                )
                merged = True
                break
        if not merged:
            selected.append(cand)
            origins.append((cand_dur, cand_mid, cand_tr))
    return selected


class VideoRAGService:
    """Coordina el pipeline de indexación, la búsqueda multimodal y la base de datos."""

    def __init__(
        self,
        engine: QdrantSearchEngine | None = None,
        vectorizer: GeminiMultimodalVectorizer | None = None,
        reranker: MultimodalReranker | None = None,
        *,
        db_path: str | None = DB_PATH,
        raw_dir: Path = RAW_DIR,
        processed_dir: Path = PROCESSED_DIR,
        vector_size: int = VECTOR_SIZE,
        reranker_model: str = RERANKER_MODEL,
    ) -> None:
        self.engine = engine if engine is not None else QdrantSearchEngine(path=db_path)
        self._vectorizer = vectorizer
        self._reranker = reranker
        self._raw_dir = Path(raw_dir)
        self._processed_dir = Path(processed_dir)
        self._vector_size = vector_size
        self._reranker_model = reranker_model
        self._query_cache: dict[str, list[float]] = {}

    def _get_vectorizer(self) -> GeminiMultimodalVectorizer:
        if self._vectorizer is None:
            self._vectorizer = GeminiMultimodalVectorizer(
                output_dimensionality=self._vector_size
            )
        return self._vectorizer

    def _get_reranker(self) -> MultimodalReranker:
        if self._reranker is None:
            self._reranker = MultimodalReranker(model=self._reranker_model)
        return self._reranker

    def _resolve_video_file(self, video_path_or_name: str | Path) -> Path:
        candidate = Path(video_path_or_name)
        if candidate.is_file():
            return candidate
        in_raw = self._raw_dir / candidate.name
        if in_raw.is_file():
            return in_raw
        raise FileNotFoundError(
            f"No existe {in_raw}. Colócalo en {self._raw_dir}/."
        )

    async def index_video(
        self,
        video_path_or_name: str | Path,
        *,
        max_concurrent: int = 3,
        threshold: float = 0.5,
        min_shot_frames: int = 12,
        max_shot_duration: float = 8.0,
        resize_width: int = 160,
        entropy_stride: int = 1,
        jpeg_quality: int = 80,
        output_dimensionality: int | None = None,
        target_width: int = 512,
        enable_audio: bool = True,
        whisper_model: str = WHISPER_MODEL,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> int:
        """Fragmenta, vectoriza e indexa un vídeo en Qdrant."""
        video_path = self._resolve_video_file(video_path_or_name)
        dim = output_dimensionality if output_dimensionality is not None else self._vector_size

        chunks = await process_video_pipeline(
            str(video_path),
            max_concurrent_requests=max_concurrent,
            progress_callback=progress_callback,
            threshold=threshold,
            min_shot_frames=min_shot_frames,
            max_shot_duration=max_shot_duration,
            resize_width=resize_width,
            entropy_stride=entropy_stride,
            jpeg_quality=jpeg_quality,
            output_dimensionality=dim,
            target_width=target_width,
            enable_audio=enable_audio,
            whisper_model=whisper_model,
        )
        self.engine.create_collection(vector_size=dim)
        inserted = self.engine.insert_chunks(chunks, video_name=video_path.name)
        logger.info("Indexación completada: %d chunks", inserted)
        return inserted

    async def search(
        self,
        query: str,
        *,
        limit: int = 5,
        video_filter: str | None = None,
        rerank: bool = True,
        oversample_factor: int = 2,
        auditor: RunAuditor | None = None,
    ) -> list[SearchResult] | list[RerankedResult]:
        """Ejecuta la recuperación en dos etapas (Qdrant + VLM opcional) y registra auditoría."""
        cfg = SearchConfig(
            top_k=limit, rerank=rerank, reranker_model=self._reranker_model
        )
        if oversample_factor < 1:
            raise ValueError("oversample_factor debe ser >= 1.")

        fetch_k = (
            max(cfg.top_k, min(cfg.top_k * oversample_factor, 10))
            if cfg.rerank
            else cfg.top_k
        )
        raw_fetch_k = max(fetch_k * 3, 25)
        cache_key = query.strip()
        query_vector = self._query_cache.get(cache_key)
        if query_vector is None:
            query_vector = await self._get_vectorizer().embed_text(query)
            self._query_cache[cache_key] = query_vector
        raw_candidates = self.engine.search_event(
            query_vector, limit=raw_fetch_k, video_name=video_filter
        )
        candidates = _deduplicate_temporal_candidates(raw_candidates)[:fetch_k]

        final_results: list[SearchResult] | list[RerankedResult] = candidates[: cfg.top_k]
        if cfg.rerank and candidates:
            try:
                reranked = await self._get_reranker().rerank(
                    query,
                    candidates,
                    raw_dir=self._raw_dir,
                    processed_dir=self._processed_dir,
                )
                final_results = reranked[: cfg.top_k]
            except (VectorizationError, ValueError, RuntimeError, OSError) as exc:
                logger.warning(
                    "Fallo en reordenación multimodal (%s), conservando orden vectorial",
                    exc,
                )

        if auditor is not None:
            auditor.save_search_results(query, list(final_results))
        return final_results

    def get_indexed_videos(self) -> list[str]:
        """Devuelve la lista ordenada de vídeos indexados en la base de datos."""
        return self.engine.get_indexed_videos()

    def delete_video(self, video_name: str) -> None:
        """Elimina los vectores y fotogramas cacheados de un vídeo."""
        self.engine.delete_video(video_name)

    def clear_database(self) -> None:
        """Vacía la colección de vectores y la caché de fotogramas clave."""
        self.engine.clear_database(vector_size=self._vector_size)

