"""Fachada de aplicación que unifica indexación, búsqueda en dos etapas y gestión de BD."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import logging
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import cast

from src.config import (
    DB_PATH,
    DEFAULT_EMBEDDING_BACKEND,
    PROCESSED_DIR,
    RAW_DIR,
    RERANKER_MODEL,
    VECTOR_SIZE,
    WHISPER_MODEL,
    SearchConfig,
    collection_name_for,
    default_dimension,
    validate_embedding_choice,
)
from src.embeddings.base import MultimodalVectorizer
from src.embeddings.factory import create_vectorizer, device_label_for
from src.models import (
    IndexStats,
    MultimodalQuery,
    RerankedResult,
    SearchResult,
    VectorizationError,
)
from src.processing.audio_transcriber import transcribe_query_audios
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
        vectorizer: MultimodalVectorizer | None = None,
        reranker: MultimodalReranker | None = None,
        *,
        db_path: str | None = DB_PATH,
        raw_dir: Path = RAW_DIR,
        processed_dir: Path = PROCESSED_DIR,
        vector_size: int | None = None,
        embedding_backend: str = DEFAULT_EMBEDDING_BACKEND,
        reranker_model: str = RERANKER_MODEL,
    ) -> None:
        default_size = default_dimension(embedding_backend)
        resolved_size = default_size if vector_size is None else vector_size
        if resolved_size < 1:
            raise ValueError("vector_size debe ser >= 1.")
        if engine is None or vectorizer is None:
            # Permite vectores sintéticos en pruebas cuando se inyectan ambos dobles.
            validate_embedding_choice(embedding_backend, resolved_size)
        if engine is None:
            collection = collection_name_for(embedding_backend, resolved_size)
            self.engine = QdrantSearchEngine(path=db_path)
            self.engine.use_collection(collection)
        else:
            self.engine = engine
        self._vectorizer = vectorizer
        self._reranker = reranker
        self._raw_dir = Path(raw_dir)
        self._processed_dir = Path(processed_dir)
        self._embedding_backend = embedding_backend
        self._vector_size = resolved_size
        self._reranker_model = reranker_model
        self._query_cache: dict[str, list[float]] = {}
        self.last_resolved_query: MultimodalQuery | None = None
        self.last_index_stats: IndexStats | None = None

    @property
    def vector_size(self) -> int:
        """Dimensión activa de los vectores de embedding."""
        return self._vector_size

    @property
    def embedding_backend(self) -> str:
        """Backend de embeddings activo."""
        return self._embedding_backend

    def set_embedding(self, backend: str, vector_size: int) -> None:
        """Actualiza backend y dimensión activos, cambia la colección y reinicia el cliente si cambia."""
        collection = collection_name_for(backend, vector_size)
        if backend != self._embedding_backend or vector_size != self._vector_size:
            self._embedding_backend = backend
            self._vector_size = vector_size
            self._vectorizer = None
        self.engine.use_collection(collection)

    def _get_vectorizer(self) -> MultimodalVectorizer:
        if self._vectorizer is None:
            self._vectorizer = create_vectorizer(
                self._embedding_backend, self._vector_size
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
        embedding_backend: str | None = None,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> int:
        """Fragmenta, vectoriza e indexa un vídeo en Qdrant."""
        video_path = self._resolve_video_file(video_path_or_name)
        backend = (
            embedding_backend
            if embedding_backend is not None
            else self._embedding_backend
        )
        default_dim = default_dimension(backend)
        if output_dimensionality is not None:
            dim = output_dimensionality
        elif backend != self._embedding_backend:
            dim = default_dim
        else:
            dim = self._vector_size
        self.set_embedding(backend, dim)

        t0 = time.perf_counter()
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
            embedding_backend=backend,
        )
        self.engine.create_collection(vector_size=dim)
        inserted = self.engine.insert_chunks(chunks, video_name=video_path.name)
        elapsed = time.perf_counter() - t0
        self.last_index_stats = IndexStats(
            backend=backend,
            device_label=device_label_for(backend),
            dimension=dim,
            seconds=elapsed,
            chunks=inserted,
        )
        logger.info("Indexación completada: %d chunks (%.2fs)", inserted, elapsed)
        return inserted

    @staticmethod
    def _build_query_cache_key(
        mquery: MultimodalQuery,
        vector_size: int = VECTOR_SIZE,
        backend: str = DEFAULT_EMBEDDING_BACKEND,
    ) -> str:
        """Calcula una clave SHA-256 determinista sobre backend, dimensión, texto, imágenes y audios."""
        hasher = hashlib.sha256()
        hasher.update(f"backend:{backend}|dim:{vector_size}|text:".encode())
        hasher.update(mquery.composed_text.encode("utf-8"))
        for img in mquery.images:
            hasher.update(b"|img:")
            hasher.update(hashlib.sha256(img).digest())
        for aud_bytes in mquery.audio_bytes:
            hasher.update(b"|aud_bytes:")
            hasher.update(hashlib.sha256(aud_bytes).digest())
        for aud_path in mquery.audio_paths:
            hasher.update(b"|aud_path:")
            if aud_path.is_file():
                hasher.update(hashlib.sha256(aud_path.read_bytes()).digest())
            else:
                hasher.update(str(aud_path).encode("utf-8"))
        return hasher.hexdigest()

    async def _resolve_multimodal_query(
        self,
        query: str | MultimodalQuery,
        *,
        txt_files: Sequence[str | Path | bytes] | None = None,
        image_files: Sequence[str | Path | bytes] | None = None,
        audio_files: Sequence[str | Path | bytes] | None = None,
        whisper_model: str = WHISPER_MODEL,
    ) -> MultimodalQuery:
        """Unifica adjuntos y transcribe las pistas de audio de consulta."""
        base = MultimodalQuery(text=query) if isinstance(query, str) else query

        extra_txts: list[str] = list(base.txt_contents)
        if txt_files:
            for item in txt_files:
                if isinstance(item, bytes):
                    extra_txts.append(item.decode("utf-8", errors="replace"))
                else:
                    extra_txts.append(Path(item).read_text(encoding="utf-8"))

        extra_imgs: list[bytes] = list(base.images)
        if image_files:
            for item in image_files:
                if isinstance(item, bytes):
                    extra_imgs.append(item)
                else:
                    extra_imgs.append(Path(item).read_bytes())

        extra_audio_paths: list[Path] = list(base.audio_paths)
        extra_audio_bytes: list[bytes] = list(base.audio_bytes)
        if audio_files:
            for item in audio_files:
                if isinstance(item, bytes):
                    extra_audio_bytes.append(item)
                else:
                    extra_audio_paths.append(Path(item))

        mquery = replace(
            base,
            txt_contents=tuple(extra_txts),
            images=tuple(extra_imgs),
            audio_paths=tuple(extra_audio_paths),
            audio_bytes=tuple(extra_audio_bytes),
        )
        if not mquery.has_content:
            raise ValueError("La consulta de búsqueda no puede estar vacía.")

        audio_inputs: list[str | Path | bytes] = [
            *mquery.audio_paths,
            *mquery.audio_bytes,
        ]
        if audio_inputs:
            transcribed = await asyncio.to_thread(
                transcribe_query_audios,
                audio_inputs,
                model_name=whisper_model,
            )
            mquery = replace(
                mquery,
                audio_transcripts=(*mquery.audio_transcripts, *transcribed),
            )

        if not mquery.composed_text and not mquery.images:
            raise ValueError(
                "El audio de consulta está en silencio o no contiene locución "
                "reconocible, y no se aportaron texto ni imágenes."
            )
        return mquery

    async def _embed_multimodal_query(self, mquery: MultimodalQuery) -> list[float]:
        """Invoca ``embed_query`` o delega en ``embed_text`` si un mock solo define este último."""
        vectorizer = self._get_vectorizer()
        embed_query_fn = getattr(vectorizer, "embed_query", None)
        if embed_query_fn is not None and inspect.iscoroutinefunction(embed_query_fn):
            return cast("list[float]", await embed_query_fn(mquery))
        return await vectorizer.embed_text(mquery.composed_text)

    async def search(
        self,
        query: str | MultimodalQuery = "",
        *,
        limit: int = 5,
        video_filter: str | None = None,
        rerank: bool = True,
        oversample_factor: int = 2,
        auditor: RunAuditor | None = None,
        txt_files: Sequence[str | Path | bytes] | None = None,
        image_files: Sequence[str | Path | bytes] | None = None,
        audio_files: Sequence[str | Path | bytes] | None = None,
        whisper_model: str = WHISPER_MODEL,
    ) -> list[SearchResult] | list[RerankedResult]:
        """Ejecuta la recuperación en dos etapas (Qdrant + VLM opcional) y registra auditoría."""
        cfg = SearchConfig(
            top_k=limit, rerank=rerank, reranker_model=self._reranker_model
        )
        if oversample_factor < 1:
            raise ValueError("oversample_factor debe ser >= 1.")

        mquery = await self._resolve_multimodal_query(
            query,
            txt_files=txt_files,
            image_files=image_files,
            audio_files=audio_files,
            whisper_model=whisper_model,
        )
        self.last_resolved_query = mquery

        fetch_k = (
            max(cfg.top_k, min(cfg.top_k * oversample_factor, 10))
            if cfg.rerank
            else cfg.top_k
        )
        raw_fetch_k = max(fetch_k * 3, 25)
        cache_key = self._build_query_cache_key(
            mquery,
            vector_size=self._vector_size,
            backend=self._embedding_backend,
        )
        query_vector = self._query_cache.get(cache_key)
        if query_vector is None:
            query_vector = await self._embed_multimodal_query(mquery)
            self._query_cache[cache_key] = query_vector
        raw_candidates = self.engine.search_event(
            query_vector, limit=raw_fetch_k, video_name=video_filter
        )
        candidates = _deduplicate_temporal_candidates(raw_candidates)[:fetch_k]

        final_results: list[SearchResult] | list[RerankedResult] = candidates[: cfg.top_k]
        if cfg.rerank and candidates:
            try:
                rerank_kwargs: dict[str, object] = {
                    "raw_dir": self._raw_dir,
                    "processed_dir": self._processed_dir,
                }
                if mquery.images:
                    rerank_kwargs["query_images"] = mquery.images
                reranked = await self._get_reranker().rerank(
                    mquery.composed_text,
                    candidates,
                    **rerank_kwargs,  # type: ignore[arg-type]
                )
                final_results = reranked[: cfg.top_k]
            except (VectorizationError, ValueError, RuntimeError, OSError) as exc:
                logger.warning(
                    "Fallo en reordenación multimodal (%s), conservando orden vectorial",
                    exc,
                )

        if auditor is not None:
            auditor.save_search_results(
                mquery.composed_text or "[consulta visual]",
                list(final_results),
                modality_summary=mquery.modality_summary,
            )
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

