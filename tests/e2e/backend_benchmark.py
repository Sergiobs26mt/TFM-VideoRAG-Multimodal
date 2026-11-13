"""Utilidades de benchmark empírico comparativo entre backends de embeddings."""

from __future__ import annotations

import csv
import gc
import importlib
import re
import statistics
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Self

import numpy as np
from numpy.typing import NDArray

from src.config import (
    DB_PATH,
    EMBEDDING_BACKENDS,
    PROJECT_ROOT,
    VECTOR_SIZE,
    collection_name_for,
    default_dimension,
)
from src.embeddings.base import MultimodalVectorizer
from src.embeddings.factory import (
    create_vectorizer,
    device_label_for,
    unload_active_vectorizer,
)
from src.embeddings.mrl import truncate_and_normalize_mrl
from src.models import MultimodalQuery, VectorizedChunk, VideoChunk
from src.rag_engine.evaluation import (
    BenchmarkSummary,
    GroundTruthQuery,
    MultiTargetEvaluation,
    MultiTargetQuery,
    QueryEvaluation,
    compute_benchmark_summary,
    evaluate_multi_target_results,
    evaluate_search_results,
    load_ground_truth_csv,
    run_benchmark,
    run_multi_target_benchmark,
)
from src.rag_engine.service import VideoRAGService, _deduplicate_temporal_candidates
from src.rag_engine.vector_db import QdrantSearchEngine
from tests.e2e.harness import (
    GROUND_TRUTH_CSV,
    MULTI_TARGET_QUERIES,
    MULTI_TARGET_TOP_K,
    PYTHON_MULTI_TARGET_QUERIES,
    PYTHON_VIDEO_NAME,
    RAW_DIR,
    TOP_K,
    VIDEO_NAME,
)

BYTES_PER_MB = 1024.0 * 1024.0
_LOG_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\s")


@dataclass(frozen=True)
class VideoBenchmarkProfile:
    """Configuración de segmentación, consultas multi-objetivo y auditoría por vídeo."""

    video_name: str
    threshold: float
    max_shot_duration: float
    multi_target_queries: tuple[MultiTargetQuery, ...]
    audit_log: Path


VIDEO_PROFILES: dict[str, VideoBenchmarkProfile] = {
    PYTHON_VIDEO_NAME: VideoBenchmarkProfile(
        video_name=PYTHON_VIDEO_NAME,
        threshold=0.6,
        max_shot_duration=5.0,
        multi_target_queries=tuple(PYTHON_MULTI_TARGET_QUERIES),
        audit_log=PROJECT_ROOT / "audit" / "run_20261109_144503" / "execution.log",
    ),
    VIDEO_NAME: VideoBenchmarkProfile(
        video_name=VIDEO_NAME,
        threshold=0.5,
        max_shot_duration=8.0,
        multi_target_queries=tuple(MULTI_TARGET_QUERIES),
        audit_log=PROJECT_ROOT / "audit" / "run_20261109_145448" / "execution.log",
    ),
}


@dataclass(frozen=True)
class BackendBenchmarkResult:
    """Métricas de calidad, velocidad y memoria para un backend de embeddings."""

    backend: str
    dimension: int
    device_label: str
    chunks: int
    index_seconds: float | None
    seconds_per_chunk: float | None
    recall_at_1: float
    recall_at_5: float
    mrr: float
    query_ms_mean: float
    query_ms_median: float
    peak_rss_mb: float
    peak_mps_mb: float | None


def _read_rss_mb() -> float:
    psutil = importlib.import_module("psutil")
    rss_bytes = float(psutil.Process().memory_info().rss)
    return rss_bytes / BYTES_PER_MB


def _read_mps_mb() -> float | None:
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return None
    mps = getattr(torch, "mps", None)
    backends = getattr(torch, "backends", None)
    mps_backend = getattr(backends, "mps", None)
    if (
        mps is None
        or mps_backend is None
        or not callable(getattr(mps_backend, "is_available", None))
        or not mps_backend.is_available()
    ):
        return None
    driver_memory = getattr(mps, "driver_allocated_memory", None)
    if not callable(driver_memory):
        return None
    return float(driver_memory()) / BYTES_PER_MB


class PeakMemorySampler:
    """Muestreador en segundo plano del pico de RSS y memoria del driver MPS."""

    def __init__(self, interval_seconds: float = 0.2) -> None:
        self._interval = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_rss_mb: float = 0.0
        self.peak_mps_mb: float | None = None

    def _sample_once(self) -> None:
        self.peak_rss_mb = max(self.peak_rss_mb, _read_rss_mb())
        mps_mb = _read_mps_mb()
        if mps_mb is not None:
            self.peak_mps_mb = (
                mps_mb if self.peak_mps_mb is None else max(self.peak_mps_mb, mps_mb)
            )

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            self._sample_once()

    def __enter__(self) -> Self:
        self._stop.clear()
        self._sample_once()
        self._thread = threading.Thread(
            target=self._loop, name="memory-sampler", daemon=True
        )
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._sample_once()


def summarize_query_latencies_ms(summary: BenchmarkSummary) -> tuple[float, float]:
    """Devuelve media y mediana de latencia por consulta en milisegundos."""
    if not summary.evaluations:
        return 0.0, 0.0
    latencies_ms = [item.latency_seconds * 1000.0 for item in summary.evaluations]
    return statistics.mean(latencies_ms), statistics.median(latencies_ms)


def _flush_mps_cache() -> None:
    gc.collect()
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return
    backends = getattr(torch, "backends", None)
    mps_backend = getattr(backends, "mps", None)
    mps = getattr(torch, "mps", None)
    if (
        mps is not None
        and mps_backend is not None
        and callable(getattr(mps_backend, "is_available", None))
        and mps_backend.is_available()
        and callable(getattr(mps, "empty_cache", None))
    ):
        mps.empty_cache()


async def execute_backend_benchmark(
    backend: str,
    dimension: int,
) -> BackendBenchmarkResult:
    """Indexa o reutiliza python_tutorial.mp4 en memoria y evalúa recuperación Etapa 1."""
    unload_active_vectorizer()
    _flush_mps_cache()
    dataset = load_ground_truth_csv(GROUND_TRUTH_CSV, video_name=PYTHON_VIDEO_NAME)
    collection = collection_name_for(backend, dimension)
    engine = QdrantSearchEngine(path=None, collection_name=collection)
    service: VideoRAGService | None = None

    try:
        reused_index = False
        chunks_count = 0
        index_seconds: float | None = None
        seconds_per_chunk: float | None = None

        if backend == "gemini" and dimension == VECTOR_SIZE:
            persistent = QdrantSearchEngine(path=DB_PATH, collection_name=collection)
            try:
                existing = persistent.get_video_chunks_with_vectors(PYTHON_VIDEO_NAME)
            finally:
                persistent._client.close()
            if existing and len(existing[0].embedding) == dimension:
                engine.create_collection(vector_size=dimension)
                engine.insert_chunks(existing, video_name=PYTHON_VIDEO_NAME)
                chunks_count = len(existing)
                reused_index = True
                index_seconds, seconds_per_chunk = _historical_index_timing(
                    PYTHON_VIDEO_NAME, chunks_count
                )

        service = VideoRAGService(
            engine=engine,
            raw_dir=RAW_DIR,
            vector_size=dimension,
            embedding_backend=backend,
        )

        with PeakMemorySampler() as sampler:
            if not reused_index:
                started = time.perf_counter()
                chunks_count = await service.index_video(
                    PYTHON_VIDEO_NAME,
                    threshold=0.6,
                    max_shot_duration=5.0,
                    output_dimensionality=dimension,
                    target_width=512,
                    enable_audio=True,
                    embedding_backend=backend,
                )
                index_seconds = time.perf_counter() - started
                seconds_per_chunk = (
                    index_seconds / chunks_count if chunks_count > 0 else None
                )

            summary = await run_benchmark(
                service,
                dataset,
                top_k=TOP_K,
                rerank=False,
            )

        query_ms_mean, query_ms_median = summarize_query_latencies_ms(summary)
        is_local = EMBEDDING_BACKENDS[backend].is_local
        return BackendBenchmarkResult(
            backend=backend,
            dimension=dimension,
            device_label=device_label_for(backend),
            chunks=chunks_count,
            index_seconds=index_seconds,
            seconds_per_chunk=seconds_per_chunk,
            recall_at_1=summary.recall_at_1,
            recall_at_5=summary.recall_at_k,
            mrr=summary.mrr,
            query_ms_mean=query_ms_mean,
            query_ms_median=query_ms_median,
            peak_rss_mb=sampler.peak_rss_mb,
            peak_mps_mb=sampler.peak_mps_mb if is_local else None,
        )
    finally:
        engine._client.close()
        if service is not None:
            service._vectorizer = None
            service = None
        unload_active_vectorizer()
        _flush_mps_cache()


def _fmt_optional(value: float | None, precision: int) -> str:
    return "" if value is None else f"{value:.{precision}f}"


def write_backend_report(
    results: Sequence[BackendBenchmarkResult],
    path: Path,
) -> None:
    """Guarda en CSV una fila por backend con todas las métricas del benchmark."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "backend",
                "dimension",
                "device_label",
                "chunks",
                "index_seconds",
                "seconds_per_chunk",
                "recall_at_1",
                "recall_at_5",
                "mrr",
                "query_ms_mean",
                "query_ms_median",
                "peak_rss_mb",
                "peak_mps_mb",
            ]
        )
        for row in results:
            writer.writerow(
                [
                    row.backend,
                    row.dimension,
                    row.device_label,
                    row.chunks,
                    _fmt_optional(row.index_seconds, 2),
                    _fmt_optional(row.seconds_per_chunk, 3),
                    f"{row.recall_at_1:.4f}",
                    f"{row.recall_at_5:.4f}",
                    f"{row.mrr:.4f}",
                    f"{row.query_ms_mean:.2f}",
                    f"{row.query_ms_median:.2f}",
                    f"{row.peak_rss_mb:.1f}",
                    _fmt_optional(row.peak_mps_mb, 1),
                ]
            )


@dataclass(frozen=True)
class FullBackendBenchmarkResult:
    """Métricas completas (literal + multi-objetivo + MRL) por vídeo, backend y dimensión."""

    video_name: str
    backend: str
    dimension: int
    device_label: str
    chunks: int
    index_seconds: float | None
    seconds_per_chunk: float | None
    literal_queries: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    multi_target_queries: int
    mt_interval_recall: float
    mt_precision: float
    query_ms_mean: float | None
    query_ms_median: float | None
    qdrant_search_ms_mean: float
    vector_kb_per_chunk: float
    total_vectors_kb: float
    storage_savings_pct: float
    peak_rss_mb: float | None
    peak_mps_mb: float | None


def parse_audit_index_seconds(log_path: Path) -> float:
    """Extrae la duración total de ``index_video`` desde un fichero ``execution.log``."""
    start_dt: datetime | None = None
    end_dt: datetime | None = None
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = _LOG_TS_RE.match(line)
            if match is None:
                continue
            if start_dt is None and "Procesando vídeo" in line:
                start_dt = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S,%f")
            elif start_dt is not None and "Indexación completada" in line:
                end_dt = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S,%f")
                break
    if start_dt is None or end_dt is None:
        raise ValueError(
            f"No se encontraron marcadores de indexación completos en {log_path}."
        )
    return (end_dt - start_dt).total_seconds()


def _historical_index_timing(
    video_name: str, chunks_count: int
) -> tuple[float | None, float | None]:
    profile = VIDEO_PROFILES.get(video_name)
    if profile is None or not profile.audit_log.is_file() or chunks_count <= 0:
        return None, None
    seconds = parse_audit_index_seconds(profile.audit_log)
    return seconds, seconds / chunks_count


def persist_mrl_collections(
    backend: str,
    video_name: str,
    native_chunks: Sequence[VectorizedChunk],
    *,
    db_path: str | Path = DB_PATH,
    overwrite_native: bool = True,
) -> list[str]:
    """Guarda los chunks en la colección nativa y genera sus colecciones MRL en Qdrant."""
    dims = EMBEDDING_BACKENDS[backend].dimensions
    native_dim = default_dimension(backend)
    engine = QdrantSearchEngine(
        path=str(db_path),
        collection_name=collection_name_for(backend, dims[0]),
    )
    updated: list[str] = []
    try:
        for dim in dims:
            collection = collection_name_for(backend, dim)
            engine.use_collection(collection)
            engine.create_collection(vector_size=dim, recreate=False)
            if (
                dim == native_dim
                and not overwrite_native
                and engine.get_video_chunks_with_vectors(video_name)
            ):
                updated.append(collection)
                continue
            dim_chunks = (
                list(native_chunks)
                if dim == native_dim
                else [
                    VectorizedChunk(
                        index=c.index,
                        start_time=c.start_time,
                        end_time=c.end_time,
                        embedding=truncate_and_normalize_mrl(c.embedding, dim),
                        transcript=c.transcript,
                    )
                    for c in native_chunks
                ]
            )
            engine.insert_chunks(dim_chunks, video_name=video_name, overwrite_video=True)
            updated.append(collection)
    finally:
        engine._client.close()
    return updated


class _RecordingVectorizer:
    """Envoltorio transparente que captura vectores de consulta ya inferidos."""

    def __init__(self, inner: MultimodalVectorizer) -> None:
        self._inner = inner
        self.query_vectors: dict[str, list[float]] = {}

    @property
    def backend(self) -> str:
        return self._inner.backend

    @property
    def device_label(self) -> str:
        return self._inner.device_label

    async def vectorize_chunk(
        self, chunk: VideoChunk, key_frames: list[NDArray[np.uint8]]
    ) -> VectorizedChunk:
        return await self._inner.vectorize_chunk(chunk, key_frames)

    async def embed_query(self, query: str | MultimodalQuery) -> list[float]:
        vec = await self._inner.embed_query(query)
        if isinstance(query, MultimodalQuery):
            if query.text is not None:
                self.query_vectors[query.text] = vec
            if query.composed_text:
                self.query_vectors[query.composed_text] = vec
        else:
            self.query_vectors[query] = vec
        return vec

    async def embed_text(self, text: str) -> list[float]:
        vec = await self._inner.embed_text(text)
        self.query_vectors[text] = vec
        return vec


def evaluate_mrl_suite_from_native(
    *,
    backend: str,
    video_name: str,
    native_chunks: Sequence[VectorizedChunk],
    literal_dataset: Sequence[GroundTruthQuery],
    multi_target_dataset: Sequence[MultiTargetQuery],
    query_vectors: Mapping[str, Sequence[float]],
    index_seconds: float | None,
    native_query_ms: tuple[float, float],
    peak_rss_mb: float,
    peak_mps_mb: float | None,
    device_label: str | None = None,
) -> list[FullBackendBenchmarkResult]:
    """Evalúa todas las dimensiones MRL del backend truncando en memoria los vectores nativos."""
    dims = EMBEDDING_BACKENDS[backend].dimensions
    native_dim = default_dimension(backend)
    is_local = EMBEDDING_BACKENDS[backend].is_local
    resolved_device = device_label if device_label is not None else device_label_for(backend)
    chunks_count = len(native_chunks)
    lit_raw_k = max(TOP_K * 3, 25)
    mt_raw_k = max(MULTI_TARGET_TOP_K * 3, 25)
    rows: list[FullBackendBenchmarkResult] = []

    for dim in dims:
        is_native = dim == native_dim
        dim_chunks = (
            list(native_chunks)
            if is_native
            else [
                VectorizedChunk(
                    index=c.index,
                    start_time=c.start_time,
                    end_time=c.end_time,
                    embedding=truncate_and_normalize_mrl(c.embedding, dim),
                    transcript=c.transcript,
                )
                for c in native_chunks
            ]
        )
        engine = QdrantSearchEngine(path=None)
        try:
            engine.create_collection(vector_size=dim, recreate=True)
            engine.insert_chunks(dim_chunks, video_name=video_name)

            lit_evals: list[QueryEvaluation] = []
            for lit_item in literal_dataset:
                raw_q = query_vectors[lit_item.query]
                q_vec = list(raw_q) if is_native else truncate_and_normalize_mrl(raw_q, dim)
                t0 = time.perf_counter()
                raw_cands = engine.search_event(
                    q_vec, limit=lit_raw_k, video_name=video_name
                )
                deduped = _deduplicate_temporal_candidates(raw_cands)[:TOP_K]
                lit_evals.append(
                    evaluate_search_results(lit_item, deduped, time.perf_counter() - t0)
                )

            mt_evals: list[MultiTargetEvaluation] = []
            for mt_item in multi_target_dataset:
                raw_q = query_vectors[mt_item.query]
                q_vec = list(raw_q) if is_native else truncate_and_normalize_mrl(raw_q, dim)
                t0 = time.perf_counter()
                raw_cands = engine.search_event(
                    q_vec, limit=mt_raw_k, video_name=video_name
                )
                deduped = _deduplicate_temporal_candidates(raw_cands)[
                    :MULTI_TARGET_TOP_K
                ]
                mt_evals.append(
                    evaluate_multi_target_results(
                        mt_item, deduped, time.perf_counter() - t0
                    )
                )
        finally:
            engine._client.close()

        summary = compute_benchmark_summary(lit_evals, top_k=TOP_K, rerank=False)
        n_mt = len(mt_evals) or 1
        mt_recall = sum(e.interval_recall for e in mt_evals) / n_mt
        mt_prec = sum(e.precision for e in mt_evals) / n_mt
        kb_per_vec = (dim * 4) / 1024.0
        rows.append(
            FullBackendBenchmarkResult(
                video_name=video_name,
                backend=backend,
                dimension=dim,
                device_label=resolved_device,
                chunks=chunks_count,
                index_seconds=index_seconds if is_native else None,
                seconds_per_chunk=(
                    index_seconds / chunks_count
                    if is_native and index_seconds is not None and chunks_count > 0
                    else None
                ),
                literal_queries=summary.total_queries,
                recall_at_1=summary.recall_at_1,
                recall_at_3=summary.recall_at_3,
                recall_at_5=summary.recall_at_k,
                mrr=summary.mrr,
                multi_target_queries=len(mt_evals),
                mt_interval_recall=mt_recall,
                mt_precision=mt_prec,
                query_ms_mean=native_query_ms[0] if is_native else None,
                query_ms_median=native_query_ms[1] if is_native else None,
                qdrant_search_ms_mean=summary.mean_latency_seconds * 1000.0,
                vector_kb_per_chunk=kb_per_vec,
                total_vectors_kb=chunks_count * kb_per_vec,
                storage_savings_pct=(1.0 - (dim / float(native_dim))) * 100.0,
                peak_rss_mb=peak_rss_mb if is_native else None,
                peak_mps_mb=(peak_mps_mb if (is_native and is_local) else None),
            )
        )
    return rows


async def execute_full_backend_benchmark(
    backend: str,
    video_name: str,
    *,
    persist: bool = True,
    db_path: str | Path = DB_PATH,
) -> list[FullBackendBenchmarkResult]:
    """Ejecuta el benchmark completo (literal, multi-objetivo, MRL y persistencia Qdrant)."""
    if video_name not in VIDEO_PROFILES:
        raise ValueError(f"Vídeo sin perfil de benchmark: {video_name!r}.")
    profile = VIDEO_PROFILES[video_name]
    native_dim = default_dimension(backend)
    unload_active_vectorizer()
    _flush_mps_cache()

    literal_dataset = load_ground_truth_csv(GROUND_TRUTH_CSV, video_name=video_name)
    mt_dataset = list(profile.multi_target_queries)
    collection = collection_name_for(backend, native_dim)
    engine = QdrantSearchEngine(path=None, collection_name=collection)
    recorder: _RecordingVectorizer | None = None
    index_service: VideoRAGService | None = None
    search_service: VideoRAGService | None = None

    try:
        reused_index = False
        chunks_count = 0
        index_seconds: float | None = None

        if backend == "gemini":
            persistent = QdrantSearchEngine(path=str(db_path), collection_name=collection)
            try:
                existing = persistent.get_video_chunks_with_vectors(video_name)
            finally:
                persistent._client.close()
            if existing and len(existing[0].embedding) == native_dim:
                engine.create_collection(vector_size=native_dim)
                engine.insert_chunks(existing, video_name=video_name)
                chunks_count = len(existing)
                reused_index = True
                index_seconds, _ = _historical_index_timing(video_name, chunks_count)

        with PeakMemorySampler() as sampler:
            if not reused_index:
                index_service = VideoRAGService(
                    engine=engine,
                    raw_dir=RAW_DIR,
                    vector_size=native_dim,
                    embedding_backend=backend,
                )
                started = time.perf_counter()
                chunks_count = await index_service.index_video(
                    video_name,
                    threshold=profile.threshold,
                    max_shot_duration=profile.max_shot_duration,
                    output_dimensionality=native_dim,
                    target_width=512,
                    enable_audio=True,
                    embedding_backend=backend,
                )
                index_seconds = time.perf_counter() - started

            recorder = _RecordingVectorizer(create_vectorizer(backend, native_dim))
            search_service = VideoRAGService(
                engine=engine,
                raw_dir=RAW_DIR,
                vector_size=native_dim,
                embedding_backend=backend,
                vectorizer=recorder,
            )
            lit_summary = await run_benchmark(
                search_service,
                literal_dataset,
                top_k=TOP_K,
                rerank=False,
            )
            await run_multi_target_benchmark(
                search_service,
                mt_dataset,
                top_k=MULTI_TARGET_TOP_K,
                rerank=False,
            )

        native_chunks = engine.get_video_chunks_with_vectors(video_name)
        if persist:
            persist_mrl_collections(
                backend,
                video_name,
                native_chunks,
                db_path=db_path,
                overwrite_native=(backend != "gemini"),
            )

        native_query_ms = summarize_query_latencies_ms(lit_summary)
        return evaluate_mrl_suite_from_native(
            backend=backend,
            video_name=video_name,
            native_chunks=native_chunks,
            literal_dataset=literal_dataset,
            multi_target_dataset=mt_dataset,
            query_vectors=recorder.query_vectors,
            index_seconds=index_seconds,
            native_query_ms=native_query_ms,
            peak_rss_mb=sampler.peak_rss_mb,
            peak_mps_mb=sampler.peak_mps_mb,
            device_label=device_label_for(backend),
        )
    finally:
        engine._client.close()
        if index_service is not None:
            index_service._vectorizer = None
            index_service = None
        if search_service is not None:
            search_service._vectorizer = None
            search_service = None
        recorder = None
        unload_active_vectorizer()
        _flush_mps_cache()


def write_full_backend_report(
    results: Sequence[FullBackendBenchmarkResult],
    path: Path,
) -> None:
    """Guarda en CSV las métricas comparativas por vídeo, backend y dimensión MRL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "video_name",
                "backend",
                "dimension",
                "device_label",
                "chunks",
                "index_seconds",
                "seconds_per_chunk",
                "literal_queries",
                "recall_at_1",
                "recall_at_3",
                "recall_at_5",
                "mrr",
                "multi_target_queries",
                "mt_interval_recall",
                "mt_precision",
                "query_ms_mean",
                "query_ms_median",
                "qdrant_search_ms_mean",
                "vector_kb_per_chunk",
                "total_vectors_kb",
                "storage_savings_pct",
                "peak_rss_mb",
                "peak_mps_mb",
            ]
        )
        for row in results:
            writer.writerow(
                [
                    row.video_name,
                    row.backend,
                    row.dimension,
                    row.device_label,
                    row.chunks,
                    _fmt_optional(row.index_seconds, 2),
                    _fmt_optional(row.seconds_per_chunk, 3),
                    row.literal_queries,
                    f"{row.recall_at_1:.4f}",
                    f"{row.recall_at_3:.4f}",
                    f"{row.recall_at_5:.4f}",
                    f"{row.mrr:.4f}",
                    row.multi_target_queries,
                    f"{row.mt_interval_recall:.4f}",
                    f"{row.mt_precision:.4f}",
                    _fmt_optional(row.query_ms_mean, 2),
                    _fmt_optional(row.query_ms_median, 2),
                    f"{row.qdrant_search_ms_mean:.4f}",
                    f"{row.vector_kb_per_chunk:.2f}",
                    f"{row.total_vectors_kb:.2f}",
                    f"{row.storage_savings_pct:.1f}",
                    _fmt_optional(row.peak_rss_mb, 1),
                    _fmt_optional(row.peak_mps_mb, 1),
                ]
            )
