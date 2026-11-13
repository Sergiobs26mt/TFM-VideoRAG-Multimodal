"""Utilidades de benchmark empírico comparativo entre backends de embeddings."""

from __future__ import annotations

import csv
import gc
import importlib
import statistics
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self

from src.config import DB_PATH, EMBEDDING_BACKENDS, VECTOR_SIZE, collection_name_for
from src.embeddings.factory import device_label_for, unload_active_vectorizer
from src.rag_engine.evaluation import (
    BenchmarkSummary,
    load_ground_truth_csv,
    run_benchmark,
)
from src.rag_engine.service import VideoRAGService
from src.rag_engine.vector_db import QdrantSearchEngine
from tests.e2e.harness import GROUND_TRUTH_CSV, PYTHON_VIDEO_NAME, RAW_DIR, TOP_K

BYTES_PER_MB = 1024.0 * 1024.0


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
