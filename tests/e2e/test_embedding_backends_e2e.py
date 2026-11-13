"""Benchmark E2E comparativo de backends (ambos vídeos, multi-objetivo y MRL)."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterator
from datetime import date
from pathlib import Path

import pytest
from dotenv import load_dotenv

from src.config import PROJECT_ROOT, default_dimension
from tests.e2e.backend_benchmark import (
    BackendBenchmarkResult,
    FullBackendBenchmarkResult,
    execute_full_backend_benchmark,
    write_backend_report,
    write_full_backend_report,
)
from tests.e2e.harness import PYTHON_VIDEO_NAME, RAW_DIR, VIDEO_NAME

load_dotenv()

pytestmark = [
    pytest.mark.e2e_local,
    pytest.mark.skipif(
        not (
            (RAW_DIR / PYTHON_VIDEO_NAME).is_file()
            and (RAW_DIR / VIDEO_NAME).is_file()
        ),
        reason=f"Faltan vídeos E2E en {RAW_DIR}.",
    ),
]


def _summary_report_path() -> Path:
    stamp = date.today().strftime("%Y%m%d")
    return PROJECT_ROOT / "data" / "e2e_results" / f"embedding_backends_{stamp}.csv"


def _full_report_path() -> Path:
    stamp = date.today().strftime("%Y%m%d")
    return (
        PROJECT_ROOT / "data" / "e2e_results" / f"embedding_backends_full_{stamp}.csv"
    )


def _to_summary_rows(
    full_rows: list[FullBackendBenchmarkResult],
) -> list[BackendBenchmarkResult]:
    out: list[BackendBenchmarkResult] = []
    for row in full_rows:
        if row.video_name != PYTHON_VIDEO_NAME:
            continue
        if row.dimension != default_dimension(row.backend):
            continue
        assert row.query_ms_mean is not None
        assert row.query_ms_median is not None
        assert row.peak_rss_mb is not None
        out.append(
            BackendBenchmarkResult(
                backend=row.backend,
                dimension=row.dimension,
                device_label=row.device_label,
                chunks=row.chunks,
                index_seconds=row.index_seconds,
                seconds_per_chunk=row.seconds_per_chunk,
                recall_at_1=row.recall_at_1,
                recall_at_5=row.recall_at_5,
                mrr=row.mrr,
                query_ms_mean=row.query_ms_mean,
                query_ms_median=row.query_ms_median,
                peak_rss_mb=row.peak_rss_mb,
                peak_mps_mb=row.peak_mps_mb,
            )
        )
    return out


def _flush_reports(collected_results: list[FullBackendBenchmarkResult]) -> None:
    if not collected_results:
        return
    write_full_backend_report(collected_results, _full_report_path())
    summary_rows = _to_summary_rows(collected_results)
    if summary_rows:
        write_backend_report(summary_rows, _summary_report_path())


@pytest.fixture(scope="module")
def collected_results(
    monkeypatch_module: None,
) -> Iterator[list[FullBackendBenchmarkResult]]:
    """Acumula métricas completas por vídeo/backend/dimensión y guarda los CSV."""
    results: list[FullBackendBenchmarkResult] = []
    yield results
    _flush_reports(results)


@pytest.fixture(scope="module")
def monkeypatch_module() -> Iterator[None]:
    """Fija el directorio de trabajo en PROJECT_ROOT para las rutas relativas."""
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(PROJECT_ROOT)
        yield


def _run_and_record(
    backend: str,
    video_name: str,
    collected_results: list[FullBackendBenchmarkResult],
) -> None:
    rows = asyncio.run(execute_full_backend_benchmark(backend, video_name, persist=True))
    collected_results.extend(rows)
    _flush_reports(collected_results)
    assert len(rows) == 3
    assert rows[0].chunks > 0
    assert rows[0].recall_at_5 > 0.0
    assert rows[0].mt_interval_recall > 0.0


@pytest.mark.parametrize("video_name", [PYTHON_VIDEO_NAME, VIDEO_NAME])
def test_benchmark_gemini_backend(
    video_name: str,
    collected_results: list[FullBackendBenchmarkResult],
) -> None:
    """Evalúa Gemini Embedding 2 (reutilizando video_chunks 3072-d) en todas sus dims MRL."""
    if not os.getenv("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY no está definida.")
    _run_and_record("gemini", video_name, collected_results)


@pytest.mark.parametrize("video_name", [PYTHON_VIDEO_NAME, VIDEO_NAME])
def test_benchmark_qwen3_vl_2b_backend(
    video_name: str,
    collected_results: list[FullBackendBenchmarkResult],
) -> None:
    """Indexa, persiste y evalúa Qwen3-VL-Embedding-2B en todas sus dims MRL."""
    _run_and_record("qwen3-vl-2b", video_name, collected_results)


@pytest.mark.parametrize("video_name", [PYTHON_VIDEO_NAME, VIDEO_NAME])
def test_benchmark_gemma2_backend(
    video_name: str,
    collected_results: list[FullBackendBenchmarkResult],
) -> None:
    """Indexa, persiste y evalúa EmbeddingGemma 2 en todas sus dims MRL."""
    _run_and_record("gemma2", video_name, collected_results)
