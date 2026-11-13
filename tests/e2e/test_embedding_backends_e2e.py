"""Benchmark E2E comparativo de backends de embeddings sobre python_tutorial.mp4."""

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
    execute_backend_benchmark,
    write_backend_report,
)
from tests.e2e.harness import PYTHON_VIDEO_NAME, RAW_DIR

load_dotenv()

pytestmark = [
    pytest.mark.e2e_local,
    pytest.mark.skipif(
        not (RAW_DIR / PYTHON_VIDEO_NAME).is_file(),
        reason=f"Falta {RAW_DIR / PYTHON_VIDEO_NAME}.",
    ),
]


def _report_path() -> Path:
    stamp = date.today().strftime("%Y%m%d")
    return PROJECT_ROOT / "data" / "e2e_results" / f"embedding_backends_{stamp}.csv"


@pytest.fixture(scope="module")
def collected_results(
    monkeypatch_module: None,
) -> Iterator[list[BackendBenchmarkResult]]:
    """Acumula métricas de cada backend y guarda el CSV comparativo al finalizar."""
    results: list[BackendBenchmarkResult] = []
    yield results
    if results:
        write_backend_report(results, _report_path())


@pytest.fixture(scope="module")
def monkeypatch_module() -> Iterator[None]:
    """Fija el directorio de trabajo en PROJECT_ROOT para las rutas relativas."""
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(PROJECT_ROOT)
        yield


def _run_and_record(
    backend: str,
    collected_results: list[BackendBenchmarkResult],
) -> None:
    result = asyncio.run(
        execute_backend_benchmark(backend, default_dimension(backend))
    )
    collected_results.append(result)
    write_backend_report(collected_results, _report_path())
    assert result.chunks > 0
    assert result.recall_at_5 > 0.0


def test_benchmark_gemini_backend(
    collected_results: list[BackendBenchmarkResult],
) -> None:
    """Evalúa Gemini Embedding 2 (reutilizando video_chunks 3072-d si ya está indexado)."""
    if not os.getenv("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY no está definida.")
    _run_and_record("gemini", collected_results)


def test_benchmark_qwen3_vl_2b_backend(
    collected_results: list[BackendBenchmarkResult],
) -> None:
    """Indexa y evalúa Qwen3-VL-Embedding-2B en dimensión por defecto (2048)."""
    _run_and_record("qwen3-vl-2b", collected_results)


def test_benchmark_gemma2_backend(
    collected_results: list[BackendBenchmarkResult],
) -> None:
    """Indexa y evalúa EmbeddingGemma 2 en dimensión por defecto (768)."""
    _run_and_record("gemma2", collected_results)
