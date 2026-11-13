"""Pruebas unitarias de las utilidades del benchmark comparativo de backends."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.rag_engine.evaluation import BenchmarkSummary, QueryEvaluation
from tests.e2e.backend_benchmark import (
    BackendBenchmarkResult,
    PeakMemorySampler,
    summarize_query_latencies_ms,
    write_backend_report,
)


def test_summarize_query_latencies_ms_returns_mean_and_median() -> None:
    """Calcula media y mediana de latencias de consulta en milisegundos."""
    evals = [
        QueryEvaluation(
            query=f"q{i}",
            video_name="python_tutorial.mp4",
            hit_rank=1,
            reciprocal_rank=1.0,
            latency_seconds=lat,
        )
        for i, lat in enumerate((0.1, 0.2, 0.6))
    ]
    summary = BenchmarkSummary(
        total_queries=3,
        top_k=5,
        rerank=False,
        recall_at_1=1.0,
        recall_at_k=1.0,
        mrr=1.0,
        mean_latency_seconds=0.3,
        evaluations=evals,
        recall_at_3=1.0,
    )

    mean_ms, median_ms = summarize_query_latencies_ms(summary)

    assert mean_ms == pytest.approx(300.0)
    assert median_ms == pytest.approx(200.0)


def test_peak_memory_sampler_records_rss_peak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Registra el pico de RSS y deja peak_mps_mb en None si torch.mps falta."""
    fake_torch = SimpleNamespace(
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with PeakMemorySampler(interval_seconds=0.05) as sampler:
        _buffer = bytearray(4 * 1024 * 1024)
        _buffer[0] = 1

    assert sampler.peak_rss_mb > 0.0
    assert sampler.peak_mps_mb is None


def test_write_backend_report_writes_one_row_per_backend(tmp_path: Path) -> None:
    """Escribe el CSV comparativo con cabecera, 3 filas y vacío en index_seconds=None."""
    results = [
        BackendBenchmarkResult(
            backend="gemini",
            dimension=3072,
            device_label="Cloud API",
            chunks=82,
            index_seconds=None,
            seconds_per_chunk=None,
            recall_at_1=0.65,
            recall_at_5=0.90,
            mrr=0.76,
            query_ms_mean=320.5,
            query_ms_median=295.0,
            peak_rss_mb=420.2,
            peak_mps_mb=None,
        ),
        BackendBenchmarkResult(
            backend="qwen3-vl-2b",
            dimension=2048,
            device_label="Local MPS",
            chunks=82,
            index_seconds=120.4,
            seconds_per_chunk=120.4 / 82,
            recall_at_1=0.55,
            recall_at_5=0.80,
            mrr=0.66,
            query_ms_mean=410.0,
            query_ms_median=390.0,
            peak_rss_mb=5120.0,
            peak_mps_mb=4600.0,
        ),
        BackendBenchmarkResult(
            backend="gemma2",
            dimension=768,
            device_label="Local MPS",
            chunks=82,
            index_seconds=45.2,
            seconds_per_chunk=45.2 / 82,
            recall_at_1=0.50,
            recall_at_5=0.75,
            mrr=0.61,
            query_ms_mean=180.0,
            query_ms_median=170.0,
            peak_rss_mb=2400.0,
            peak_mps_mb=1850.0,
        ),
    ]
    out_csv = tmp_path / "e2e_results" / "embedding_backends.csv"

    write_backend_report(results, out_csv)

    with out_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 3
    assert [r["backend"] for r in rows] == ["gemini", "qwen3-vl-2b", "gemma2"]
    assert rows[0]["index_seconds"] == ""
    assert rows[0]["seconds_per_chunk"] == ""
    assert rows[0]["peak_mps_mb"] == ""
    assert float(rows[1]["index_seconds"]) == pytest.approx(120.4)
    assert float(rows[2]["peak_mps_mb"]) == pytest.approx(1850.0)
