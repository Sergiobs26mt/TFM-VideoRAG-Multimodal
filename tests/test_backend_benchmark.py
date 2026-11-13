"""Pruebas unitarias de las utilidades del benchmark comparativo de backends."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.config import collection_name_for
from src.models import VectorizedChunk
from src.rag_engine.evaluation import (
    BenchmarkSummary,
    GroundTruthQuery,
    MultiTargetQuery,
    QueryEvaluation,
)
from src.rag_engine.vector_db import QdrantSearchEngine
from tests.e2e.backend_benchmark import (
    BackendBenchmarkResult,
    FullBackendBenchmarkResult,
    PeakMemorySampler,
    evaluate_mrl_suite_from_native,
    parse_audit_index_seconds,
    persist_mrl_collections,
    summarize_query_latencies_ms,
    write_backend_report,
    write_full_backend_report,
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


def test_parse_audit_index_seconds_extracts_elapsed_seconds(tmp_path: Path) -> None:
    log_file = tmp_path / "execution.log"
    content = (
        "2026-11-09 14:45:03,222 INFO src.utils.audit: Auditoría iniciada\n"
        "2026-11-09 14:45:03,228 INFO src.processing.pipeline: "
        "Procesando vídeo en pasada única (SSIM + fotogramas): /tmp/v.mp4\n"
        "2026-11-09 14:48:50,533 INFO src.rag_engine.service: "
        "Indexación completada: 82 chunks\n"
    )
    log_file.write_text(content, encoding="utf-8")
    assert parse_audit_index_seconds(log_file) == pytest.approx(227.305)


def test_parse_audit_index_seconds_raises_when_markers_missing(tmp_path: Path) -> None:
    log_file = tmp_path / "incomplete.log"
    log_file.write_text("2026-11-09 14:45:03,228 INFO otra cosa\n", encoding="utf-8")
    with pytest.raises(ValueError, match="marcadores"):
        parse_audit_index_seconds(log_file)


def test_persist_mrl_collections_stores_all_dims_and_both_videos(tmp_path: Path) -> None:
    db_dir = tmp_path / "vector_db"

    def _chunks(seed: float) -> list[VectorizedChunk]:
        vec = [seed + (i * 0.01) for i in range(2048)]
        return [VectorizedChunk(0, 0.0, 2.0, vec, "c0")]

    persist_mrl_collections("qwen3-vl-2b", "v1.mp4", _chunks(1.0), db_path=db_dir)
    created = persist_mrl_collections(
        "qwen3-vl-2b", "v2.mp4", _chunks(2.0), db_path=db_dir
    )
    assert created == [
        collection_name_for("qwen3-vl-2b", 2048),
        collection_name_for("qwen3-vl-2b", 1536),
        collection_name_for("qwen3-vl-2b", 768),
    ]

    for dim in (2048, 1536, 768):
        engine = QdrantSearchEngine(
            path=str(db_dir), collection_name=collection_name_for("qwen3-vl-2b", dim)
        )
        try:
            v1 = engine.get_video_chunks_with_vectors("v1.mp4")
            v2 = engine.get_video_chunks_with_vectors("v2.mp4")
            assert len(v1) == 1 and len(v1[0].embedding) == dim
            assert len(v2) == 1 and len(v2[0].embedding) == dim
            if dim < 2048:
                norm = sum(x * x for x in v1[0].embedding) ** 0.5
                assert norm == pytest.approx(1.0, abs=1e-4)
        finally:
            engine._client.close()


def test_persist_mrl_collections_preserves_gemini_native_collection(tmp_path: Path) -> None:
    db_dir = tmp_path / "vector_db"
    native_vec = [float(i == 0) for i in range(3072)]
    sentinel_vec = [float(i == 1) for i in range(3072)]
    engine = QdrantSearchEngine(path=str(db_dir), collection_name="video_chunks")
    try:
        engine.create_collection(vector_size=3072)
        engine.insert_chunks(
            [VectorizedChunk(0, 0.0, 2.0, sentinel_vec, "orig")], video_name="a.mp4"
        )
    finally:
        engine._client.close()

    persist_mrl_collections(
        "gemini",
        "a.mp4",
        [VectorizedChunk(0, 0.0, 2.0, native_vec, "new")],
        db_path=db_dir,
        overwrite_native=False,
    )

    check = QdrantSearchEngine(path=str(db_dir), collection_name="video_chunks")
    try:
        stored = check.get_video_chunks_with_vectors("a.mp4")
        assert stored[0].embedding[:2] == pytest.approx([0.0, 1.0], abs=1e-5)
        check.use_collection("video_chunks__gemini__1536")
        sub = check.get_video_chunks_with_vectors("a.mp4")
        assert len(sub[0].embedding) == 1536
        assert sub[0].embedding[0] == pytest.approx(1.0, abs=1e-5)
    finally:
        check._client.close()


def test_evaluate_mrl_suite_from_native_computes_all_dims() -> None:
    c0 = VectorizedChunk(0, 0.0, 2.0, [1.0, 0.0] + [0.0] * 766, "perro")
    c1 = VectorizedChunk(1, 2.0, 4.0, [0.0, 1.0] + [0.0] * 766, "gato")
    literal = [
        GroundTruthQuery(
            query="perro", video_name="animals.mp4", expected_start=0.0, expected_end=2.0
        )
    ]
    multi = [
        MultiTargetQuery(
            query="perro y gato",
            video_name="animals.mp4",
            expected_intervals=((0.0, 2.0), (2.0, 4.0)),
        )
    ]
    q_vecs = {
        "perro": [1.0, 0.0] + [0.0] * 766,
        "perro y gato": [0.7, 0.7] + [0.0] * 766,
    }
    rows = evaluate_mrl_suite_from_native(
        backend="gemma2",
        video_name="animals.mp4",
        native_chunks=[c0, c1],
        literal_dataset=literal,
        multi_target_dataset=multi,
        query_vectors=q_vecs,
        index_seconds=100.0,
        native_query_ms=(34.0, 32.0),
        peak_rss_mb=2000.0,
        peak_mps_mb=1500.0,
    )
    assert [r.dimension for r in rows] == [768, 512, 256]
    assert rows[0].index_seconds == pytest.approx(100.0)
    assert rows[0].query_ms_mean == pytest.approx(34.0)
    assert rows[1].index_seconds is None
    assert rows[1].query_ms_mean is None
    assert [r.storage_savings_pct for r in rows] == pytest.approx(
        [0.0, (1.0 - 512 / 768) * 100.0, (1.0 - 256 / 768) * 100.0]
    )
    assert rows[0].recall_at_1 == pytest.approx(1.0)
    assert rows[2].mt_interval_recall == pytest.approx(1.0)


def test_write_full_backend_report_writes_csv(tmp_path: Path) -> None:
    row = FullBackendBenchmarkResult(
        video_name="animals.mp4",
        backend="gemini",
        dimension=1536,
        device_label="Cloud API",
        chunks=82,
        index_seconds=None,
        seconds_per_chunk=None,
        literal_queries=46,
        recall_at_1=0.80,
        recall_at_3=0.90,
        recall_at_5=0.95,
        mrr=0.86,
        multi_target_queries=10,
        mt_interval_recall=0.75,
        mt_precision=0.50,
        query_ms_mean=None,
        query_ms_median=None,
        qdrant_search_ms_mean=1.42,
        vector_kb_per_chunk=6.0,
        total_vectors_kb=492.0,
        storage_savings_pct=50.0,
        peak_rss_mb=None,
        peak_mps_mb=None,
    )
    out = tmp_path / "full.csv"
    write_full_backend_report([row], out)
    with out.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["video_name"] == "animals.mp4"
    assert rows[0]["recall_at_3"] == "0.9000"
    assert rows[0]["mt_interval_recall"] == "0.7500"
    assert rows[0]["index_seconds"] == ""
