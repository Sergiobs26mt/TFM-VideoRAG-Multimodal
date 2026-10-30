"""Pruebas unitarias de evaluación empírica y benchmark (Recall@K y MRR)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.models import SearchResult, VectorizedChunk
from src.rag_engine.evaluation import (
    GroundTruthQuery,
    compute_benchmark_summary,
    evaluate_search_results,
    load_ground_truth,
    run_benchmark,
)
from src.rag_engine.service import VideoRAGService
from src.rag_engine.vector_db import QdrantSearchEngine
from src.utils.audit import RunAuditor


def test_evaluate_search_results_matches_video_and_temporal_overlap() -> None:
    target = GroundTruthQuery(
        query="camaleón",
        video_name="animals.mp4",
        expected_start=24.0,
        expected_end=30.0,
    )
    results = [
        SearchResult(0.9, "other.mp4", 0, 24.0, 30.0),
        SearchResult(0.8, "animals.mp4", 1, 0.0, 10.0),
        SearchResult(0.7, "animals.mp4", 2, 22.0, 26.0),
    ]
    ev = evaluate_search_results(target, results, latency_seconds=0.12)
    assert ev.hit_rank == 3
    assert ev.reciprocal_rank == pytest.approx(1.0 / 3.0)
    assert ev.latency_seconds == pytest.approx(0.12)

    miss = evaluate_search_results(
        target, [SearchResult(0.8, "animals.mp4", 1, 0.0, 10.0)]
    )
    assert miss.hit_rank is None
    assert miss.reciprocal_rank == 0.0


def test_compute_benchmark_summary_calculates_recall_and_mrr() -> None:
    t1 = GroundTruthQuery("q1", "v.mp4", 0.0, 5.0)
    t2 = GroundTruthQuery("q2", "v.mp4", 10.0, 15.0)
    t3 = GroundTruthQuery("q3", "v.mp4", 20.0, 25.0)

    ev1 = evaluate_search_results(
        t1, [SearchResult(0.9, "v.mp4", 0, 1.0, 4.0)], latency_seconds=0.1
    )
    ev2 = evaluate_search_results(
        t2,
        [
            SearchResult(0.9, "v.mp4", 0, 0.0, 2.0),
            SearchResult(0.8, "v.mp4", 1, 11.0, 14.0),
        ],
        latency_seconds=0.2,
    )
    ev3 = evaluate_search_results(
        t3, [SearchResult(0.9, "v.mp4", 0, 0.0, 2.0)], latency_seconds=0.3
    )

    summary = compute_benchmark_summary([ev1, ev2, ev3], top_k=3, rerank=False)
    assert summary.total_queries == 3
    assert summary.recall_at_1 == pytest.approx(1.0 / 3.0)
    assert summary.recall_at_k == pytest.approx(2.0 / 3.0)
    assert summary.mrr == pytest.approx(0.5)
    assert summary.mean_latency_seconds == pytest.approx(0.2)


def test_load_ground_truth_and_run_benchmark_persists_summary(tmp_path: Path) -> None:
    gt_file = tmp_path / "gt.json"
    gt_file.write_text(
        json.dumps(
            [
                {
                    "query": "bucles for",
                    "video_name": "python.mp4",
                    "expected_start": 10.0,
                    "expected_end": 20.0,
                }
            ]
        ),
        encoding="utf-8",
    )
    dataset = load_ground_truth(gt_file)
    assert len(dataset) == 1

    engine = QdrantSearchEngine(path=None)
    engine.create_collection(vector_size=3)
    engine.insert_chunks(
        [
            VectorizedChunk(0, 0.0, 5.0, [1.0, 0.0, 0.0]),
            VectorizedChunk(1, 12.0, 18.0, [0.0, 1.0, 0.0]),
        ],
        video_name="python.mp4",
    )

    fake_vectorizer = MagicMock()
    fake_vectorizer.embed_text = AsyncMock(return_value=[0.0, 1.0, 0.0])

    service = VideoRAGService(
        engine=engine,
        vectorizer=fake_vectorizer,
        vector_size=3,
    )

    with RunAuditor(tmp_path / "audit") as auditor:
        summary = asyncio.run(
            run_benchmark(service, dataset, top_k=2, rerank=False, auditor=auditor)
        )
        run_dir = auditor.run_dir

    assert summary.recall_at_1 == 1.0
    assert summary.mrr == 1.0
    saved = json.loads(
        (run_dir / "benchmark_summary.json").read_text(encoding="utf-8")
    )
    assert saved["recall_at_1"] == 1.0
    assert saved["mrr"] == 1.0

