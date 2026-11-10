"""Pruebas unitarias de evaluación empírica y benchmark (Recall@K y MRR)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.models import RerankedResult, SearchResult, VectorizedChunk
from src.rag_engine.evaluation import (
    GroundTruthQuery,
    MatryoshkaDimensionMetrics,
    MatryoshkaMultiTargetMetrics,
    MatryoshkaNegativeMetrics,
    MultiTargetQuery,
    compute_benchmark_summary,
    evaluate_matryoshka_dimensions,
    evaluate_matryoshka_multi_target,
    evaluate_matryoshka_negative,
    evaluate_multi_target_results,
    evaluate_negative_results,
    evaluate_search_results,
    load_ground_truth,
    load_ground_truth_csv,
    run_benchmark,
    run_multi_target_benchmark,
    run_negative_benchmark,
    write_matryoshka_ablation_csv,
    write_matryoshka_multi_target_csv,
    write_matryoshka_negative_csv,
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
    assert summary.recall_at_3 == pytest.approx(2.0 / 3.0)
    assert summary.recall_at_k == pytest.approx(2.0 / 3.0)
    assert summary.mrr == pytest.approx(0.5)
    assert summary.mean_latency_seconds == pytest.approx(0.2)


def test_load_ground_truth_csv_maps_dataset_columns(tmp_path: Path) -> None:
    csv_file = tmp_path / "gt.csv"
    csv_file.write_text(
        "chunk_id,video_name,start_time,end_time,test_query,expected_answer\n"
        'c0,animals.mp4,1.0,5.0,"¿Cuándo salen lémures, o monos?",respuesta\n'
        "c1,animals.mp4,5.0,10.5,león tumbado,otra respuesta\n",
        encoding="utf-8",
    )

    assert load_ground_truth_csv(csv_file) == [
        GroundTruthQuery("¿Cuándo salen lémures, o monos?", "animals.mp4", 1.0, 5.0),
        GroundTruthQuery("león tumbado", "animals.mp4", 5.0, 10.5),
    ]


def test_load_ground_truth_csv_filters_by_video_name() -> None:
    from tests.e2e.harness import GROUND_TRUTH_CSV, PYTHON_VIDEO_NAME, VIDEO_NAME

    all_rows = load_ground_truth_csv(GROUND_TRUTH_CSV)
    animals_rows = load_ground_truth_csv(GROUND_TRUTH_CSV, video_name=VIDEO_NAME)
    python_rows = load_ground_truth_csv(GROUND_TRUTH_CSV, video_name=PYTHON_VIDEO_NAME)

    assert len(animals_rows) == 46
    assert len(python_rows) == 56
    assert len(all_rows) == 102
    assert all(r.video_name == VIDEO_NAME for r in animals_rows)
    assert all(r.video_name == PYTHON_VIDEO_NAME for r in python_rows)



def test_load_ground_truth_csv_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_ground_truth_csv(tmp_path / "missing.csv")


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


def test_evaluate_multi_target_results_counts_distinct_covered_intervals() -> None:
    target = MultiTargetQuery(
        query="Felinos",
        video_name="animals.mp4",
        expected_intervals=((5.0, 10.0), (18.0, 23.0), (35.0, 39.0), (193.0, 196.0)),
    )
    results = [
        RerankedResult(
            SearchResult(0.95, "animals.mp4", 0, 14.0, 18.0), 0.1, False, "elefantes"
        ),
        RerankedResult(
            SearchResult(0.90, "animals.mp4", 1, 5.0, 8.0), 0.95, True, "león parte 1"
        ),
        RerankedResult(
            SearchResult(0.88, "animals.mp4", 2, 8.0, 10.0), 0.92, True, "león parte 2"
        ),
        RerankedResult(
            SearchResult(0.85, "animals.mp4", 3, 18.0, 23.0), 0.90, True, "gato gris"
        ),
        RerankedResult(
            SearchResult(0.80, "animals.mp4", 4, 59.0, 64.0), 0.60, True, "falso positivo"
        ),
    ]

    ev = evaluate_multi_target_results(target, results, latency_seconds=0.45)
    assert ev.expected_count == 4
    assert ev.matched_intervals == 2
    assert ev.interval_recall == pytest.approx(2.0 / 4.0)
    assert ev.relevant_returned == 4
    assert ev.true_positives == 3
    assert ev.precision == pytest.approx(3.0 / 4.0)
    assert ev.first_hit_rank == 2
    assert ev.latency_seconds == pytest.approx(0.45)


def test_evaluate_negative_results_distinguishes_raw_and_reranked() -> None:
    raw_results = [
        SearchResult(0.65, "animals.mp4", 0, 32.0, 35.0),
        SearchResult(0.61, "animals.mp4", 1, 212.0, 217.0),
    ]
    raw_ev = evaluate_negative_results("Tiburón blanco", raw_results, latency_seconds=0.1)
    assert raw_ev.total_candidates == 2
    assert raw_ev.false_positives == 2
    assert raw_ev.correctly_rejected is False

    reranked_rejected = [
        RerankedResult(
            SearchResult(0.65, "animals.mp4", 0, 32.0, 35.0), 0.0, False, "es una medusa"
        ),
        RerankedResult(
            SearchResult(0.61, "animals.mp4", 1, 212.0, 217.0), 0.0, False, "otra medusa"
        ),
    ]
    rejected_ev = evaluate_negative_results(
        "Tiburón blanco", reranked_rejected, latency_seconds=0.5
    )
    assert rejected_ev.total_candidates == 2
    assert rejected_ev.false_positives == 0
    assert rejected_ev.correctly_rejected is True


def test_run_multi_target_and_negative_benchmarks_execute_queries() -> None:
    service = MagicMock(spec=VideoRAGService)
    service.search = AsyncMock(
        side_effect=[
            [
                RerankedResult(
                    SearchResult(0.9, "animals.mp4", 0, 23.0, 29.0),
                    0.9,
                    True,
                    "camaleón",
                )
            ],
            [
                RerankedResult(
                    SearchResult(0.5, "animals.mp4", 1, 0.0, 1.0),
                    0.0,
                    False,
                    "elefante",
                )
            ],
        ]
    )

    multi_evals = asyncio.run(
        run_multi_target_benchmark(
            service,
            [MultiTargetQuery("Reptiles", "animals.mp4", ((23.0, 29.0), (232.0, 241.0)))],
            top_k=5,
            rerank=True,
        )
    )
    neg_evals = asyncio.run(
        run_negative_benchmark(service, ["Tiburón blanco"], top_k=5, rerank=True)
    )

    assert len(multi_evals) == 1
    assert multi_evals[0].matched_intervals == 1
    assert len(neg_evals) == 1
    assert neg_evals[0].correctly_rejected is True


def test_harness_complex_reports_serialize_multi_target_and_negative_metrics(
    tmp_path: Path,
) -> None:
    from tests.e2e.harness import (
        MULTI_TARGET_QUERIES,
        NEGATIVE_QUERIES,
        PYTHON_MULTI_TARGET_QUERIES,
        PYTHON_NEGATIVE_QUERIES,
        write_multi_target_report,
        write_negative_report,
    )

    assert len(MULTI_TARGET_QUERIES) == 10
    assert len(NEGATIVE_QUERIES) == 8
    assert len(PYTHON_MULTI_TARGET_QUERIES) == 10
    assert len(PYTHON_NEGATIVE_QUERIES) == 8

    target = MULTI_TARGET_QUERIES[0]
    ev_plain = evaluate_multi_target_results(
        target, [SearchResult(0.9, "animals.mp4", 0, 5.0, 10.0)]
    )
    ev_rerank = evaluate_multi_target_results(
        target,
        [
            RerankedResult(
                SearchResult(0.9, "animals.mp4", 0, 5.0, 10.0), 0.95, True, "león"
            ),
            RerankedResult(
                SearchResult(0.8, "animals.mp4", 1, 18.0, 23.0), 0.90, True, "gato"
            ),
        ],
    )
    multi_csv = tmp_path / "multi.csv"
    write_multi_target_report(multi_csv, [ev_plain], [ev_rerank])
    assert multi_csv.is_file()

    neg_plain = evaluate_negative_results(
        "Tiburón", [SearchResult(0.6, "animals.mp4", 0, 0.0, 5.0)]
    )
    neg_rerank = evaluate_negative_results(
        "Tiburón",
        [
            RerankedResult(
                SearchResult(0.6, "animals.mp4", 0, 0.0, 5.0), 0.0, False, "no hay"
            )
        ],
    )
    neg_csv = tmp_path / "neg.csv"
    write_negative_report(neg_csv, [neg_plain], [neg_rerank])
    assert neg_csv.is_file()


def test_evaluate_matryoshka_dimensions_and_csv_export(tmp_path: Path) -> None:
    vec_a = [1.0 if i == 0 else 0.0 for i in range(3072)]
    vec_b = [1.0 if i == 1 else 0.0 for i in range(3072)]
    chunks = [
        VectorizedChunk(0, 0.0, 5.0, vec_a, transcript="chunk a"),
        VectorizedChunk(1, 5.0, 10.0, vec_b, transcript="chunk b"),
    ]
    dataset = [
        GroundTruthQuery("buscar a", "demo.mp4", 0.0, 5.0),
        GroundTruthQuery("buscar b", "demo.mp4", 5.0, 10.0),
    ]
    query_vectors = {
        "buscar a": vec_a,
        "buscar b": vec_b,
    }

    metrics = evaluate_matryoshka_dimensions(
        video_name="demo.mp4",
        chunks_3072=chunks,
        dataset=dataset,
        query_vectors_3072=query_vectors,
        dimensions=(3072, 1536, 768),
        top_k=5,
    )
    assert len(metrics) == 3
    assert [m.dimension for m in metrics] == [3072, 1536, 768]
    assert [m.bytes_per_vector for m in metrics] == [12288, 6144, 3072]
    assert [m.kb_per_vector for m in metrics] == [12.0, 6.0, 3.0]
    assert [m.storage_savings_pct for m in metrics] == pytest.approx([0.0, 50.0, 75.0])
    for m in metrics:
        assert isinstance(m, MatryoshkaDimensionMetrics)
        assert m.recall_at_1 == pytest.approx(1.0)
        assert m.recall_at_3 == pytest.approx(1.0)
        assert m.recall_at_5 == pytest.approx(1.0)
        assert m.mrr == pytest.approx(1.0)
        assert m.mean_search_ms >= 0.0

    csv_path = tmp_path / "mrl_ablation.csv"
    write_matryoshka_ablation_csv(csv_path, metrics)
    assert csv_path.is_file()
    content = csv_path.read_text(encoding="utf-8")
    assert "3072" in content and "1536" in content and "768" in content


def test_evaluate_matryoshka_multi_target_and_negative(tmp_path: Path) -> None:
    vec_a = [1.0 if i == 0 else 0.0 for i in range(3072)]
    vec_b = [1.0 if i == 1 else 0.0 for i in range(3072)]
    chunks = [
        VectorizedChunk(0, 5.0, 10.0, vec_a, transcript="león"),
        VectorizedChunk(1, 18.0, 23.0, vec_b, transcript="gato"),
    ]
    multi_queries = [
        MultiTargetQuery("Felinos", "animals.mp4", ((5.0, 10.0), (18.0, 23.0)))
    ]
    neg_queries = ["Tiburón blanco"]
    query_vectors = {
        "Felinos": [1.0 if i in (0, 1) else 0.0 for i in range(3072)],
        "Tiburón blanco": vec_a,
    }

    async def fake_rerank(
        query: str, candidates: list[SearchResult]
    ) -> list[RerankedResult]:
        is_rel = query == "Felinos"
        return [
            RerankedResult(
                result=c,
                relevance_score=0.95 if is_rel else 0.0,
                is_relevant=is_rel,
                reasoning="juez",
            )
            for c in candidates
        ]

    multi_metrics = asyncio.run(
        evaluate_matryoshka_multi_target(
            video_name="animals.mp4",
            chunks_3072=chunks,
            queries=multi_queries,
            query_vectors_3072=query_vectors,
            rerank_fn=fake_rerank,
            dimensions=(3072, 1536, 768),
            top_k=10,
        )
    )
    assert len(multi_metrics) == 3
    for mm in multi_metrics:
        assert isinstance(mm, MatryoshkaMultiTargetMetrics)
        assert mm.mean_interval_recall_stage1 == pytest.approx(1.0)
        assert mm.mean_interval_recall_stage2 == pytest.approx(1.0)
        assert mm.mean_precision_stage2 == pytest.approx(1.0)

    neg_metrics = asyncio.run(
        evaluate_matryoshka_negative(
            video_name="animals.mp4",
            chunks_3072=chunks,
            queries=neg_queries,
            query_vectors_3072=query_vectors,
            rerank_fn=fake_rerank,
            dimensions=(3072, 1536, 768),
            top_k=5,
        )
    )
    assert len(neg_metrics) == 3
    for nm in neg_metrics:
        assert isinstance(nm, MatryoshkaNegativeMetrics)
        assert nm.false_positives_stage1 == 2
        assert nm.false_positives_stage2 == 0
        assert nm.rejection_rate_stage2 == pytest.approx(1.0)

    multi_csv = tmp_path / "mrl_multi.csv"
    neg_csv = tmp_path / "mrl_neg.csv"
    write_matryoshka_multi_target_csv(multi_csv, multi_metrics)
    write_matryoshka_negative_csv(neg_csv, neg_metrics)
    assert multi_csv.is_file() and neg_csv.is_file()





