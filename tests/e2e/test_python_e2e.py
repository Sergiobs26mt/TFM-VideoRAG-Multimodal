"""Pruebas E2E con la API real de Gemini: python_tutorial.mp4 frente a data/video_chunks_ground_truth.csv."""

from __future__ import annotations

import csv
import os

import pytest
from dotenv import load_dotenv

from tests.e2e.harness import (
    PYTHON_MULTI_TARGET_QUERIES,
    PYTHON_NEGATIVE_QUERIES,
    PYTHON_VIDEO_NAME,
    RAW_DIR,
    E2ERun,
)

load_dotenv()

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY no está definida."
    ),
    pytest.mark.skipif(
        not (RAW_DIR / PYTHON_VIDEO_NAME).is_file(),
        reason=f"Falta {RAW_DIR / PYTHON_VIDEO_NAME}.",
    ),
]

# 242 s de vídeo con planos de 8 s como máximo obligan a un mínimo de 31 chunks.
MIN_INDEXED_CHUNKS = 31
MIN_RECALL_AT_K = 0.85
MIN_MRR = 0.75
RERANK_RECALL_AT_1_TOLERANCE = 0.05
MIN_MULTI_TARGET_MEAN_RECALL = 0.50
MIN_NEGATIVE_REJECTION_RATE = 0.75


def test_python_indexing_covers_the_whole_video(python_e2e_run: E2ERun) -> None:
    assert python_e2e_run.indexed_videos == [PYTHON_VIDEO_NAME]
    assert python_e2e_run.indexed_chunks >= MIN_INDEXED_CHUNKS


def test_python_retrieval_finds_the_labelled_interval(python_e2e_run: E2ERun) -> None:
    summary = python_e2e_run.retrieval
    assert summary.total_queries == len(python_e2e_run.dataset)
    assert summary.recall_at_k >= MIN_RECALL_AT_K, (
        f"Recall@{summary.top_k}={summary.recall_at_k:.2%}"
    )
    assert summary.mrr >= MIN_MRR, f"MRR={summary.mrr:.3f}"


def test_python_rerank_does_not_hurt_top_result(python_e2e_run: E2ERun) -> None:
    floor = python_e2e_run.retrieval.recall_at_1 - RERANK_RECALL_AT_1_TOLERANCE
    assert python_e2e_run.reranked.recall_at_1 >= floor, (
        f"Recall@1 con rerank={python_e2e_run.reranked.recall_at_1:.2%}, "
        f"sin rerank={python_e2e_run.retrieval.recall_at_1:.2%}"
    )


def test_python_report_lists_every_query(python_e2e_run: E2ERun) -> None:
    with python_e2e_run.report_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["query"] for row in rows] == [
        item.query for item in python_e2e_run.dataset
    ]


def test_python_multi_target_queries_find_multiple_matching_intervals(
    python_e2e_run: E2ERun,
) -> None:
    retrieval = python_e2e_run.multi_target_retrieval
    reranked = python_e2e_run.multi_target_reranked
    assert len(retrieval) == len(PYTHON_MULTI_TARGET_QUERIES)
    assert len(reranked) == len(PYTHON_MULTI_TARGET_QUERIES)

    mean_recall_retrieval = sum(e.interval_recall for e in retrieval) / len(retrieval)
    mean_precision_retrieval = sum(e.precision for e in retrieval) / len(retrieval)
    mean_precision_rerank = sum(e.precision for e in reranked) / len(reranked)

    assert mean_recall_retrieval >= MIN_MULTI_TARGET_MEAN_RECALL, (
        f"Cobertura media multi-objetivo={mean_recall_retrieval:.2%}"
    )
    assert mean_precision_rerank >= mean_precision_retrieval, (
        f"Precisión media con rerank={mean_precision_rerank:.2%}, "
        f"sin rerank={mean_precision_retrieval:.2%}"
    )


def test_python_negative_queries_are_rejected_by_reranker(
    python_e2e_run: E2ERun,
) -> None:
    retrieval = python_e2e_run.negative_retrieval
    reranked = python_e2e_run.negative_reranked
    assert len(retrieval) == len(PYTHON_NEGATIVE_QUERIES)
    assert len(reranked) == len(PYTHON_NEGATIVE_QUERIES)

    assert all(ev.false_positives == ev.total_candidates for ev in retrieval)
    rejected_count = sum(1 for ev in reranked if ev.correctly_rejected)
    rejection_rate = rejected_count / len(reranked)
    assert rejection_rate >= MIN_NEGATIVE_REJECTION_RATE, (
        f"Tasa de rechazo negativo={rejection_rate:.2%} ({rejected_count}/{len(reranked)})"
    )


def test_python_complex_reports_list_every_query(python_e2e_run: E2ERun) -> None:
    with python_e2e_run.multi_target_report_path.open(
        encoding="utf-8", newline=""
    ) as handle:
        multi_rows = list(csv.DictReader(handle))
    assert [row["query"] for row in multi_rows] == [
        item.query for item in PYTHON_MULTI_TARGET_QUERIES
    ]

    with python_e2e_run.negative_report_path.open(
        encoding="utf-8", newline=""
    ) as handle:
        neg_rows = list(csv.DictReader(handle))
    assert [row["query"] for row in neg_rows] == list(PYTHON_NEGATIVE_QUERIES)
