"""Pruebas E2E de ablación Matryoshka (3072, 1536 y 768 dimensiones) sobre python_tutorial.mp4 y animals.mp4."""

from __future__ import annotations

import csv
import os

import pytest
from dotenv import load_dotenv

from tests.e2e.harness import (
    EXPECTED_E2E_CHUNKS,
    MATRYOSHKA_DIMENSIONS,
    PYTHON_VIDEO_NAME,
    RAW_DIR,
    VIDEO_NAME,
    MatryoshkaAblationRun,
)

load_dotenv()

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY no está definida."
    ),
    pytest.mark.skipif(
        not (RAW_DIR / VIDEO_NAME).is_file(), reason=f"Falta {RAW_DIR / VIDEO_NAME}."
    ),
    pytest.mark.skipif(
        not (RAW_DIR / PYTHON_VIDEO_NAME).is_file(),
        reason=f"Falta {RAW_DIR / PYTHON_VIDEO_NAME}.",
    ),
]


def test_matryoshka_indexes_both_videos_with_expected_chunks(
    matryoshka_e2e_run: MatryoshkaAblationRun,
) -> None:
    assert matryoshka_e2e_run.python_chunks == EXPECTED_E2E_CHUNKS
    assert matryoshka_e2e_run.animals_chunks == EXPECTED_E2E_CHUNKS


def test_matryoshka_evaluates_all_official_dimensions(
    matryoshka_e2e_run: MatryoshkaAblationRun,
) -> None:
    assert [m.dimension for m in matryoshka_e2e_run.python_metrics] == list(
        MATRYOSHKA_DIMENSIONS
    )
    assert [m.dimension for m in matryoshka_e2e_run.animals_metrics] == list(
        MATRYOSHKA_DIMENSIONS
    )

    for m in [*matryoshka_e2e_run.python_metrics, *matryoshka_e2e_run.animals_metrics]:
        assert m.recall_at_5 >= 0.85, (
            f"{m.video_name} (d={m.dimension}): Recall@5={m.recall_at_5:.2%}"
        )
        assert m.mrr >= 0.75, (
            f"{m.video_name} (d={m.dimension}): MRR={m.mrr:.4f}"
        )


def test_matryoshka_csv_report_contains_six_rows(
    matryoshka_e2e_run: MatryoshkaAblationRun,
) -> None:
    assert matryoshka_e2e_run.csv_path.is_file()
    with matryoshka_e2e_run.csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    assert [int(r["dimension"]) for r in rows] == [3072, 1536, 3072 - 2304, 3072, 1536, 768]


def test_matryoshka_multi_target_and_negative_across_dimensions(
    matryoshka_e2e_run: MatryoshkaAblationRun,
) -> None:
    assert matryoshka_e2e_run.multi_target_csv_path.is_file()
    assert matryoshka_e2e_run.negative_csv_path.is_file()

    for mm in [
        *matryoshka_e2e_run.python_multi_target,
        *matryoshka_e2e_run.animals_multi_target,
    ]:
        assert mm.mean_interval_recall_stage1 >= 0.45
        assert mm.mean_precision_stage2 >= mm.mean_precision_stage1

    for nm in [
        *matryoshka_e2e_run.python_negative,
        *matryoshka_e2e_run.animals_negative,
    ]:
        assert nm.false_positives_stage1 > 0
        assert nm.rejection_rate_stage2 >= 0.75

