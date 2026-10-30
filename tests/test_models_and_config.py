"""Pruebas unitarias de ``src.models`` y ``src.config``."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from src.config import (
    MAX_CHUNKS_PER_VIDEO,
    VECTOR_SIZE,
    ChunkingConfig,
    SearchConfig,
    VectorizerConfig,
    build_run_config,
)
from src.models import (
    QuotaSafetyError,
    RerankedResult,
    SearchResult,
    VectorDBError,
    VectorizationError,
    VectorizedChunk,
    VideoChunk,
)


def test_domain_models_are_immutable_and_delegate_fields() -> None:
    chunk = VideoChunk(index=0, start_frame=0, end_frame=10, start_time=0.0, end_time=1.0)
    with pytest.raises(FrozenInstanceError):
        chunk.index = 1  # type: ignore[misc]

    vec_chunk = VectorizedChunk(index=0, start_time=0.0, end_time=1.0, embedding=[0.1, 0.2])
    with pytest.raises(FrozenInstanceError):
        vec_chunk.index = 2  # type: ignore[misc]

    res = SearchResult(
        score=0.42,
        video_name="demo.mp4",
        chunk_index=3,
        start_time=12.0,
        end_time=18.5,
    )
    reranked = RerankedResult(
        result=res,
        relevance_score=0.91,
        is_relevant=True,
        reasoning="Coincide con la consulta.",
    )
    assert reranked.video_name == "demo.mp4"
    assert reranked.chunk_index == 3
    assert reranked.start_time == 12.0
    assert reranked.end_time == 18.5
    assert reranked.vector_score == 0.42
    assert reranked.score == 0.42


def test_domain_exceptions_inherit_from_runtime_error() -> None:
    for exc_cls in (QuotaSafetyError, VectorizationError, VectorDBError):
        assert issubclass(exc_cls, RuntimeError)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"threshold": 0.0},
        {"threshold": 1.0},
        {"min_shot_frames": 0},
        {"max_shot_duration": 0.0},
        {"max_shot_duration": -2.0},
        {"resize_width": 7},
        {"entropy_stride": 0},
    ],
)
def test_chunking_config_rejects_invalid_values(kwargs: dict[str, float | int]) -> None:
    with pytest.raises(ValueError):
        ChunkingConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"output_dimensionality": 0},
        {"max_retries": -1},
        {"retry_base_delay": -0.1},
        {"jpeg_quality": 0},
        {"jpeg_quality": 101},
        {"target_width": -1},
    ],
)
def test_vectorizer_config_rejects_invalid_values(kwargs: dict[str, float | int]) -> None:
    with pytest.raises(ValueError):
        VectorizerConfig(**kwargs)  # type: ignore[arg-type]


def test_search_config_rejects_non_positive_top_k() -> None:
    with pytest.raises(ValueError):
        SearchConfig(top_k=0)


def test_build_run_config_includes_defaults_and_mode() -> None:
    ui_cfg = build_run_config(mode="streamlit_ui")
    assert ui_cfg["mode"] == "streamlit_ui"
    assert "cli_args" not in ui_cfg
    assert ui_cfg["vector_size"] == VECTOR_SIZE
    assert ui_cfg["max_chunks_per_video"] == MAX_CHUNKS_PER_VIDEO
    assert ui_cfg["chunking_defaults"] == {
        "threshold": 0.5,
        "min_shot_frames": 12,
        "max_shot_duration": 8.0,
        "resize_width": 160,
        "entropy_stride": 1,
    }
    assert ui_cfg["vectorizer_defaults"] == {
        "model": "gemini-embedding-2-preview",
        "output_dimensionality": 3072,
        "max_retries": 4,
        "retry_base_delay": 4.0,
        "jpeg_quality": 80,
        "target_width": 512,
    }
    assert ui_cfg["search_defaults"] == {
        "top_k": 5,
        "rerank": True,
        "reranker_model": "gemini-3.5-flash-lite",
    }

    cli_cfg = build_run_config(cli_args={"video": "test.mp4", "top_k": 3})
    assert cli_cfg["cli_args"] == {"video": "test.mp4", "top_k": 3}
    assert "mode" not in cli_cfg

