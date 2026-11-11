"""Pruebas unitarias de ``src.models`` y ``src.config``."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from src.config import (
    EMBEDDING_BACKENDS,
    MAX_CHUNKS_PER_VIDEO,
    VECTOR_SIZE,
    AudioConfig,
    ChunkingConfig,
    SearchConfig,
    VectorizerConfig,
    build_run_config,
    collection_name_for,
    default_dimension,
    validate_embedding_choice,
)
from src.models import (
    AudioTranscriptionError,
    IndexStats,
    MultimodalQuery,
    QuotaSafetyError,
    RerankedResult,
    SearchResult,
    TranscriptSegment,
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
        transcript="diccionarios en python",
    )
    reranked = RerankedResult(
        result=res,
        relevance_score=0.91,
        is_relevant=True,
        reasoning="Coincide con la consulta.",
    )
    assert chunk.transcript is None
    assert vec_chunk.transcript is None
    assert res.transcript == "diccionarios en python"
    assert reranked.video_name == "demo.mp4"
    assert reranked.chunk_index == 3
    assert reranked.start_time == 12.0
    assert reranked.end_time == 18.5
    assert reranked.vector_score == 0.42
    assert reranked.score == 0.42
    assert reranked.transcript == "diccionarios en python"

    seg = TranscriptSegment(start_time=1.0, end_time=3.5, text="hola mundo")
    assert (seg.start_time, seg.end_time, seg.text) == (1.0, 3.5, "hola mundo")
    with pytest.raises(FrozenInstanceError):
        seg.text = "otro"  # type: ignore[misc]


def test_domain_exceptions_inherit_from_runtime_error() -> None:
    for exc_cls in (
        QuotaSafetyError,
        VectorizationError,
        VectorDBError,
        AudioTranscriptionError,
    ):
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
        {"output_dimensionality": 127},
        {"output_dimensionality": 3073},
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


def test_gemini_backend_dimensions_are_valid_vectorizer_dimensions() -> None:
    assert EMBEDDING_BACKENDS["gemini"].dimensions == (3072, 1536, 768)
    for dim in EMBEDDING_BACKENDS["gemini"].dimensions:
        cfg = VectorizerConfig(output_dimensionality=dim)
        assert cfg.output_dimensionality == dim


def test_backend_dimensions_and_defaults() -> None:
    assert EMBEDDING_BACKENDS["gemini"].dimensions == (3072, 1536, 768)
    assert EMBEDDING_BACKENDS["qwen3-vl-2b"].dimensions == (2048, 1536, 768)
    assert EMBEDDING_BACKENDS["gemma2"].dimensions == (768, 512, 256)
    assert [default_dimension(b) for b in ("gemini", "qwen3-vl-2b", "gemma2")] == [
        3072,
        2048,
        768,
    ]
    assert [EMBEDDING_BACKENDS[b].is_local for b in ("gemini", "qwen3-vl-2b", "gemma2")] == [
        False,
        True,
        True,
    ]
    assert EMBEDDING_BACKENDS["qwen3-vl-2b"].model_id == "Qwen/Qwen3-VL-Embedding-2B"
    assert EMBEDDING_BACKENDS["gemma2"].model_id == "google/embeddinggemma-2"


def test_collection_name_for_keeps_legacy_name_for_gemini_3072() -> None:
    assert collection_name_for("gemini", 3072) == "video_chunks"


def test_collection_name_for_uses_suffix() -> None:
    assert collection_name_for("gemini", 1536) == "video_chunks__gemini__1536"
    assert collection_name_for("qwen3-vl-2b", 2048) == "video_chunks__qwen3-vl-2b__2048"
    assert collection_name_for("gemma2", 768) == "video_chunks__gemma2__768"


@pytest.mark.parametrize(
    ("backend", "dim"),
    [("gemma2", 3072), ("gemma2", 1536), ("qwen3-vl-2b", 512), ("gemini", 2048), ("otro", 768)],
)
def test_validate_embedding_choice_rejects_foreign_dimension_or_backend(
    backend: str, dim: int
) -> None:
    with pytest.raises(ValueError):
        validate_embedding_choice(backend, dim)
    with pytest.raises(ValueError):
        collection_name_for(backend, dim)


def test_build_run_config_records_embedding_fields() -> None:
    stats = IndexStats("gemma2", "Local MPS", 768, 12.5, 40)
    cfg = build_run_config(vector_size=768, embedding_backend="gemma2", index_stats=stats)
    assert cfg["embedding_backend"] == "gemma2"
    assert cfg["embedding_dimension"] == 768
    assert cfg["device"] == "Local MPS"
    assert cfg["index_seconds"] == 12.5


def test_build_run_config_without_index_stats_has_empty_device_and_seconds() -> None:
    cfg = build_run_config()
    assert cfg["embedding_backend"] == "gemini"
    assert cfg["device"] is None
    assert cfg["index_seconds"] is None


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
    assert ui_cfg["audio_defaults"] == {
        "enable_audio": True,
        "whisper_model": "base",
        "language": "es",
    }

    with pytest.raises(ValueError):
        AudioConfig(whisper_model="   ")

    cli_cfg = build_run_config(cli_args={"video": "test.mp4", "top_k": 3})
    assert cli_cfg["cli_args"] == {"video": "test.mp4", "top_k": 3}
    assert "mode" not in cli_cfg


def test_multimodal_query_properties_and_immutability() -> None:
    empty_q = MultimodalQuery(text="   ", txt_contents=("  ",))
    assert not empty_q.has_content
    assert empty_q.composed_text == ""
    assert empty_q.modality_summary == {
        "has_text": False,
        "txt_count": 0,
        "image_count": 0,
        "audio_count": 0,
        "audio_transcript_count": 0,
        "composed_text": "",
    }

    img_only = MultimodalQuery(images=(b"fake-jpeg",))
    assert img_only.has_content
    assert img_only.composed_text == ""
    assert img_only.modality_summary["image_count"] == 1

    audio_raw_only = MultimodalQuery(audio_paths=(Path("voice.wav"),), audio_bytes=(b"RIFF",))
    assert audio_raw_only.has_content
    assert audio_raw_only.modality_summary["audio_count"] == 2

    full_q = MultimodalQuery(
        text="  león rugiendo ",
        txt_contents=(" en la sabana ", ""),
        images=(b"img1", b"img2"),
        audio_paths=(Path("a.mp3"),),
        audio_transcripts=(" documental de naturaleza ",),
    )
    assert full_q.has_content
    assert full_q.composed_text == "león rugiendo en la sabana documental de naturaleza"
    assert full_q.modality_summary == {
        "has_text": True,
        "txt_count": 1,
        "image_count": 2,
        "audio_count": 1,
        "audio_transcript_count": 1,
        "composed_text": "león rugiendo en la sabana documental de naturaleza",
    }

    with pytest.raises(FrozenInstanceError):
        full_q.text = "otro"  # type: ignore[misc]


