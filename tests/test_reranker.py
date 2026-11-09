"""Tests unitarios para el módulo de reordenación multimodal (Etapa 2)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Generator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from google.genai import types

from src.models import RerankedResult, SearchResult
from src.rag_engine.reranker import MultimodalReranker


def _make_candidate(
    score: float,
    chunk_index: int,
    start_time: float,
    end_time: float,
    video_name: str = "test.mp4",
) -> SearchResult:
    return SearchResult(
        score=score,
        video_name=video_name,
        chunk_index=chunk_index,
        start_time=start_time,
        end_time=end_time,
    )


@pytest.fixture
def mock_genai_client() -> Generator[MagicMock, None, None]:
    with patch("src.utils.gemini_client.genai.Client") as mock_cls:
        client = MagicMock()
        client.aio.models.generate_content = AsyncMock()
        mock_cls.return_value = client
        yield client


def test_rerank_empty_candidates(mock_genai_client: MagicMock) -> None:
    reranker = MultimodalReranker(api_key="fake-key")
    results = asyncio.run(reranker.rerank("consulta", []))
    assert results == []
    mock_genai_client.aio.models.generate_content.assert_not_called()


def test_rerank_properties_accessors() -> None:
    cand = SearchResult(
        score=0.35,
        video_name="animals.mp4",
        chunk_index=13,
        start_time=24.0,
        end_time=30.0,
    )
    reranked = RerankedResult(
        result=cand,
        relevance_score=0.95,
        is_relevant=True,
        reasoning="Es un camaleón.",
    )
    assert reranked.video_name == "animals.mp4"
    assert reranked.chunk_index == 13
    assert reranked.start_time == 24.0
    assert reranked.end_time == 30.0
    assert reranked.vector_score == 0.35
    assert reranked.score == 0.35
    assert reranked.relevance_score == 0.95
    assert reranked.is_relevant is True


def test_rerank_success_ordering(mock_genai_client: MagicMock, tmp_path: Path) -> None:
    reranker = MultimodalReranker(api_key="fake-key")

    c1 = _make_candidate(0.357, 13, 24.0, 30.0)
    c2 = _make_candidate(0.352, 56, 159.0, 163.0)

    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    with patch(
        "src.rag_engine.reranker.extract_interval_frames", return_value=[dummy_frame]
    ):
        mock_response = MagicMock()
        mock_response.text = json.dumps(
            [
                {
                    "candidate_index": 0,
                    "is_relevant": True,
                    "relevance_score": 0.98,
                    "reasoning": "Se observa un camaleón claramente.",
                },
                {
                    "candidate_index": 1,
                    "is_relevant": False,
                    "relevance_score": 0.05,
                    "reasoning": "Son células sanguíneas microscópicas.",
                },
            ]
        )
        mock_genai_client.aio.models.generate_content.return_value = mock_response

        reranked = asyncio.run(reranker.rerank("Reptil", [c1, c2], raw_dir=tmp_path))

        assert len(reranked) == 2
        assert reranked[0].chunk_index == 13
        assert reranked[0].is_relevant is True
        assert reranked[0].relevance_score == 0.98

        assert reranked[1].chunk_index == 56
        assert reranked[1].is_relevant is False
        assert reranked[1].relevance_score == 0.05


def test_rerank_fallback_on_api_error(
    mock_genai_client: MagicMock, tmp_path: Path
) -> None:
    reranker = MultimodalReranker(
        api_key="fake-key", max_retries=1, retry_base_delay=0.001
    )
    c1 = _make_candidate(0.38, 1, 10.0, 20.0)
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    with patch(
        "src.rag_engine.reranker.extract_interval_frames", return_value=[dummy_frame]
    ):
        mock_genai_client.aio.models.generate_content.side_effect = OSError("API caída")

        reranked = asyncio.run(reranker.rerank("test", [c1], raw_dir=tmp_path))
        assert len(reranked) == 1
        assert reranked[0].chunk_index == 1
        assert "no disponible" in reranked[0].reasoning


def test_rerank_fallback_when_no_frames(
    mock_genai_client: MagicMock, tmp_path: Path
) -> None:
    reranker = MultimodalReranker(api_key="fake-key")
    c1 = _make_candidate(0.38, 1, 10.0, 20.0)

    with patch("src.rag_engine.reranker.extract_interval_frames", return_value=[]):
        reranked = asyncio.run(
            reranker.rerank(
                "test",
                [c1],
                raw_dir=tmp_path / "raw",
                processed_dir=tmp_path / "processed",
            )
        )
        assert len(reranked) == 1
        assert reranked[0].chunk_index == 1
        mock_genai_client.aio.models.generate_content.assert_not_called()


def test_rerank_uses_cached_keyframes_without_reading_raw_video(
    mock_genai_client: MagicMock, tmp_path: Path
) -> None:
    from src.processing.keyframe_cache import save_video_keyframes

    processed_dir = tmp_path / "processed"
    dummy_frame = np.zeros((60, 80, 3), dtype=np.uint8)
    save_video_keyframes(
        "test.mp4",
        {13: [dummy_frame, dummy_frame, dummy_frame]},
        processed_dir=processed_dir,
    )

    reranker = MultimodalReranker(api_key="fake-key")
    c1 = _make_candidate(0.45, 13, 24.0, 30.0, video_name="test.mp4")

    mock_response = MagicMock()
    mock_response.text = json.dumps(
        [
            {
                "candidate_index": 0,
                "is_relevant": True,
                "relevance_score": 0.92,
                "reasoning": "Fotograma recuperado de caché.",
            }
        ]
    )
    mock_genai_client.aio.models.generate_content.return_value = mock_response

    with patch("src.rag_engine.reranker.extract_interval_frames") as mock_extract:
        reranked = asyncio.run(
            reranker.rerank(
                "Reptil",
                [c1],
                raw_dir=tmp_path / "raw",
                processed_dir=processed_dir,
            )
        )
        mock_extract.assert_not_called()
        assert len(reranked) == 1
        assert reranked[0].relevance_score == 0.92
        sent_contents = mock_genai_client.aio.models.generate_content.call_args.kwargs[
            "contents"
        ]
        image_parts = [p for p in sent_contents if isinstance(p, types.Part)]
        assert len(image_parts) == 1


def test_rerank_downscales_cached_keyframes_exceeding_target_width(
    mock_genai_client: MagicMock, tmp_path: Path
) -> None:
    import cv2

    from src.processing.keyframe_cache import save_video_keyframes

    processed_dir = tmp_path / "processed"
    wide_frame = np.zeros((288, 512, 3), dtype=np.uint8)
    save_video_keyframes(
        "wide.mp4",
        {0: [wide_frame, wide_frame, wide_frame]},
        processed_dir=processed_dir,
        target_width=512,
    )

    reranker = MultimodalReranker(api_key="fake-key", target_width=384)
    c1 = _make_candidate(0.41, 0, 0.0, 8.0, video_name="wide.mp4")

    mock_response = MagicMock()
    mock_response.text = json.dumps(
        [
            {
                "candidate_index": 0,
                "is_relevant": True,
                "relevance_score": 0.99,
                "reasoning": "Código visible.",
            }
        ]
    )
    mock_genai_client.aio.models.generate_content.return_value = mock_response

    asyncio.run(
        reranker.rerank(
            "Bucles while",
            [c1],
            raw_dir=tmp_path / "raw",
            processed_dir=processed_dir,
        )
    )
    sent_contents = mock_genai_client.aio.models.generate_content.call_args.kwargs[
        "contents"
    ]
    image_parts = [p for p in sent_contents if isinstance(p, types.Part)]
    assert len(image_parts) == 1
    for part in image_parts:
        assert part.inline_data is not None
        assert part.inline_data.data is not None
        decoded = cv2.imdecode(
            np.frombuffer(part.inline_data.data, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        assert decoded is not None
        assert decoded.shape[1] == 384


def test_rerank_disables_thinking_and_populates_cache_on_miss(
    mock_genai_client: MagicMock, tmp_path: Path
) -> None:
    processed_dir = tmp_path / "processed"
    raw_dir = tmp_path / "raw"
    reranker = MultimodalReranker(api_key="fake-key")
    assert reranker._target_width == 384

    c1 = _make_candidate(0.39, 8, 36.6, 40.3, video_name="animals.mp4")
    dummy_frame = np.zeros((60, 80, 3), dtype=np.uint8)

    mock_response = MagicMock()
    mock_response.text = json.dumps(
        [
            {
                "candidate_index": 0,
                "is_relevant": True,
                "relevance_score": 1.0,
                "reasoning": "Leona caminando.",
            }
        ]
    )
    mock_genai_client.aio.models.generate_content.return_value = mock_response

    with patch(
        "src.rag_engine.reranker.extract_interval_frames", return_value=[dummy_frame]
    ) as mock_extract:
        asyncio.run(
            reranker.rerank("Busca leones", [c1], raw_dir=raw_dir, processed_dir=processed_dir)
        )
        assert mock_extract.call_count == 1

        # Segunda búsqueda sobre el mismo chunk: debe leerse de caché sin abrir OpenCV
        asyncio.run(
            reranker.rerank("Busca leones", [c1], raw_dir=raw_dir, processed_dir=processed_dir)
        )
        assert mock_extract.call_count == 1

    call_kwargs = mock_genai_client.aio.models.generate_content.call_args.kwargs
    config = call_kwargs["config"]
    assert config.max_output_tokens == 2048
    assert config.thinking_config is not None
    assert config.thinking_config.thinking_level == types.ThinkingLevel.MINIMAL
    assert config.thinking_config.thinking_budget is None


def test_rerank_recovers_complete_json_objects_when_response_is_truncated(
    mock_genai_client: MagicMock, tmp_path: Path
) -> None:
    reranker = MultimodalReranker(api_key="fake-key")
    c0 = _make_candidate(0.40, 0, 0.0, 4.0)
    c1 = _make_candidate(0.38, 1, 4.0, 8.0)

    dummy_frame = np.zeros((60, 80, 3), dtype=np.uint8)
    truncated_json = (
        '[\n'
        '  {"candidate_index": 0, "is_relevant": false, "relevance_score": 0.0, "reasoning": "No es un cocodrilo."},\n'
        '  {"candidate_index": 1, "is_relevant": false, "relevance_score": 0.0, "reasoning": "Muestra un'
    )
    mock_response = MagicMock()
    mock_response.text = truncated_json
    mock_genai_client.aio.models.generate_content.return_value = mock_response

    with patch(
        "src.rag_engine.reranker.extract_interval_frames", return_value=[dummy_frame]
    ):
        reranked = asyncio.run(
            reranker.rerank(
                "Cocodrilo",
                [c0, c1],
                raw_dir=tmp_path / "raw",
                processed_dir=tmp_path / "processed",
            )
        )

    assert len(reranked) == 2
    by_idx = {r.chunk_index: r for r in reranked}
    assert by_idx[0].is_relevant is False
    assert by_idx[0].reasoning == "No es un cocodrilo."


def test_rerank_fallback_on_httpx_read_error(
    mock_genai_client: MagicMock, tmp_path: Path
) -> None:
    import httpx

    reranker = MultimodalReranker(
        api_key="fake-key", max_retries=1, retry_base_delay=0.001
    )
    c1 = _make_candidate(0.38, 1, 10.0, 20.0)
    dummy_frame = np.zeros((60, 80, 3), dtype=np.uint8)

    with patch(
        "src.rag_engine.reranker.extract_interval_frames", return_value=[dummy_frame]
    ):
        mock_genai_client.aio.models.generate_content.side_effect = httpx.ReadError(
            "Conexión interrumpida"
        )
        reranked = asyncio.run(
            reranker.rerank(
                "test",
                [c1],
                raw_dir=tmp_path / "raw",
                processed_dir=tmp_path / "processed",
            )
        )
        assert len(reranked) == 1
        assert "no disponible" in reranked[0].reasoning
        assert mock_genai_client.aio.models.generate_content.call_count == 2


def test_rerank_combines_distinct_cached_keyframes_into_vertical_collage(
    mock_genai_client: MagicMock, tmp_path: Path
) -> None:
    import cv2

    from src.processing.keyframe_cache import save_video_keyframes

    processed_dir = tmp_path / "processed"
    f0 = np.zeros((60, 80, 3), dtype=np.uint8)
    f1 = np.full((60, 80, 3), 120, dtype=np.uint8)
    f2 = np.full((60, 80, 3), 240, dtype=np.uint8)
    save_video_keyframes(
        "tutorial.mp4",
        {7: [f0, f1, f2]},
        processed_dir=processed_dir,
        target_width=80,
    )

    reranker = MultimodalReranker(api_key="fake-key", target_width=80)
    c1 = _make_candidate(0.45, 7, 33.0, 41.0, video_name="tutorial.mp4")

    mock_response = MagicMock()
    mock_response.text = json.dumps(
        [
            {
                "candidate_index": 0,
                "is_relevant": True,
                "relevance_score": 0.95,
                "reasoning": "Rótulo visible en el tercer panel.",
            }
        ]
    )
    mock_genai_client.aio.models.generate_content.return_value = mock_response

    asyncio.run(
        reranker.rerank(
            "bool",
            [c1],
            raw_dir=tmp_path / "raw",
            processed_dir=processed_dir,
        )
    )
    sent_contents = mock_genai_client.aio.models.generate_content.call_args.kwargs[
        "contents"
    ]
    image_parts = [p for p in sent_contents if isinstance(p, types.Part)]
    assert len(image_parts) == 1
    assert image_parts[0].inline_data is not None
    assert image_parts[0].inline_data.data is not None
    decoded = cv2.imdecode(
        np.frombuffer(image_parts[0].inline_data.data, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    assert decoded is not None
    assert decoded.shape == (180, 80, 3)


def test_rerank_injects_bimodal_instructions_and_candidate_transcripts(
    mock_genai_client: MagicMock, tmp_path: Path
) -> None:
    reranker = MultimodalReranker(api_key="fake-key")
    c_with_audio = SearchResult(
        score=0.44,
        video_name="python_tutorial.mp4",
        chunk_index=12,
        start_time=61.1,
        end_time=68.0,
        transcript="Ahora definimos un diccionario con llaves clave y valor.",
    )
    c_silent = SearchResult(
        score=0.41,
        video_name="python_tutorial.mp4",
        chunk_index=13,
        start_time=68.0,
        end_time=72.0,
        transcript=None,
    )

    dummy_frame = np.zeros((60, 80, 3), dtype=np.uint8)
    mock_response = MagicMock()
    mock_response.text = json.dumps(
        [
            {
                "candidate_index": 0,
                "is_relevant": True,
                "relevance_score": 0.96,
                "reasoning": "La locución explica diccionarios.",
            },
            {
                "candidate_index": 1,
                "is_relevant": False,
                "relevance_score": 0.10,
                "reasoning": "Sin relación con diccionarios.",
            },
        ]
    )
    mock_genai_client.aio.models.generate_content.return_value = mock_response

    with patch(
        "src.rag_engine.reranker.extract_interval_frames", return_value=[dummy_frame]
    ):
        reranked = asyncio.run(
            reranker.rerank(
                "Diccionarios",
                [c_with_audio, c_silent],
                raw_dir=tmp_path / "raw",
                processed_dir=tmp_path / "processed",
            )
        )

    assert len(reranked) == 2
    assert reranked[0].is_relevant is True
    assert reranked[0].transcript == (
        "Ahora definimos un diccionario con llaves clave y valor."
    )

    sent_contents = mock_genai_client.aio.models.generate_content.call_args.kwargs[
        "contents"
    ]
    text_segments = "\n".join(p for p in sent_contents if isinstance(p, str))
    assert (
        "Transcripción hablada en este segmento: "
        "Ahora definimos un diccionario con llaves clave y valor."
    ) in text_segments
    assert "Transcripción hablada: [Sin locución detectada]" in text_segments
    assert "bimodal" in text_segments.lower()




