"""Pruebas unitarias de la interfaz de usuario (Streamlit UI)."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from src.models import SearchResult
from src.ui import app as ui_app


def test_execute_search() -> None:
    """Verifica que execute_search delega en VideoRAGService."""
    fake_results = [
        SearchResult(
            score=0.92,
            video_name="animals.mp4",
            chunk_index=1,
            start_time=5.0,
            end_time=10.0,
        )
    ]

    mock_service = MagicMock()
    mock_service.search = AsyncMock(return_value=fake_results)

    results = asyncio.run(
        ui_app.execute_search(
            query="un león rugiendo",
            service=mock_service,
            limit=3,
            video_filter="animals.mp4",
        )
    )

    assert results == fake_results
    mock_service.search.assert_called_once_with(
        "un león rugiendo",
        limit=3,
        video_filter="animals.mp4",
        rerank=True,
        auditor=None,
    )


def test_index_video_in_ui(tmp_path: Path) -> None:
    """Verifica que index_video_in_ui invoca VideoRAGService.index_video y actualiza progreso."""
    dummy_video = tmp_path / "dummy.mp4"
    dummy_video.write_bytes(b"dummy video content")

    mock_service = MagicMock()
    mock_service.index_video = AsyncMock(return_value=5)

    with patch("src.ui.app.st.progress") as mock_progress_creator:
        mock_bar = MagicMock()
        mock_progress_creator.return_value = mock_bar

        n = ui_app.index_video_in_ui(dummy_video, mock_service, max_concurrent=2)

    assert n == 5
    mock_service.index_video.assert_called_once()
    mock_bar.progress.assert_called()


def test_index_video_in_ui_custom_params(tmp_path: Path) -> None:
    """Verifica que index_video_in_ui reenvía parámetros personalizados a VideoRAGService.index_video."""
    dummy_video = tmp_path / "custom.mp4"
    dummy_video.write_bytes(b"data")

    mock_service = MagicMock()
    mock_service.index_video = AsyncMock(return_value=3)

    with patch("src.ui.app.st.progress"):
        n = ui_app.index_video_in_ui(
            dummy_video,
            mock_service,
            max_concurrent=4,
            threshold=0.35,
            min_shot_frames=30,
            entropy_stride=2,
            jpeg_quality=80,
            target_width=640,
        )

    assert n == 3
    mock_service.index_video.assert_called_once()
    kwargs = mock_service.index_video.call_args.kwargs
    assert kwargs["max_concurrent"] == 4
    assert kwargs["threshold"] == 0.35
    assert kwargs["min_shot_frames"] == 30
    assert kwargs["entropy_stride"] == 2
    assert kwargs["jpeg_quality"] == 80
    assert kwargs["target_width"] == 640


def test_execute_search_with_auditor() -> None:
    """Verifica que execute_search reenvía el auditor a VideoRAGService.search."""
    fake_results = [
        SearchResult(
            score=0.95,
            video_name="animals.mp4",
            chunk_index=2,
            start_time=1.0,
            end_time=3.0,
        )
    ]

    mock_service = MagicMock()
    mock_service.search = AsyncMock(return_value=fake_results)
    mock_auditor = MagicMock()

    results = asyncio.run(
        ui_app.execute_search(
            query="pinguinos",
            service=mock_service,
            limit=5,
            auditor=mock_auditor,
        )
    )

    assert results == fake_results
    mock_service.search.assert_called_once_with(
        "pinguinos",
        limit=5,
        video_filter=None,
        rerank=True,
        auditor=mock_auditor,
    )


def test_build_ui_run_config() -> None:
    """Verifica que build_ui_run_config genera los parámetros esperados."""
    config = ui_app.build_ui_run_config()
    assert config["mode"] == "streamlit_ui"
    assert config["vector_size"] == 3072
    assert "chunking_defaults" in config
    assert "vectorizer_defaults" in config


def test_execute_search_truncates_to_limit() -> None:
    """Verifica que execute_search respeta el límite solicitado por el usuario."""
    fake_results = [
        SearchResult(
            score=0.9 - i * 0.1,
            video_name="tutorial.mp4",
            chunk_index=i,
            start_time=float(i),
            end_time=float(i + 1),
        )
        for i in range(6)
    ]
    mock_service = MagicMock()
    mock_service.search = AsyncMock(return_value=fake_results)

    results = asyncio.run(
        ui_app.execute_search(query="bucles", service=mock_service, limit=3)
    )
    assert len(results) == 3

