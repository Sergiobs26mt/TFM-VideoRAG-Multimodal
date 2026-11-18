"""Pruebas unitarias de la interfaz de usuario (Streamlit UI)."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from src.config import EMBEDDING_BACKENDS
from src.models import IndexStats, MultimodalQuery, SearchResult
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
            output_dimensionality=1536,
            target_width=640,
            enable_audio=False,
            whisper_model="small",
        )

    assert n == 3
    mock_service.index_video.assert_called_once()
    kwargs = mock_service.index_video.call_args.kwargs
    assert kwargs["max_concurrent"] == 4
    assert kwargs["threshold"] == 0.35
    assert kwargs["min_shot_frames"] == 30
    assert kwargs["entropy_stride"] == 2
    assert kwargs["jpeg_quality"] == 80
    assert kwargs["output_dimensionality"] == 1536
    assert kwargs["target_width"] == 640
    assert kwargs["enable_audio"] is False
    assert kwargs["whisper_model"] == "small"


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
    assert config["embedding_backend"] == "gemini"
    assert config["vector_size"] == 3072
    assert "chunking_defaults" in config
    assert "vectorizer_defaults" in config


def test_build_ui_run_config_records_index_stats() -> None:
    """Verifica que build_ui_run_config incluye estadísticas de indexación."""
    stats = IndexStats(
        backend="gemma2",
        device_label="Apple MPS (bfloat16)",
        dimension=768,
        seconds=12.5,
        chunks=10,
    )
    config = ui_app.build_ui_run_config(
        backend="gemma2",
        vector_size=768,
        index_stats=stats,
    )
    assert config["embedding_backend"] == "gemma2"
    assert config["vector_size"] == 768
    assert config["embedding_dimension"] == 768
    assert config["device"] == "Apple MPS (bfloat16)"
    assert config["index_seconds"] == 12.5


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


def test_build_ui_multimodal_query_combines_text_txt_images_and_audio() -> None:
    fake_txt = MagicMock()
    fake_txt.getvalue.return_value = b"archivo de texto"

    fake_img = MagicMock()
    fake_img.getvalue.return_value = b"fake-jpeg-bytes"

    fake_mic = MagicMock()
    fake_mic.getvalue.return_value = b"RIFF-mic"

    fake_audio_file = MagicMock()
    fake_audio_file.getvalue.return_value = b"ID3-mp3"

    mquery = ui_app.build_ui_multimodal_query(
        text_input="  consulta base ",
        uploaded_txts=[fake_txt],
        uploaded_images=[fake_img],
        recorded_audio=fake_mic,
        uploaded_audios=[fake_audio_file],
    )
    assert isinstance(mquery, MultimodalQuery)
    assert mquery.has_content
    assert mquery.text == "consulta base"
    assert mquery.txt_contents == ("archivo de texto",)
    assert mquery.images == (b"fake-jpeg-bytes",)
    assert mquery.audio_bytes == (b"RIFF-mic", b"ID3-mp3")


def test_execute_search_accepts_multimodal_query() -> None:
    fake_results = [
        SearchResult(
            score=0.91,
            video_name="animals.mp4",
            chunk_index=0,
            start_time=0.0,
            end_time=4.0,
        )
    ]
    mock_service = MagicMock()
    mock_service.search = AsyncMock(return_value=fake_results)

    mquery = MultimodalQuery(images=(b"fake-img",))
    results = asyncio.run(
        ui_app.execute_search(
            query=mquery,
            service=mock_service,
            limit=2,
            video_filter="animals.mp4",
        )
    )
    assert results == fake_results
    mock_service.search.assert_called_once_with(
        mquery,
        limit=2,
        video_filter="animals.mp4",
        rerank=True,
        auditor=None,
    )


def test_mrl_options_per_backend() -> None:
    assert ui_app.mrl_options("gemini") == (3072, 1536, 768)
    assert ui_app.mrl_options("qwen3-vl-2b") == (2048, 1536, 768)
    assert ui_app.mrl_options("gemma2") == (768, 512, 256)


def test_backend_ui_labels_cover_all_backends() -> None:
    assert set(ui_app.BACKEND_UI_LABELS) == set(EMBEDDING_BACKENDS)


def test_index_video_in_ui_forwards_embedding_backend(tmp_path: Path) -> None:
    dummy_video = tmp_path / "gemma.mp4"
    dummy_video.write_bytes(b"data")

    mock_service = MagicMock()
    mock_service.index_video = AsyncMock(return_value=4)

    with patch("src.ui.app.st.progress"):
        n = ui_app.index_video_in_ui(
            dummy_video,
            mock_service,
            embedding_backend="gemma2",
            output_dimensionality=768,
        )

    assert n == 4
    kwargs = mock_service.index_video.call_args.kwargs
    assert kwargs["embedding_backend"] == "gemma2"
    assert kwargs["output_dimensionality"] == 768


def test_save_embedding_config_writes_backend_and_stats() -> None:
    mock_auditor = MagicMock()
    mock_service = MagicMock()
    mock_service.embedding_backend = "qwen3-vl-2b"
    mock_service.vector_size = 2048
    mock_service.last_index_stats = IndexStats(
        backend="qwen3-vl-2b",
        device_label="Apple MPS (bfloat16)",
        dimension=2048,
        seconds=25.0,
        chunks=8,
    )

    ui_app.save_embedding_config(mock_auditor, mock_service)

    mock_auditor.save_config.assert_called_once()
    saved = mock_auditor.save_config.call_args.args[0]
    assert saved["embedding_backend"] == "qwen3-vl-2b"
    assert saved["device"] == "Apple MPS (bfloat16)"
    assert saved["index_seconds"] == 25.0


def _app_entrypoint() -> Path:
    return Path(__file__).resolve().parent.parent / "app.py"


def _build_mock_service(indexed: list[str]) -> MagicMock:
    service = MagicMock()
    service.embedding_backend = "gemini"
    service.vector_size = 3072
    service.last_index_stats = None

    def _set_embedding(backend: str, dim: int) -> None:
        service.embedding_backend = backend
        service.vector_size = dim

    service.set_embedding.side_effect = _set_embedding
    service.get_indexed_videos.return_value = list(indexed)
    return service


def test_app_renders_empty_state_and_opens_ingest_dialog(tmp_path: Path) -> None:
    """Verifica el estado vacío accionable y la apertura del diálogo de ingesta."""
    from streamlit.testing.v1 import AppTest

    mock_service = _build_mock_service([])
    mock_auditor = MagicMock()
    mock_auditor.run_dir = tmp_path

    with (
        patch("src.ui.app.get_rag_service", return_value=mock_service),
        patch("src.ui.app.get_auditor", return_value=mock_auditor),
        patch("src.ui.app.RAW_DIR", tmp_path),
    ):
        at = AppTest.from_file(_app_entrypoint(), default_timeout=15)
        at.run()
        assert not at.exception
        assert [t.value for t in at.title] == ["Video RAG Multimodal"]
        assert any("No hay vídeos indexados" in str(info.value) for info in at.info)

        ingest_btn = next(b for b in at.button if "Indexar primer vídeo" in b.label)
        ingest_btn.click().run()
        assert not at.exception


def test_app_search_displays_results_and_jumps_timestamp(tmp_path: Path) -> None:
    """Verifica búsqueda multimodal, tarjetas de hallazgos y salto de código de tiempo."""
    from streamlit.testing.v1 import AppTest

    from src.models import RerankedResult

    (tmp_path / "animals.mp4").write_bytes(b"fake-mp4")
    mock_service = _build_mock_service(["animals.mp4"])
    mock_service.last_resolved_query = None
    mock_service.search = AsyncMock(
        return_value=[
            RerankedResult(
                result=SearchResult(
                    score=0.89,
                    video_name="animals.mp4",
                    chunk_index=3,
                    start_time=24.0,
                    end_time=31.0,
                    transcript="el león ruge en la sabana",
                ),
                relevance_score=0.94,
                is_relevant=True,
                reasoning="Coincide visual y acústicamente.",
            )
        ]
    )
    mock_auditor = MagicMock()
    mock_auditor.run_dir = tmp_path

    with (
        patch("src.ui.app.get_rag_service", return_value=mock_service),
        patch("src.ui.app.get_auditor", return_value=mock_auditor),
        patch("src.ui.app.RAW_DIR", tmp_path),
    ):
        at = AppTest.from_file(_app_entrypoint(), default_timeout=15)
        at.run()
        assert not at.exception

        at.text_input[0].set_value("león rugiendo").run()
        search_btn = next(b for b in at.button if "Buscar evento" in b.label)
        search_btn.click().run()
        assert not at.exception
        assert any("00:24.0 - 00:31.0" in m.value for m in at.markdown)

        jump_btn = next(b for b in at.button if "Ir a 00:24.0" in b.label)
        jump_btn.click().run()
        assert not at.exception
        assert at.session_state["play_time"] == 24
        assert at.session_state["active_video"] == "animals.mp4"


def test_app_switches_backend_and_requires_maintenance_confirmation(tmp_path: Path) -> None:
    """Verifica cambio de motor por segmented_control y confirmación de borrado."""
    from streamlit.testing.v1 import AppTest

    (tmp_path / "animals.mp4").write_bytes(b"fake-mp4")
    mock_service = _build_mock_service(["animals.mp4"])
    mock_auditor = MagicMock()
    mock_auditor.run_dir = tmp_path

    with (
        patch("src.ui.app.get_rag_service", return_value=mock_service),
        patch("src.ui.app.get_auditor", return_value=mock_auditor),
        patch("src.ui.app.RAW_DIR", tmp_path),
    ):
        at = AppTest.from_file(_app_entrypoint(), default_timeout=15)
        at.run()
        assert not at.exception

        at.sidebar.segmented_control[0].set_value("gemma2").run()
        assert not at.exception
        assert mock_service.embedding_backend == "gemma2"
        assert mock_service.vector_size == 768

        maint_btn = next(
            b for b in at.sidebar.button if "Mantenimiento de base de datos" in b.label
        )
        maint_btn.click().run()
        assert not at.exception
        mock_service.clear_database.assert_not_called()




