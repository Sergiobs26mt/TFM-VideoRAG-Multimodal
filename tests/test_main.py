"""Pruebas unitarias de ``main`` (sin red ni Gemini)."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from src import main
from src.models import IndexStats, RerankedResult, SearchResult, VectorizationError
from src.utils.audit import RunAuditor


@pytest.fixture(autouse=True)
def _isolated_audit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Evita escribir auditorías de test en el audit/ real del proyecto."""
    monkeypatch.setattr(main, "AUDIT_DIR", tmp_path / "audit")


def test_format_timestamp() -> None:
    assert main.format_timestamp(75.5) == "01:15.5"
    assert main.format_timestamp(0.0) == "00:00.0"


def test_print_results(capsys: pytest.CaptureFixture[str]) -> None:
    main.print_results(
        [
            SearchResult(
                0.91234,
                "v.mp4",
                2,
                60.0,
                75.5,
                transcript="Locución de prueba sobre diccionarios",
            )
        ]
    )
    out = capsys.readouterr().out
    assert "01:00.0 - 01:15.5" in out and "0.9123" in out
    assert 'Transcripción: "Locución de prueba sobre diccionarios"' in out


def test_print_results_reranked(capsys: pytest.CaptureFixture[str]) -> None:
    res = SearchResult(
        0.35, "v.mp4", 1, 10.0, 20.0, transcript="Voz explicando el código"
    )
    reranked = RerankedResult(
        result=res,
        relevance_score=0.98,
        is_relevant=True,
        reasoning="Es un reptil válido.",
    )
    main.print_results([reranked])
    out = capsys.readouterr().out
    assert "VALIDADO" in out and "Es un reptil válido" in out
    assert 'Transcripción: "Voz explicando el código"' in out


def test_print_results_empty(capsys: pytest.CaptureFixture[str]) -> None:
    main.print_results([])
    assert "No se encontraron" in capsys.readouterr().out


def test_parse_args_defaults() -> None:
    args = main.parse_args([])
    assert not args.index and args.video == "test_video.mp4" and args.top_k == 5
    assert args.threshold == 0.5 and args.min_shot_frames == 12
    assert (
        args.entropy_stride == 1
        and args.jpeg_quality == 80
        and args.target_width == 512
    )
    assert args.rerank is True
    assert args.no_audio is False
    assert args.whisper_model == "base"
    assert args.embedding_backend == "gemini"
    assert args.output_dimensionality == 3072

    custom = main.parse_args(
        [
            "--no-audio",
            "--whisper-model",
            "small",
            "--output-dimensionality",
            "1536",
            "--query",
            "camaleón",
            "--query-txt",
            "a.txt",
            "--query-images",
            "img1.jpg",
            "img2.png",
            "--query-audios",
            "voz.wav",
        ]
    )
    assert custom.no_audio is True
    assert custom.whisper_model == "small"
    assert custom.output_dimensionality == 1536
    assert custom.query == "camaleón"
    assert custom.query_txt == ["a.txt"]
    assert custom.query_images == ["img1.jpg", "img2.png"]
    assert custom.query_audios == ["voz.wav"]


def _fake_inputs(monkeypatch: pytest.MonkeyPatch, lines: list[str]) -> None:
    it = iter(lines)
    monkeypatch.setattr("builtins.input", lambda _: next(it))


def test_loop_runs_until_exit_and_ignores_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    queries: list[str] = []

    async def fake_search(
        query: str, *, auditor: RunAuditor | None = None, **_: Any
    ) -> list[SearchResult]:
        queries.append(query)
        res = [SearchResult(0.9, "v.mp4", 1, 1.0, 2.0)]
        if auditor is not None:
            auditor.save_search_results(query, res)
        return res

    fake_service = MagicMock()
    fake_service.search = fake_search

    _fake_inputs(monkeypatch, ["un gol", "   ", "otra cosa", "SALIR"])
    with RunAuditor(tmp_path) as auditor:
        done = main.run_search_loop(fake_service, 5, auditor)
        run_dir = auditor.run_dir
    assert done == 2 and queries == ["un gol", "otra cosa"]
    data = json.loads((run_dir / "search_results.json").read_text(encoding="utf-8"))
    assert [s["query"] for s in data["searches"]] == ["un gol", "otra cosa"]


def test_loop_survives_failed_query(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = {"n": 0}

    async def flaky_search(query: str, **_: Any) -> list[SearchResult]:
        calls["n"] += 1
        if calls["n"] == 1:
            raise VectorizationError("cuota agotada")
        return []

    fake_service = MagicMock()
    fake_service.search = flaky_search

    _fake_inputs(monkeypatch, ["primera", "segunda", "q"])
    with RunAuditor(tmp_path) as auditor:
        done = main.run_search_loop(fake_service, 5, auditor)
    assert calls["n"] == 2 and done == 1


@pytest.mark.parametrize("exc", [EOFError, KeyboardInterrupt])
def test_loop_ends_on_ctrl_c_or_ctrl_d(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, exc: type[BaseException]
) -> None:
    def raise_exc(_: str) -> str:
        raise exc

    monkeypatch.setattr("builtins.input", raise_exc)
    with RunAuditor(tmp_path) as auditor:
        assert main.run_search_loop(MagicMock(), 5, auditor) == 0


def test_main_exits_cleanly_on_quit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "VideoRAGService", lambda **_: MagicMock())
    _fake_inputs(monkeypatch, ["exit"])
    assert main.main([]) == 0


def test_main_missing_video(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.rag_engine.service import VideoRAGService
    from src.rag_engine.vector_db import QdrantSearchEngine

    monkeypatch.setattr(
        main,
        "VideoRAGService",
        lambda **kw: VideoRAGService(engine=QdrantSearchEngine(path=None), **kw),
    )
    assert main.main(["--index", "--video", "no_existe.mp4"]) == 1


def test_main_runs_benchmark_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from unittest.mock import AsyncMock

    gt_file = tmp_path / "gt.json"
    gt_file.write_text(
        json.dumps(
            [
                {
                    "query": "reptil",
                    "video_name": "animals.mp4",
                    "expected_start": 20.0,
                    "expected_end": 30.0,
                }
            ]
        ),
        encoding="utf-8",
    )

    fake_service = MagicMock()
    fake_service.search = AsyncMock(
        return_value=[SearchResult(0.9, "animals.mp4", 1, 24.0, 29.0)]
    )
    monkeypatch.setattr(main, "VideoRAGService", lambda **_: fake_service)

    assert main.main(["--benchmark", str(gt_file), "--no-rerank"]) == 0
    out = capsys.readouterr().out
    assert "Recall@1" in out and "MRR" in out


def test_main_runs_one_shot_multimodal_search(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from unittest.mock import AsyncMock

    fake_service = MagicMock()
    fake_service.search = AsyncMock(
        return_value=[SearchResult(0.93, "animals.mp4", 4, 12.0, 16.0)]
    )
    monkeypatch.setattr(main, "VideoRAGService", lambda **_: fake_service)

    rc = main.main(
        [
            "--query-txt",
            "nota.txt",
            "--query-images",
            "foto.jpg",
            "--query-audios",
            "clip.wav",
            "--top-k",
            "2",
        ]
    )
    assert rc == 0
    fake_service.search.assert_called_once()
    kwargs = fake_service.search.call_args.kwargs
    assert kwargs["txt_files"] == ["nota.txt"]
    assert kwargs["image_files"] == ["foto.jpg"]
    assert kwargs["audio_files"] == ["clip.wav"]
    assert kwargs["limit"] == 2
    out = capsys.readouterr().out
    assert "animals.mp4" in out


def test_parse_args_resolves_default_dimension_per_backend() -> None:
    gemma_args = main.parse_args(["--embedding-backend", "gemma2"])
    qwen_args = main.parse_args(["--embedding-backend", "qwen3-vl-2b"])
    assert gemma_args.output_dimensionality == 768
    assert qwen_args.output_dimensionality == 2048


def test_parse_args_rejects_dimension_not_valid_for_backend() -> None:
    with pytest.raises(SystemExit):
        main.parse_args(
            ["--embedding-backend", "gemma2", "--output-dimensionality", "1536"]
        )


def test_build_run_config_includes_embedding_backend() -> None:
    args = main.parse_args(["--embedding-backend", "qwen3-vl-2b", "--output-dimensionality", "1536"])
    stats = IndexStats("qwen3-vl-2b", "Local MPS", 1536, 9.2, 20)
    cfg = main.build_run_config(args, index_stats=stats)
    assert cfg["embedding_backend"] == "qwen3-vl-2b"
    assert cfg["embedding_dimension"] == 1536
    assert cfg["device"] == "Local MPS"
    assert cfg["index_seconds"] == 9.2


def test_main_saves_config_again_with_index_stats_after_indexing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from unittest.mock import AsyncMock

    fake_service = MagicMock()
    fake_service.index_video = AsyncMock(return_value=10)
    fake_service.last_index_stats = IndexStats("gemma2", "Local MPS", 768, 4.5, 10)
    fake_service.search = AsyncMock(return_value=[])
    service_kwargs: list[dict[str, Any]] = []

    def fake_service_cls(**kw: Any) -> MagicMock:
        service_kwargs.append(kw)
        return fake_service

    monkeypatch.setattr(main, "VideoRAGService", fake_service_cls)
    saved_configs: list[dict[str, Any]] = []
    orig_save = RunAuditor.save_config

    def spy_save(self: RunAuditor, config: dict[str, Any]) -> Path:
        saved_configs.append(config)
        return orig_save(self, config)

    monkeypatch.setattr(RunAuditor, "save_config", spy_save)

    rc = main.main(
        [
            "--index",
            "--video",
            "demo.mp4",
            "--embedding-backend",
            "gemma2",
            "--query",
            "hola",
        ]
    )
    assert rc == 0
    assert service_kwargs[0]["embedding_backend"] == "gemma2"
    assert service_kwargs[0]["vector_size"] == 768
    assert fake_service.index_video.call_args.kwargs["embedding_backend"] == "gemma2"
    assert len(saved_configs) == 2
    assert saved_configs[0]["index_seconds"] is None
    assert saved_configs[1]["embedding_backend"] == "gemma2"
    assert saved_configs[1]["device"] == "Local MPS"
    assert saved_configs[1]["index_seconds"] == 4.5


def test_main_warns_when_max_concurrent_ignored_for_local_backend(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from unittest.mock import AsyncMock

    fake_service = MagicMock()
    fake_service.index_video = AsyncMock(return_value=2)
    fake_service.last_index_stats = IndexStats("gemma2", "Local MPS", 768, 1.0, 2)
    fake_service.search = AsyncMock(return_value=[])
    monkeypatch.setattr(main, "VideoRAGService", lambda **_: fake_service)

    rc = main.main(
        [
            "--index",
            "--embedding-backend",
            "gemma2",
            "--max-concurrent",
            "4",
            "--query",
            "hola",
        ]
    )
    assert rc == 0
    err = capsys.readouterr().err.lower()
    assert "concurrencia" in err and "max-concurrent" in err

