"""Pruebas unitarias de ``audit.RunAuditor``."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pytest

from src.utils.audit import RunAuditor


@dataclass
class _Res:
    score: float
    start_time: float


def test_creates_run_dir_and_logs_all_levels(tmp_path: Path) -> None:
    with RunAuditor(tmp_path / "audit") as auditor:
        run_dir = auditor.run_dir
        logging.getLogger("src.processing.pipeline").debug("mensaje debug")
        logging.getLogger("src.embeddings.gemini_vectorizer").warning("mensaje warning")
    assert run_dir.parent == tmp_path / "audit"
    assert run_dir.name.startswith("run_")
    text = (run_dir / "execution.log").read_text(encoding="utf-8")
    assert "mensaje debug" in text and "mensaje warning" in text


def test_unique_dirs_same_second(tmp_path: Path) -> None:
    with RunAuditor(tmp_path) as a, RunAuditor(tmp_path) as b:
        assert a.run_dir != b.run_dir


def test_json_files(tmp_path: Path) -> None:
    with RunAuditor(tmp_path) as auditor:
        auditor.save_config({"video": "v.mp4", "path": Path("x")})
        auditor.save_search_results("un gol", [_Res(0.9, 1.5)])
        auditor.save_search_results("un pase", [_Res(0.7, 3.0)])
        run_dir = auditor.run_dir
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    assert config["video"] == "v.mp4"
    search = json.loads((run_dir / "search_results.json").read_text(encoding="utf-8"))
    assert search == {
        "searches": [
            {"query": "un gol", "results": [{"score": 0.9, "start_time": 1.5}]},
            {"query": "un pase", "results": [{"score": 0.7, "start_time": 3.0}]},
        ]
    }


def test_restores_logging_and_logs_exception(tmp_path: Path) -> None:
    root = logging.getLogger()
    handlers_before, level_before = list(root.handlers), root.level
    with pytest.raises(RuntimeError), RunAuditor(tmp_path) as auditor:
        run_dir = auditor.run_dir
        raise RuntimeError("fallo")
    assert root.handlers == handlers_before and root.level == level_before
    assert "fallo" in (run_dir / "execution.log").read_text(encoding="utf-8")


def test_run_dir_outside_context() -> None:
    with pytest.raises(RuntimeError):
        _ = RunAuditor().run_dir


def test_save_search_results_includes_modality_summary_when_provided(
    tmp_path: Path,
) -> None:
    summary = {
        "has_text": True,
        "txt_count": 1,
        "image_count": 2,
        "audio_count": 1,
        "audio_transcript_count": 1,
        "composed_text": "león en la sabana",
    }
    with RunAuditor(tmp_path) as auditor:
        auditor.save_search_results(
            "león en la sabana",
            [_Res(0.88, 4.0)],
            modality_summary=summary,
        )
        run_dir = auditor.run_dir

    search = json.loads((run_dir / "search_results.json").read_text(encoding="utf-8"))
    assert search["searches"][0]["modalities"] == summary

