"""Pruebas unitarias de la factoría de vectorizadores (sin pesos ni red)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from src.embeddings import factory
from src.embeddings.local_vectorizer import LocalMultimodalVectorizer
from src.models import VectorizationError


class _Recorder:
    """Cuenta cargas de modelo y liberaciones de memoria."""

    def __init__(self) -> None:
        self.loaded: list[str] = []
        self.releases = 0

    def load(self, model_id: str, device: str) -> object:
        self.loaded.append(model_id)
        return object()

    def release(self) -> None:
        self.releases += 1


@pytest.fixture
def rec(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    recorder = _Recorder()
    monkeypatch.setattr(factory, "_active", None)
    monkeypatch.setattr(factory, "load_sentence_transformer", recorder.load)
    monkeypatch.setattr(factory, "_release_memory", recorder.release)
    monkeypatch.setattr(factory, "select_device", lambda: "mps")
    return recorder


def test_create_local_vectorizer_loads_model_once_for_different_dimensions(
    rec: _Recorder,
) -> None:
    first = factory.create_vectorizer("qwen3-vl-2b", 2048)
    second = factory.create_vectorizer("qwen3-vl-2b", 768)
    assert rec.loaded == ["Qwen/Qwen3-VL-Embedding-2B"]
    assert rec.releases == 0
    assert isinstance(first, LocalMultimodalVectorizer)
    assert isinstance(second, LocalMultimodalVectorizer)
    assert (first.backend, first.device_label) == ("qwen3-vl-2b", "Local MPS")


def test_switching_model_releases_previous_model(rec: _Recorder) -> None:
    factory.create_vectorizer("qwen3-vl-2b", 768)
    factory.create_vectorizer("gemma2", 768)
    assert rec.loaded == ["Qwen/Qwen3-VL-Embedding-2B", "google/embeddinggemma-2"]
    assert rec.releases == 1


def test_switching_back_and_forth_keeps_only_one_model(rec: _Recorder) -> None:
    for backend in ("qwen3-vl-2b", "gemma2", "qwen3-vl-2b", "gemma2"):
        factory.create_vectorizer(backend, 768)
        assert factory._active is not None
    assert len(rec.loaded) == 4
    assert rec.releases == 3
    assert factory._active is not None
    assert factory._active[0] == "google/embeddinggemma-2"


def test_create_gemini_unloads_local_model_and_returns_gemini(
    rec: _Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_cls = MagicMock()
    monkeypatch.setattr(factory, "GeminiMultimodalVectorizer", fake_cls)
    factory.create_vectorizer("qwen3-vl-2b", 768)
    result = factory.create_vectorizer("gemini", 1536, jpeg_quality=70, target_width=640)
    assert result is fake_cls.return_value
    fake_cls.assert_called_once_with(
        output_dimensionality=1536, jpeg_quality=70, target_width=640
    )
    assert rec.releases == 1
    assert factory._active is None


def test_unload_without_model_is_noop(rec: _Recorder) -> None:
    factory.unload_active_vectorizer()
    assert rec.releases == 0


def test_unload_releases_loaded_model(rec: _Recorder) -> None:
    factory.create_vectorizer("gemma2", 256)
    factory.unload_active_vectorizer()
    assert factory._active is None
    assert rec.releases == 1


@pytest.mark.parametrize(("backend", "dim"), [("gemma2", 3072), ("gemini", 2048), ("otro", 1)])
def test_invalid_choice_raises_value_error(rec: _Recorder, backend: str, dim: int) -> None:
    with pytest.raises(ValueError):
        factory.create_vectorizer(backend, dim)
    assert rec.loaded == []


def test_loader_failure_propagates_vectorization_error(
    rec: _Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing(model_id: str, device: str) -> Any:
        raise VectorizationError("sin red")

    monkeypatch.setattr(factory, "load_sentence_transformer", failing)
    with pytest.raises(VectorizationError):
        factory.create_vectorizer("gemma2", 768)
    assert factory._active is None


def test_device_label_for_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    assert factory.device_label_for("gemini") == "Cloud API"
    monkeypatch.setattr(factory, "select_device", lambda: "mps")
    assert factory.device_label_for("qwen3-vl-2b") == "Local MPS"
    monkeypatch.setattr(factory, "select_device", lambda: "cpu")
    assert factory.device_label_for("gemma2") == "Local CPU"
