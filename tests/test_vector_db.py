"""Pruebas unitarias de ``vector_db`` (Qdrant en memoria, sin red)."""

from pathlib import Path

import pytest

from src.models import VectorDBError, VectorizedChunk
from src.rag_engine.vector_db import QdrantSearchEngine


def _engine() -> QdrantSearchEngine:
    engine = QdrantSearchEngine(path=None)
    engine.create_collection(vector_size=3)
    return engine


def _chunks() -> list[VectorizedChunk]:
    return [
        VectorizedChunk(0, 0.0, 2.0, [1.0, 0.0, 0.0]),
        VectorizedChunk(1, 2.0, 5.0, [0.0, 1.0, 0.0]),
        VectorizedChunk(2, 5.0, 9.0, [0.0, 0.0, 1.0]),
    ]


def test_insert_and_search_returns_closest() -> None:
    engine = _engine()
    chunks = [
        VectorizedChunk(0, 0.0, 2.0, [1.0, 0.0, 0.0]),
        VectorizedChunk(1, 2.0, 5.0, [0.0, 1.0, 0.0]),
        VectorizedChunk(2, 5.0, 9.0, [0.0, 0.0, 1.0]),
    ]
    assert engine.insert_chunks(chunks, video_name="v.mp4") == 3
    results = engine.search_event([0.0, 0.9, 0.1], limit=2)
    assert len(results) == 2
    assert results[0].chunk_index == 1
    assert (results[0].start_time, results[0].end_time) == (2.0, 5.0)
    assert results[0].video_name == "v.mp4"
    assert results[0].score >= results[1].score


def test_reinsert_overwrites() -> None:
    engine = _engine()
    engine.insert_chunks(_chunks(), video_name="v.mp4")
    # Reinsertar con menos chunks no debe dejar chunks huérfanos
    engine.insert_chunks([_chunks()[0]], video_name="v.mp4")
    assert len(engine.search_event([1.0, 0.0, 0.0], limit=10)) == 1



def test_dimension_mismatch_raises() -> None:
    engine = _engine()
    with pytest.raises(VectorDBError):
        engine.insert_chunks([VectorizedChunk(0, 0.0, 1.0, [1.0, 2.0])], "v.mp4")


def test_invalid_params() -> None:
    engine = _engine()
    with pytest.raises(ValueError):
        engine.search_event([1.0, 0.0, 0.0], limit=0)
    with pytest.raises(ValueError):
        engine.create_collection(vector_size=0)


def test_local_persistence(tmp_path: Path) -> None:
    db = str(tmp_path / "db")
    first = QdrantSearchEngine(path=db)
    first.create_collection(vector_size=3)
    first.insert_chunks(_chunks(), "v.mp4")
    first._client.close()

    second = QdrantSearchEngine(path=db)
    assert len(second.search_event([1.0, 0.0, 0.0])) == 3


def test_get_indexed_videos_and_filtered_search() -> None:
    engine = _engine()
    assert engine.get_indexed_videos() == []

    engine.insert_chunks([VectorizedChunk(0, 0.0, 1.0, [1.0, 0.0, 0.0])], "video_a.mp4")
    engine.insert_chunks([VectorizedChunk(0, 0.0, 1.0, [0.0, 1.0, 0.0])], "video_b.mp4")

    videos = engine.get_indexed_videos()
    assert videos == ["video_a.mp4", "video_b.mp4"]

    # Búsqueda filtrada por vídeo específico
    results_a = engine.search_event([0.0, 1.0, 0.0], limit=5, video_name="video_a.mp4")
    assert len(results_a) == 1
    assert results_a[0].video_name == "video_a.mp4"

    # Búsqueda sin filtro devuelve ambos
    results_all = engine.search_event([0.0, 1.0, 0.0], limit=5)
    assert len(results_all) == 2


def test_delete_video() -> None:
    engine = _engine()
    engine.insert_chunks([VectorizedChunk(0, 0.0, 1.0, [1.0, 0.0, 0.0])], "video_a.mp4")
    engine.insert_chunks([VectorizedChunk(0, 0.0, 1.0, [0.0, 1.0, 0.0])], "video_b.mp4")
    assert engine.get_indexed_videos() == ["video_a.mp4", "video_b.mp4"]

    engine.delete_video("video_a.mp4")
    assert engine.get_indexed_videos() == ["video_b.mp4"]


def test_clear_database() -> None:
    engine = _engine()
    engine.insert_chunks(_chunks(), "v.mp4")
    assert len(engine.get_indexed_videos()) == 1

    engine.clear_database(vector_size=3)
    assert engine.get_indexed_videos() == []


def test_delete_video_and_clear_database_remove_cached_keyframes(tmp_path: Path) -> None:
    import numpy as np

    from src.processing.keyframe_cache import (
        load_chunk_keyframes_jpeg,
        save_video_keyframes,
    )

    processed_dir = tmp_path / "processed"
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    save_video_keyframes("video_a.mp4", {0: [frame, frame, frame]}, processed_dir=processed_dir)
    save_video_keyframes("video_b.mp4", {0: [frame, frame, frame]}, processed_dir=processed_dir)

    engine = QdrantSearchEngine(path=None, processed_dir=processed_dir)
    engine.create_collection(vector_size=3)
    engine.insert_chunks([VectorizedChunk(0, 0.0, 1.0, [1.0, 0.0, 0.0])], "video_a.mp4")
    engine.insert_chunks([VectorizedChunk(0, 0.0, 1.0, [0.0, 1.0, 0.0])], "video_b.mp4")

    # insert_chunks must not wipe newly cached keyframes
    assert len(load_chunk_keyframes_jpeg("video_a.mp4", 0, processed_dir=processed_dir)) == 3

    engine.delete_video("video_a.mp4")
    assert load_chunk_keyframes_jpeg("video_a.mp4", 0, processed_dir=processed_dir) == []
    assert len(load_chunk_keyframes_jpeg("video_b.mp4", 0, processed_dir=processed_dir)) == 3

    engine.clear_database(vector_size=3)
    assert load_chunk_keyframes_jpeg("video_b.mp4", 0, processed_dir=processed_dir) == []


def test_insert_and_search_persists_transcript() -> None:
    engine = _engine()
    chunks = [
        VectorizedChunk(
            0, 0.0, 3.0, [1.0, 0.0, 0.0], transcript="Explicamos diccionarios en Python."
        ),
        VectorizedChunk(1, 3.0, 6.0, [0.0, 1.0, 0.0], transcript=None),
    ]
    engine.insert_chunks(chunks, video_name="python_tutorial.mp4")

    results = engine.search_event([1.0, 0.1, 0.0], limit=2)
    assert len(results) == 2
    assert results[0].chunk_index == 0
    assert results[0].transcript == "Explicamos diccionarios en Python."
    assert results[1].chunk_index == 1
    assert results[1].transcript is None

