"""Pruebas unitarias de la fachada de aplicación ``VideoRAGService``."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models import (
    RerankedResult,
    SearchResult,
    VectorizationError,
    VectorizedChunk,
)
from src.rag_engine.service import VideoRAGService, format_timestamp
from src.rag_engine.vector_db import QdrantSearchEngine
from src.utils.audit import RunAuditor


def test_format_timestamp() -> None:
    assert format_timestamp(0.0) == "00:00.0"
    assert format_timestamp(75.5) == "01:15.5"


def test_service_index_video_stores_vectors_in_qdrant(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    video_file = raw_dir / "demo.mp4"
    video_file.write_bytes(b"fake-mp4")

    engine = QdrantSearchEngine(path=None)
    service = VideoRAGService(engine=engine, raw_dir=raw_dir, vector_size=3)

    fake_chunks = [
        VectorizedChunk(0, 0.0, 2.0, [1.0, 0.0, 0.0]),
        VectorizedChunk(1, 2.0, 4.0, [0.0, 1.0, 0.0]),
    ]

    with patch(
        "src.rag_engine.service.process_video_pipeline",
        new_callable=AsyncMock,
        return_value=fake_chunks,
    ) as mock_pipe:
        inserted = asyncio.run(
            service.index_video("demo.mp4", max_concurrent=2, threshold=0.4)
        )

    assert inserted == 2
    mock_pipe.assert_called_once()
    assert service.get_indexed_videos() == ["demo.mp4"]

    with pytest.raises(FileNotFoundError):
        asyncio.run(service.index_video("missing.mp4"))


def test_service_search_two_stage_and_auditor(tmp_path: Path) -> None:
    engine = QdrantSearchEngine(path=None)
    engine.create_collection(vector_size=3)
    engine.insert_chunks(
        [
            VectorizedChunk(0, 0.0, 2.0, [1.0, 0.0, 0.0]),
            VectorizedChunk(1, 2.0, 5.0, [0.0, 1.0, 0.0]),
        ],
        video_name="animals.mp4",
    )

    fake_vectorizer = MagicMock()
    fake_vectorizer.embed_text = AsyncMock(return_value=[0.0, 1.0, 0.0])

    async def fake_rerank(
        query: str, candidates: list[SearchResult], **_: Any
    ) -> list[RerankedResult]:
        return [
            RerankedResult(
                result=candidates[0],
                relevance_score=0.95,
                is_relevant=True,
                reasoning="Validado por VLM.",
            )
        ]

    fake_reranker = MagicMock()
    fake_reranker.rerank = AsyncMock(side_effect=fake_rerank)

    service = VideoRAGService(
        engine=engine,
        vectorizer=fake_vectorizer,
        reranker=fake_reranker,
        raw_dir=tmp_path,
        vector_size=3,
    )

    with RunAuditor(tmp_path / "audit") as auditor:
        results = asyncio.run(
            service.search(
                "camaleón",
                limit=1,
                video_filter="animals.mp4",
                rerank=True,
                auditor=auditor,
            )
        )
        run_dir = auditor.run_dir

    assert len(results) == 1
    assert isinstance(results[0], RerankedResult)
    assert results[0].chunk_index == 1
    assert (run_dir / "search_results.json").is_file()


def test_service_search_falls_back_when_reranker_raises() -> None:
    engine = QdrantSearchEngine(path=None)
    engine.create_collection(vector_size=3)
    engine.insert_chunks(
        [VectorizedChunk(0, 0.0, 3.0, [1.0, 0.0, 0.0])],
        video_name="v.mp4",
    )

    fake_vectorizer = MagicMock()
    fake_vectorizer.embed_text = AsyncMock(return_value=[1.0, 0.0, 0.0])

    fake_reranker = MagicMock()
    fake_reranker.rerank = AsyncMock(side_effect=VectorizationError("cuota agotada"))

    service = VideoRAGService(
        engine=engine,
        vectorizer=fake_vectorizer,
        reranker=fake_reranker,
        vector_size=3,
    )

    results = asyncio.run(service.search("consulta", limit=1, rerank=True))
    assert len(results) == 1
    assert isinstance(results[0], SearchResult)
    assert results[0].chunk_index == 0


def test_service_search_oversamples_stage1_and_truncates_stage2() -> None:
    engine = QdrantSearchEngine(path=None)
    engine.create_collection(vector_size=3)
    chunks = [
        VectorizedChunk(i, float(i * 2), float(i * 2 + 2), [1.0, float(i) * 0.01, 0.0])
        for i in range(6)
    ]
    engine.insert_chunks(chunks, video_name="tutorial.mp4")

    fake_vectorizer = MagicMock()
    fake_vectorizer.embed_text = AsyncMock(return_value=[1.0, 0.0, 0.0])

    captured_candidates: list[SearchResult] = []

    async def fake_rerank(
        query: str, candidates: list[SearchResult], **_: Any
    ) -> list[RerankedResult]:
        captured_candidates.extend(candidates)
        return [
            RerankedResult(
                result=c,
                relevance_score=1.0 if c.chunk_index == 3 else 0.1,
                is_relevant=(c.chunk_index == 3),
                reasoning="Evaluado",
            )
            for c in reversed(candidates)
        ]

    fake_reranker = MagicMock()
    fake_reranker.rerank = AsyncMock(side_effect=fake_rerank)

    service = VideoRAGService(
        engine=engine,
        vectorizer=fake_vectorizer,
        reranker=fake_reranker,
        vector_size=3,
    )

    results = asyncio.run(service.search("bucles while", limit=2, rerank=True))
    assert len(captured_candidates) == 4
    assert len(results) == 2

    # Segunda consulta idéntica: reutiliza el embedding en caché sin llamar a la API
    asyncio.run(service.search("bucles while", limit=2, rerank=True))
    assert fake_vectorizer.embed_text.call_count == 1

    with pytest.raises(ValueError):
        asyncio.run(service.search("bucles", limit=2, oversample_factor=0))



def test_service_reuses_lazy_clients_and_maintains_db() -> None:
    engine = QdrantSearchEngine(path=None)
    engine.create_collection(vector_size=3)
    engine.insert_chunks(
        [VectorizedChunk(0, 0.0, 2.0, [1.0, 0.0, 0.0])],
        video_name="a.mp4",
    )
    engine.insert_chunks(
        [VectorizedChunk(0, 0.0, 2.0, [0.0, 1.0, 0.0])],
        video_name="b.mp4",
    )

    service = VideoRAGService(engine=engine, vector_size=3)

    with (
        patch("src.rag_engine.service.GeminiMultimodalVectorizer") as mock_vec_cls,
        patch("src.rag_engine.service.MultimodalReranker") as mock_rerank_cls,
    ):
        vec_instance = MagicMock()
        vec_instance.embed_text = AsyncMock(return_value=[1.0, 0.0, 0.0])
        mock_vec_cls.return_value = vec_instance

        rerank_instance = MagicMock()
        rerank_instance.rerank = AsyncMock(return_value=[])
        mock_rerank_cls.return_value = rerank_instance

        asyncio.run(service.search("q1", rerank=True))
        asyncio.run(service.search("q2", rerank=True))

        assert mock_vec_cls.call_count == 1
        assert mock_rerank_cls.call_count == 1

    service.delete_video("a.mp4")
    assert service.get_indexed_videos() == ["b.mp4"]

    service.clear_database()
    assert service.get_indexed_videos() == []

