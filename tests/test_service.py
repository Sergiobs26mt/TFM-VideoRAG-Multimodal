"""Pruebas unitarias de la fachada de aplicación ``VideoRAGService``."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models import (
    MultimodalQuery,
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
            service.index_video(
                "demo.mp4",
                max_concurrent=2,
                threshold=0.4,
                enable_audio=False,
                whisper_model="small",
            )
        )

    assert inserted == 2
    mock_pipe.assert_called_once()
    assert mock_pipe.call_args.kwargs["enable_audio"] is False
    assert mock_pipe.call_args.kwargs["whisper_model"] == "small"
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


def test_service_search_deduplicates_micro_chunks_from_same_scene() -> None:
    engine = QdrantSearchEngine(path=None)
    engine.create_collection(vector_size=3)
    # 5 micro-chunks de 0.4s en la escena 96-98s, más otros 2 planos largos en 14-18s y 124-130s
    chunks = [
        VectorizedChunk(28, 96.0, 96.4, [1.0, 0.01, 0.0]),
        VectorizedChunk(29, 96.4, 96.8, [1.0, 0.02, 0.0]),
        VectorizedChunk(30, 96.8, 97.2, [1.0, 0.03, 0.0]),
        VectorizedChunk(31, 97.2, 97.6, [1.0, 0.04, 0.0]),
        VectorizedChunk(32, 97.6, 98.0, [1.0, 0.05, 0.0]),
        VectorizedChunk(4, 14.0, 18.0, [0.9, 0.3, 0.0]),
        VectorizedChunk(26, 124.0, 130.0, [0.85, 0.4, 0.0]),
    ]
    engine.insert_chunks(chunks, video_name="animals.mp4")

    fake_vectorizer = MagicMock()
    fake_vectorizer.embed_text = AsyncMock(return_value=[1.0, 0.0, 0.0])

    service = VideoRAGService(
        engine=engine,
        vectorizer=fake_vectorizer,
        vector_size=3,
    )

    results = asyncio.run(
        service.search("Manada de animales", limit=3, rerank=False)
    )
    returned_indices = [r.chunk_index for r in results]
    assert 4 in returned_indices
    assert 26 in returned_indices
    assert results[0].start_time == pytest.approx(96.0)
    assert results[0].end_time == pytest.approx(98.0)


def test_service_search_preserves_distinct_short_shots_over_one_second() -> None:
    engine = QdrantSearchEngine(path=None)
    engine.create_collection(vector_size=3)
    chunks = [
        VectorizedChunk(21, 108.1, 109.03, [1.0, 0.0, 0.0]),
        VectorizedChunk(22, 109.03, 110.2, [1.0, 0.02, 0.0]),
    ]
    engine.insert_chunks(chunks, video_name="python_tutorial.mp4")

    fake_vectorizer = MagicMock()
    fake_vectorizer.embed_text = AsyncMock(return_value=[1.0, 0.0, 0.0])

    service = VideoRAGService(
        engine=engine,
        vectorizer=fake_vectorizer,
        vector_size=3,
    )

    results = asyncio.run(
        service.search("Plano en blanco y negro", limit=2, rerank=False)
    )
    assert [r.chunk_index for r in results] == [21, 22]


def test_service_search_deduplicates_overlapping_transcripts_across_merged_micro_chunks() -> None:
    engine = QdrantSearchEngine(path=None)
    engine.create_collection(vector_size=3)
    # 3 micro-chunks de 0.4s en la misma escena donde la frase fronteriza
    # "Hoy veremos diccionarios." fue asignada a chunks adyacentes por solapamiento.
    chunks = [
        VectorizedChunk(
            10,
            60.0,
            60.4,
            [1.0, 0.01, 0.0],
            transcript="Pasamos al código. Hoy veremos diccionarios.",
        ),
        VectorizedChunk(
            11,
            60.4,
            60.8,
            [1.0, 0.05, 0.0],
            transcript="Hoy veremos diccionarios. Tienen clave y valor.",
        ),
        VectorizedChunk(
            12,
            60.8,
            61.2,
            [1.0, 0.02, 0.0],
            transcript="Tienen clave y valor.",
        ),
    ]
    engine.insert_chunks(chunks, video_name="python_tutorial.mp4")

    fake_vectorizer = MagicMock()
    fake_vectorizer.embed_text = AsyncMock(return_value=[1.0, 0.05, 0.0])

    service = VideoRAGService(
        engine=engine,
        vectorizer=fake_vectorizer,
        vector_size=3,
    )

    results = asyncio.run(
        service.search("Diccionarios", limit=3, rerank=False)
    )
    assert len(results) == 1
    assert results[0].start_time == pytest.approx(60.0)
    assert results[0].end_time == pytest.approx(61.2)
    assert results[0].transcript == (
        "Pasamos al código. Hoy veremos diccionarios. Tienen clave y valor."
    )


def test_service_search_deduplicates_adjacent_chunks_with_identical_transcript() -> None:
    engine = QdrantSearchEngine(path=None)
    engine.create_collection(vector_size=3)
    shared_tr = (
        "Cuentan con un complejo sistema nervioso, además de órganos de los sentidos."
    )
    chunks = [
        VectorizedChunk(
            66,
            193.8,
            197.7,
            [1.0, 0.0, 0.0],
            transcript="Destaca el sistema respiratorio pulmonar de los mamíferos.",
        ),
        VectorizedChunk(48, 118.8, 120.5, [0.98, 0.02, 0.0], transcript=shared_tr),
        VectorizedChunk(44, 115.3, 116.5, [0.97, 0.03, 0.0], transcript=shared_tr),
        VectorizedChunk(46, 117.8, 118.8, [0.96, 0.04, 0.0], transcript=shared_tr),
        VectorizedChunk(45, 116.5, 117.8, [0.95, 0.05, 0.0], transcript=shared_tr),
        VectorizedChunk(
            12,
            18.9,
            24.1,
            [0.90, 0.10, 0.0],
            transcript="El reino animalia es uno de los cinco reinos.",
        ),
    ]
    engine.insert_chunks(chunks, video_name="animals.mp4")

    fake_vectorizer = MagicMock()
    fake_vectorizer.embed_text = AsyncMock(return_value=[1.0, 0.0, 0.0])

    service = VideoRAGService(
        engine=engine,
        vectorizer=fake_vectorizer,
        vector_size=3,
    )

    results = asyncio.run(
        service.search("Felinos", limit=3, rerank=False)
    )
    assert [r.chunk_index for r in results] == [66, 48, 12]
    assert results[1].start_time == pytest.approx(115.3)
    assert results[1].end_time == pytest.approx(120.5)
    assert results[1].transcript == shared_tr


def test_service_search_multimodal_query_transcribes_audio_and_passes_images(
    tmp_path: Path,
) -> None:
    engine = QdrantSearchEngine(path=None)
    engine.create_collection(vector_size=3)
    engine.insert_chunks(
        [VectorizedChunk(0, 0.0, 4.0, [1.0, 0.0, 0.0], transcript="camaleón verde")],
        video_name="animals.mp4",
    )

    txt_file = tmp_path / "extra.txt"
    txt_file.write_text("en rama", encoding="utf-8")
    img_file = tmp_path / "ref.jpg"
    img_file.write_bytes(b"fake-jpg-bytes")
    audio_file = tmp_path / "voice.wav"
    audio_file.write_bytes(b"RIFF-audio-1")

    seen_queries: list[MultimodalQuery] = []

    async def fake_embed_query(q: str | MultimodalQuery) -> list[float]:
        assert isinstance(q, MultimodalQuery)
        seen_queries.append(q)
        return [1.0, 0.0, 0.0]

    fake_vectorizer = MagicMock()
    fake_vectorizer.embed_query = AsyncMock(side_effect=fake_embed_query)

    seen_rerank_kwargs: dict[str, Any] = {}

    async def fake_rerank(
        query: str, candidates: list[SearchResult], **kwargs: Any
    ) -> list[RerankedResult]:
        seen_rerank_kwargs["query"] = query
        seen_rerank_kwargs.update(kwargs)
        return [
            RerankedResult(
                result=candidates[0],
                relevance_score=0.97,
                is_relevant=True,
                reasoning="Coincide con imagen y audio.",
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

    with (
        patch(
            "src.rag_engine.service.transcribe_query_audios",
            return_value=["locución hablada"],
        ) as mock_transcribe,
        RunAuditor(tmp_path / "audit") as auditor,
    ):
        results = asyncio.run(
            service.search(
                "camaleón",
                limit=1,
                rerank=True,
                auditor=auditor,
                txt_files=[txt_file],
                image_files=[img_file],
                audio_files=[audio_file],
            )
        )
        run_dir = auditor.run_dir

    assert len(results) == 1
    mock_transcribe.assert_called_once()
    assert len(seen_queries) == 1
    assert seen_queries[0].composed_text == "camaleón en rama locución hablada"
    assert seen_queries[0].images == (b"fake-jpg-bytes",)
    assert seen_rerank_kwargs["query"] == "camaleón en rama locución hablada"
    assert seen_rerank_kwargs["query_images"] == (b"fake-jpg-bytes",)

    import json

    saved = json.loads((run_dir / "search_results.json").read_text(encoding="utf-8"))
    assert saved["searches"][0]["modalities"]["image_count"] == 1
    assert saved["searches"][0]["modalities"]["audio_count"] == 1
    assert saved["searches"][0]["modalities"]["txt_count"] == 1


def test_service_search_cache_distinguishes_different_audio_and_images() -> None:
    engine = QdrantSearchEngine(path=None)
    engine.create_collection(vector_size=3)
    engine.insert_chunks(
        [VectorizedChunk(0, 0.0, 4.0, [1.0, 0.0, 0.0])],
        video_name="animals.mp4",
    )

    fake_vectorizer = MagicMock()
    fake_vectorizer.embed_query = AsyncMock(return_value=[1.0, 0.0, 0.0])

    service = VideoRAGService(
        engine=engine,
        vectorizer=fake_vectorizer,
        vector_size=3,
    )

    transcripts_iter = iter([["primera nota de voz"], ["segunda nota de voz"]])
    with patch(
        "src.rag_engine.service.transcribe_query_audios",
        side_effect=lambda *_, **__: next(transcripts_iter),
    ):
        q1 = MultimodalQuery(audio_bytes=(b"audio-bytes-1",))
        q2 = MultimodalQuery(audio_bytes=(b"audio-bytes-2",))
        asyncio.run(service.search(q1, rerank=False))
        asyncio.run(service.search(q2, rerank=False))

    assert fake_vectorizer.embed_query.call_count == 2


def test_service_search_raises_value_error_when_audio_only_query_is_silent() -> None:
    engine = QdrantSearchEngine(path=None)
    engine.create_collection(vector_size=3)
    fake_vectorizer = MagicMock()
    fake_vectorizer.embed_query = AsyncMock(return_value=[1.0, 0.0, 0.0])

    service = VideoRAGService(
        engine=engine,
        vectorizer=fake_vectorizer,
        vector_size=3,
    )

    with (
        patch("src.rag_engine.service.transcribe_query_audios", return_value=[]),
        pytest.raises(ValueError, match="silencio|locución"),
    ):
        asyncio.run(
            service.search(
                MultimodalQuery(audio_bytes=(b"RIFF-silent",)),
                rerank=False,
            )
        )
    fake_vectorizer.embed_query.assert_not_called()


def test_service_set_vector_size_updates_vectorizer_and_cache_key() -> None:
    engine = QdrantSearchEngine(path=None)
    service = VideoRAGService(engine=engine, vector_size=3072)
    assert service.vector_size == 3072

    with patch("src.rag_engine.service.GeminiMultimodalVectorizer") as mock_vec_cls:
        vec_3072 = MagicMock()
        vec_3072.embed_query = AsyncMock(return_value=[1.0, 0.0, 0.0])
        vec_1536 = MagicMock()
        vec_1536.embed_query = AsyncMock(return_value=[1.0, 0.0])
        mock_vec_cls.side_effect = [vec_3072, vec_1536]

        engine.create_collection(vector_size=3)
        engine.insert_chunks([VectorizedChunk(0, 0.0, 2.0, [1.0, 0.0, 0.0])], "v.mp4")
        asyncio.run(service.search("hola", rerank=False))

        service.set_vector_size(1536)
        assert service.vector_size == 1536
        engine.create_collection(vector_size=2, recreate=True)
        engine.insert_chunks([VectorizedChunk(0, 0.0, 2.0, [1.0, 0.0])], "v.mp4")
        asyncio.run(service.search("hola", rerank=False))

        assert mock_vec_cls.call_count == 2
        assert mock_vec_cls.call_args_list[0].kwargs["output_dimensionality"] == 3072
        assert mock_vec_cls.call_args_list[1].kwargs["output_dimensionality"] == 1536






