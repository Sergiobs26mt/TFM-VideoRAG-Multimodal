"""Pruebas unitarias de ``gemini_vectorizer`` (API simulada, sin red)."""

import asyncio
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from google.genai import errors

from src.embeddings.gemini_vectorizer import GeminiMultimodalVectorizer
from src.models import MultimodalQuery, VectorizationError, VideoChunk
from src.utils import gemini_client as gc

CHUNK = VideoChunk(index=3, start_frame=0, end_frame=9, start_time=1.5, end_time=4.0)


def _frames() -> list[Any]:
    rng = np.random.default_rng(0)
    return [rng.integers(0, 256, (32, 32, 3), dtype=np.uint8) for _ in range(3)]


def _response(values: list[float]) -> Any:
    return SimpleNamespace(embeddings=[SimpleNamespace(values=values)])


def _make(
    monkeypatch: pytest.MonkeyPatch, fake_embed: Any, **kw: Any
) -> GeminiMultimodalVectorizer:
    monkeypatch.setenv("GEMINI_API_KEY", "clave-falsa")
    monkeypatch.setattr(gc, "load_dotenv", lambda: None)
    vec = GeminiMultimodalVectorizer(retry_base_delay=0.0, **kw)
    vec._client = SimpleNamespace(  # type: ignore[assignment]
        aio=SimpleNamespace(models=SimpleNamespace(embed_content=fake_embed))
    )
    return vec


def test_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(gc, "load_dotenv", lambda: None)
    with pytest.raises(ValueError):
        GeminiMultimodalVectorizer()


def test_payload_has_one_part_per_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    vec = _make(monkeypatch, None)
    content = vec.build_payload(_frames())
    assert content.parts is not None and len(content.parts) == 3
    assert content.parts[0].inline_data is not None
    assert content.parts[0].inline_data.mime_type == "image/jpeg"


def test_payload_includes_transcript_part_when_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vec = _make(monkeypatch, None)
    content = vec.build_payload(_frames(), transcript="  explicamos diccionarios  ")
    assert content.parts is not None and len(content.parts) == 4
    assert content.parts[3].text == (
        "Transcripción de audio del fragmento: explicamos diccionarios"
    )

    blank_content = vec.build_payload(_frames(), transcript="   ")
    assert blank_content.parts is not None and len(blank_content.parts) == 3


def test_vectorize_chunk_propagates_transcript_to_payload_and_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    async def fake(**kwargs: Any) -> Any:
        seen.update(kwargs)
        return _response([0.1, 0.2, 0.3])

    chunk_with_audio = VideoChunk(
        index=3,
        start_frame=0,
        end_frame=9,
        start_time=1.5,
        end_time=4.0,
        transcript="clave y valor en dict",
    )
    result = asyncio.run(
        _make(monkeypatch, fake).vectorize_chunk(chunk_with_audio, _frames())
    )
    assert result.transcript == "clave y valor en dict"
    sent_parts = seen["contents"].parts
    assert len(sent_parts) == 4
    assert sent_parts[3].text == (
        "Transcripción de audio del fragmento: clave y valor en dict"
    )


def test_empty_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError):
        _make(monkeypatch, None).build_payload([])


def test_vectorize_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake(**_: Any) -> Any:
        return _response([0.1, 0.2, 0.3])

    result = asyncio.run(_make(monkeypatch, fake).vectorize_chunk(CHUNK, _frames()))
    assert (result.index, result.start_time, result.end_time) == (3, 1.5, 4.0)
    assert result.embedding == [0.1, 0.2, 0.3]


def test_retry_then_success(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    async def fake(**_: Any) -> Any:
        calls["n"] += 1
        if calls["n"] < 3:
            raise errors.APIError(429, {"error": {"message": "cuota"}})
        return _response([1.0])

    result = asyncio.run(_make(monkeypatch, fake).vectorize_chunk(CHUNK, _frames()))
    assert calls["n"] == 3 and result.embedding == [1.0]


def test_non_retryable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake(**_: Any) -> Any:
        raise errors.APIError(400, {"error": {"message": "mala petición"}})

    with pytest.raises(VectorizationError):
        asyncio.run(_make(monkeypatch, fake).vectorize_chunk(CHUNK, _frames()))


def test_retries_exhausted(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake(**_: Any) -> Any:
        raise OSError("red caída")

    with pytest.raises(VectorizationError):
        asyncio.run(
            _make(monkeypatch, fake, max_retries=1).vectorize_chunk(CHUNK, _frames())
        )


def test_normalizes_truncated_dimensions(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake(**_: Any) -> Any:
        return _response([3.0, 4.0])

    vec = _make(monkeypatch, fake, output_dimensionality=768)
    result = asyncio.run(vec.vectorize_chunk(CHUNK, _frames()))
    assert result.embedding == pytest.approx([0.6, 0.8])


def test_embed_text(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    async def fake(**kwargs: Any) -> Any:
        seen.update(kwargs)
        return _response([0.5, 0.5])

    vec = _make(monkeypatch, fake)
    assert asyncio.run(vec.embed_text("un gol")) == [0.5, 0.5]
    assert seen["contents"].parts is not None
    assert len(seen["contents"].parts) == 1
    assert seen["contents"].parts[0].text == "un gol"


def test_embed_text_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError):
        asyncio.run(_make(monkeypatch, None).embed_text("   "))


def test_build_query_payload_and_embed_query_multimodal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.processing.video_chunker import encode_frame_jpeg

    seen: dict[str, Any] = {}

    async def fake(**kwargs: Any) -> Any:
        seen.update(kwargs)
        return _response([0.2, 0.4, 0.6])

    vec = _make(monkeypatch, fake, target_width=256, jpeg_quality=80)
    raw_img = encode_frame_jpeg(np.zeros((400, 600, 3), dtype=np.uint8))

    mquery = MultimodalQuery(
        text="busca este animal",
        txt_contents=("en su hábitat",),
        images=(raw_img,),
        audio_transcripts=("caminando despacio",),
    )

    vector = asyncio.run(vec.embed_query(mquery))
    assert vector == [0.2, 0.4, 0.6]
    parts = seen["contents"].parts
    assert parts is not None and len(parts) == 2
    assert parts[0].inline_data is not None
    assert parts[0].inline_data.mime_type == "image/jpeg"
    assert parts[1].text == "busca este animal en su hábitat caminando despacio"


def test_build_query_payload_rejects_empty_or_corrupt_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vec = _make(monkeypatch, None)
    with pytest.raises(ValueError):
        vec.build_query_payload(MultimodalQuery(text="  "))

    with pytest.raises(ValueError):
        vec.build_query_payload(MultimodalQuery(images=(b"not-an-image",)))


def test_payload_resizes_large_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    import cv2

    vec = _make(monkeypatch, None, target_width=512, jpeg_quality=80)
    large_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    content = vec.build_payload([large_frame])
    assert content.parts is not None and len(content.parts) == 1
    part_bytes = content.parts[0].inline_data.data  # type: ignore[union-attr]
    assert isinstance(part_bytes, bytes)
    decoded = cv2.imdecode(np.frombuffer(part_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape[1] == 512
    assert decoded.shape[0] == 288


def test_gemini_exposes_backend_and_device_label(monkeypatch: pytest.MonkeyPatch) -> None:
    vec = _make(monkeypatch, None)
    assert vec.backend == "gemini"
    assert vec.device_label == "Cloud API"
