"""Pruebas unitarias del cliente y transporte asíncrono de Gemini."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import httpcore
import pytest
from google.genai import errors

from src.models import VectorizationError
from src.utils import gemini_client as gc
from src.utils.gemini_client import call_with_retry, create_gemini_client


def test_create_gemini_client_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(gc, "load_dotenv", lambda: None)
    with pytest.raises(ValueError):
        create_gemini_client()


def test_create_gemini_client_uses_explicit_or_env_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gc, "load_dotenv", lambda: None)
    with patch("src.utils.gemini_client.genai.Client") as mock_cls:
        mock_cls.return_value = MagicMock()
        create_gemini_client(api_key="explicit-key")
        assert mock_cls.call_count == 1
        assert mock_cls.call_args.kwargs["api_key"] == "explicit-key"


def test_create_gemini_client_survives_multiple_asyncio_run_loops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gc, "load_dotenv", lambda: None)

    class _LoopBoundStream(httpcore.AsyncNetworkStream):
        def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
            self._loop = loop
            self._chunks = [
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}"
            ]

        async def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
            return self._chunks.pop(0) if self._chunks else b""

        async def write(self, buffer: bytes, timeout: float | None = None) -> None:
            return None

        async def aclose(self) -> None:
            if self._loop.is_closed():
                raise RuntimeError("Event loop is closed")

        async def start_tls(self, *args: Any, **kwargs: Any) -> httpcore.AsyncNetworkStream:
            return self

    class _LoopBoundBackend(httpcore.AsyncNetworkBackend):
        async def connect_tcp(self, *args: Any, **kwargs: Any) -> httpcore.AsyncNetworkStream:
            return _LoopBoundStream(asyncio.get_running_loop())

    client = create_gemini_client(api_key="test-key")
    async_httpx: Any = client._api_client._async_httpx_client
    pool: Any = async_httpx._transport._pool
    pool._network_backend = _LoopBoundBackend()

    resp1 = asyncio.run(async_httpx.get("https://generativelanguage.googleapis.com/"))
    assert resp1.status_code == 200
    for conn in pool.connections:
        conn._keepalive_expiry = 0.0

    resp2 = asyncio.run(async_httpx.get("https://generativelanguage.googleapis.com/"))
    assert resp2.status_code == 200
    assert pool.connections == []


def test_call_with_retry_succeeds_after_transient_errors() -> None:
    attempts = 0

    async def flaky() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise errors.APIError(429, {"error": {"message": "rate limit"}})
        if attempts == 2:
            raise OSError("timeout")
        return "ok"

    res = asyncio.run(
        call_with_retry(flaky, label="test-op", max_retries=3, retry_base_delay=0.0)
    )
    assert res == "ok"
    assert attempts == 3


def test_call_with_retry_aborts_on_non_retryable_api_error() -> None:
    attempts = 0

    async def bad_request() -> str:
        nonlocal attempts
        attempts += 1
        raise errors.APIError(400, {"error": {"message": "invalid"}})

    with pytest.raises(VectorizationError, match="400"):
        asyncio.run(
            call_with_retry(
                bad_request, label="bad-op", max_retries=3, retry_base_delay=0.0
            )
        )
    assert attempts == 1


def test_call_with_retry_raises_when_exhausted() -> None:
    async def always_down() -> str:
        raise OSError("network unreachable")

    with pytest.raises(VectorizationError, match="Se agotaron los reintentos"):
        asyncio.run(
            call_with_retry(
                always_down, label="down-op", max_retries=1, retry_base_delay=0.0
            )
        )

