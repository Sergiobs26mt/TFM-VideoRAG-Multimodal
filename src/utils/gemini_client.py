"""Cliente compartido y transporte asíncrono con reintentos para la API de Gemini."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from typing import TypeVar

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from src.models import VectorizationError

logger = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})

T = TypeVar("T")


def create_gemini_client(api_key: str | None = None) -> genai.Client:
    """Inicializa el cliente de Google GenAI validando ``GEMINI_API_KEY``."""
    load_dotenv()
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError("La variable de entorno GEMINI_API_KEY no está definida.")
    # Desactiva keep-alive en httpx asíncrono para que las conexiones se cierren
    # en el bucle de eventos activo antes de que finalice cada llamada a asyncio.run().
    http_options = types.HttpOptions(
        async_client_args={"limits": httpx.Limits(max_keepalive_connections=0)}
    )
    return genai.Client(api_key=key, http_options=http_options)


async def call_with_retry(
    operation: Callable[[], Awaitable[T]],
    *,
    label: str,
    max_retries: int,
    retry_base_delay: float,
) -> T:
    """Ejecuta una llamada asíncrona a Gemini con backoff exponencial."""
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return await operation()
        except errors.APIError as exc:
            last_error = exc
            if exc.code not in RETRYABLE_STATUS_CODES:
                raise VectorizationError(
                    f"Error no recuperable de la API (código {exc.code}): {exc}"
                ) from exc
            logger.warning("%s: error %s (intento %d)", label, exc.code, attempt + 1)
        except (OSError, httpx.HTTPError) as exc:
            last_error = exc
            logger.warning("%s: fallo de red (intento %d): %s", label, attempt + 1, exc)

        if attempt < max_retries:
            await asyncio.sleep(retry_base_delay * (2**attempt))

    raise VectorizationError(
        f"Se agotaron los reintentos para {label}: {last_error}"
    ) from last_error

