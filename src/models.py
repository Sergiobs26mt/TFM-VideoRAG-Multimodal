"""Modelos de dominio y excepciones compartidas del sistema Video RAG."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


class QuotaSafetyError(RuntimeError):
    """El vídeo supera el límite de chunks permitido para proteger la cuota de la API."""


class VectorizationError(RuntimeError):
    """Fallo al generar el embedding multimodal o textual."""


class VectorDBError(RuntimeError):
    """Error al operar con la base de datos vectorial."""


class AudioTranscriptionError(RuntimeError):
    """Error en la extracción o transcripción de la pista de audio."""


@dataclass(frozen=True)
class TranscriptSegment:
    """Segmento de locución transcrita con marcas temporales en segundos."""

    start_time: float
    end_time: float
    text: str


@dataclass(frozen=True)
class VideoChunk:
    """Plano temporal coherente dentro de un vídeo."""

    index: int
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    transcript: str | None = None


@dataclass(frozen=True)
class VectorizedChunk:
    """Fragmento de vídeo asociado a su representación vectorial densa."""

    index: int
    start_time: float
    end_time: float
    embedding: list[float]
    transcript: str | None = None


@dataclass(frozen=True)
class SearchResult:
    """Candidato recuperado por similitud vectorial en la Etapa 1."""

    score: float
    video_name: str
    chunk_index: int
    start_time: float
    end_time: float
    transcript: str | None = None


@dataclass(frozen=True)
class RerankedResult:
    """Candidato verificado y reordenado visualmente en la Etapa 2."""

    result: SearchResult
    relevance_score: float
    is_relevant: bool
    reasoning: str

    @property
    def video_name(self) -> str:
        """Nombre del archivo de vídeo."""
        return self.result.video_name

    @property
    def chunk_index(self) -> int:
        """Índice del fragmento dentro del vídeo."""
        return self.result.chunk_index

    @property
    def start_time(self) -> float:
        """Instante de inicio en segundos."""
        return self.result.start_time

    @property
    def end_time(self) -> float:
        """Instante de fin en segundos."""
        return self.result.end_time

    @property
    def vector_score(self) -> float:
        """Similitud del coseno obtenida en la Etapa 1."""
        return self.result.score

    @property
    def score(self) -> float:
        """Alias de ``vector_score`` para compatibilidad uniforme en resultados."""
        return self.result.score

    @property
    def transcript(self) -> str | None:
        """Transcripción de audio alineada con el fragmento."""
        return self.result.transcript


@dataclass(frozen=True)
class MultimodalQuery:
    """Consulta de búsqueda que combina texto, documentos .txt, imágenes y audio."""

    text: str | None = None
    txt_contents: tuple[str, ...] = ()
    images: tuple[bytes, ...] = ()
    audio_paths: tuple[Path, ...] = ()
    audio_bytes: tuple[bytes, ...] = ()
    audio_transcripts: tuple[str, ...] = ()

    @property
    def has_content(self) -> bool:
        """Indica si existe al menos una modalidad no vacía en la consulta."""
        return bool(
            (self.text and self.text.strip())
            or any(t.strip() for t in self.txt_contents)
            or self.images
            or self.audio_paths
            or self.audio_bytes
            or any(t.strip() for t in self.audio_transcripts)
        )

    @property
    def composed_text(self) -> str:
        """Une texto directo, archivos .txt y transcripciones de audio en una sola cadena."""
        parts: list[str] = []
        if self.text and self.text.strip():
            parts.append(self.text.strip())
        for item in self.txt_contents:
            if item.strip():
                parts.append(item.strip())
        for tr in self.audio_transcripts:
            if tr.strip():
                parts.append(tr.strip())
        return " ".join(parts)

    @property
    def modality_summary(self) -> dict[str, Any]:
        """Resumen serializable de las modalidades empleadas en la consulta."""
        return {
            "has_text": bool(self.text and self.text.strip()),
            "txt_count": len([t for t in self.txt_contents if t.strip()]),
            "image_count": len(self.images),
            "audio_count": len(self.audio_paths) + len(self.audio_bytes),
            "audio_transcript_count": len([t for t in self.audio_transcripts if t.strip()]),
            "composed_text": self.composed_text,
        }

