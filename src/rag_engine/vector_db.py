"""Motor de almacenamiento y búsqueda vectorial de chunks de vídeo con Qdrant."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from src.config import COLLECTION_NAME, DB_PATH, PROCESSED_DIR, VECTOR_SIZE
from src.models import SearchResult, VectorDBError, VectorizedChunk
from src.processing.keyframe_cache import clear_all_keyframes, delete_video_keyframes

logger = logging.getLogger(__name__)

# Espacio de nombres fijo para generar IDs deterministas (reinsertar = sobrescribir)
_ID_NAMESPACE: uuid.UUID = uuid.UUID("6f1c2b7e-3a4d-4c8e-9b55-0d7a1e2f3a4b")


class QdrantSearchEngine:
    """Motor de recuperación vectorial basado en Qdrant (modo local/embebido)."""

    def __init__(
        self,
        path: str | None = DB_PATH,
        collection_name: str = COLLECTION_NAME,
        processed_dir: Path | None = None,
    ) -> None:
        """Inicializa el cliente de Qdrant.

        Args:
            path: Directorio de persistencia local. Con ``None`` se usa
                memoria (los datos se pierden al terminar el proceso).
            collection_name: Nombre de la colección de vectores.
            processed_dir: Carpeta de caché de fotogramas clave asociada.

        Raises:
            VectorDBError: Si no se puede abrir el almacenamiento local.
        """
        self._collection = collection_name
        self._processed_dir = (
            processed_dir
            if processed_dir is not None
            else (PROCESSED_DIR if path is not None else None)
        )
        try:
            self._client = QdrantClient(path=path) if path else QdrantClient(":memory:")
        except Exception as exc:  # Qdrant local puede fallar por bloqueo de fichero, etc.
            raise VectorDBError(f"No se pudo inicializar Qdrant: {exc}") from exc

    def create_collection(self, vector_size: int = VECTOR_SIZE, recreate: bool = False) -> None:
        """Crea la colección con métrica de similitud del coseno.

        Args:
            vector_size: Dimensión de los embeddings (p. ej. 768 o 3072).
            recreate: Si es ``True`` borra y recrea la colección existente.

        Raises:
            ValueError: Si la dimensión no es positiva.
            VectorDBError: Si Qdrant falla al crear la colección.
        """
        if vector_size < 1:
            raise ValueError("vector_size debe ser >= 1.")
        try:
            exists = self._client.collection_exists(self._collection)
            if exists and not recreate:
                logger.info("La colección '%s' ya existe", self._collection)
                return
            if exists:
                self._client.delete_collection(self._collection)
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=models.VectorParams(
                    size=vector_size, distance=models.Distance.COSINE
                ),
            )
            logger.info("Colección '%s' creada (dim=%d)", self._collection, vector_size)
        except (UnexpectedResponse, ValueError, OSError) as exc:
            raise VectorDBError(f"Error al crear la colección: {exc}") from exc

    def insert_chunks(
        self,
        chunks: list[VectorizedChunk],
        video_name: str = "unknown",
        batch_size: int = 64,
        overwrite_video: bool = True,
    ) -> int:
        """Inserta chunks vectorizados con sus metadatos como payload.

        El ID de cada punto es un UUID determinista derivado del vídeo y del
        índice. Si ``overwrite_video`` es True (por defecto), se eliminan previamente
        todos los puntos asociados a dicho vídeo para evitar chunks huérfanos cuando
        la nueva fragmentación produce menos planos que una anterior.

        Args:
            chunks: Chunks vectorizados a almacenar.
            video_name: Nombre del vídeo de origen.
            batch_size: Número de puntos por petición ``upsert``.
            overwrite_video: Si es True, elimina previamente los chunks existentes del vídeo.

        Returns:
            Número de puntos insertados.

        Raises:
            VectorDBError: Si Qdrant falla al insertar o eliminar.
        """
        if overwrite_video:
            self._delete_video_points(video_name)

        points = [
            models.PointStruct(
                id=str(uuid.uuid5(_ID_NAMESPACE, f"{video_name}:{c.index}")),
                vector=c.embedding,
                payload={
                    "video_name": video_name,
                    "chunk_index": c.index,
                    "start_time": c.start_time,
                    "end_time": c.end_time,
                    "transcript": c.transcript,
                },
            )
            for c in chunks
        ]
        try:
            for i in range(0, len(points), batch_size):
                self._client.upsert(
                    collection_name=self._collection, points=points[i : i + batch_size]
                )
        except (UnexpectedResponse, ValueError, OSError) as exc:
            raise VectorDBError(f"Error al insertar chunks: {exc}") from exc

        logger.info("Insertados %d chunks de '%s'", len(points), video_name)
        return len(points)

    def get_indexed_videos(self) -> list[str]:
        """Devuelve los nombres únicos de los vídeos almacenados en la colección.

        Returns:
            Lista de nombres de vídeos ordenados alfabéticamente.

        Raises:
            VectorDBError: Si ocurre un error al consultar Qdrant.
        """
        try:
            if not self._client.collection_exists(self._collection):
                return []
            points, _ = self._client.scroll(
                collection_name=self._collection,
                limit=10000,
                with_payload=["video_name"],
                with_vectors=False,
            )
            videos: set[str] = set()
            for point in points:
                if point.payload and "video_name" in point.payload:
                    videos.add(str(point.payload["video_name"]))
            return sorted(videos)
        except (UnexpectedResponse, ValueError, OSError) as exc:
            raise VectorDBError(f"Error al obtener vídeos indexados: {exc}") from exc

    def _delete_video_points(self, video_name: str) -> None:
        """Elimina exclusivamente los puntos de un vídeo en Qdrant."""
        try:
            if not self._client.collection_exists(self._collection):
                return
            self._client.delete(
                collection_name=self._collection,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="video_name",
                                match=models.MatchValue(value=video_name),
                            )
                        ]
                    )
                ),
            )
            logger.info(
                "Eliminados chunks de '%s' en la colección '%s'",
                video_name,
                self._collection,
            )
        except (UnexpectedResponse, ValueError, OSError) as exc:
            raise VectorDBError(f"Error al eliminar vídeo '{video_name}': {exc}") from exc

    def delete_video(self, video_name: str) -> None:
        """Elimina los vectores de un vídeo en Qdrant y su caché de fotogramas en disco."""
        self._delete_video_points(video_name)
        if self._processed_dir is not None:
            delete_video_keyframes(video_name, processed_dir=self._processed_dir)

    def clear_database(self, vector_size: int = VECTOR_SIZE) -> None:
        """Elimina por completo la colección, la recrea vacía y limpia la caché de fotogramas."""
        self.create_collection(vector_size=vector_size, recreate=True)
        if self._processed_dir is not None:
            clear_all_keyframes(processed_dir=self._processed_dir)
        logger.info(
            "Base de datos vectorial limpiada por completo (colección '%s')",
            self._collection,
        )

    def search_event(
        self,
        query_vector: list[float],
        limit: int = 5,
        video_name: str | None = None,
    ) -> list[SearchResult]:
        """Busca los chunks más similares a un vector de consulta.

        Usa ``query_points``, que sustituye al antiguo ``client.search``
        (eliminado en las versiones recientes de qdrant-client).

        Args:
            query_vector: Embedding de la consulta del usuario.
            limit: Número máximo de resultados (top-K).
            video_name: Filtro opcional para buscar solo en un vídeo específico.

        Returns:
            Resultados ordenados de mayor a menor similitud.

        Raises:
            ValueError: Si ``limit`` es menor que 1.
            VectorDBError: Si Qdrant falla durante la búsqueda.
        """
        if limit < 1:
            raise ValueError("limit debe ser >= 1.")

        query_filter = None
        if video_name:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="video_name",
                        match=models.MatchValue(value=video_name),
                    )
                ]
            )

        try:
            response = self._client.query_points(
                collection_name=self._collection,
                query=query_vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )
        except (UnexpectedResponse, ValueError, OSError) as exc:
            raise VectorDBError(f"Error en la búsqueda: {exc}") from exc

        results: list[SearchResult] = []
        for point in response.points:
            payload = point.payload or {}
            raw_transcript = payload.get("transcript")
            results.append(
                SearchResult(
                    score=float(point.score),
                    video_name=str(payload.get("video_name", "unknown")),
                    chunk_index=int(payload.get("chunk_index", -1)),
                    start_time=float(payload.get("start_time", 0.0)),
                    end_time=float(payload.get("end_time", 0.0)),
                    transcript=str(raw_transcript) if raw_transcript is not None else None,
                )
            )
        return results

