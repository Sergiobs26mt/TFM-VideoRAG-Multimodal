"""Trazabilidad de ejecuciones: logs, configuración y resultados por run."""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from src.config import AUDIT_DIR

logger = logging.getLogger(__name__)

_LOG_FORMAT: str = "%(asctime)s %(levelname)s %(name)s: %(message)s"


class RunAuditor:
    """Gestor de contexto que audita una ejecución completa del programa.

    Al entrar crea ``audit/run_YYYYMMDD_HHMMSS/`` y redirige TODO el logging
    (todos los módulos y todos los niveles) a ``execution.log``, además de la
    consola. Genera también ``run_config.json`` y ``search_results.json``.

    Example:
        >>> with RunAuditor() as auditor:
        ...     auditor.save_config({"video": "test.mp4"})
    """

    def __init__(self, base_dir: Path | str = AUDIT_DIR) -> None:
        """Prepara el auditor sin tocar el disco hasta entrar en el contexto.

        Args:
            base_dir: Carpeta raíz de las auditorías (por defecto ``audit/``).
        """
        self._base_dir = Path(base_dir)
        self._run_dir: Path | None = None
        self._searches: list[dict[str, Any]] = []
        self._file_handler: logging.FileHandler | None = None
        self._console_handler: logging.StreamHandler[Any] | None = None
        self._previous_level: int = logging.WARNING
        self._previous_handlers: list[logging.Handler] = []

    @property
    def run_dir(self) -> Path:
        """Carpeta de la ejecución actual.

        Returns:
            Ruta de ``audit/run_<fecha>_<hora>/``.

        Raises:
            RuntimeError: Si se consulta fuera del contexto.
        """
        if self._run_dir is None:
            raise RuntimeError("RunAuditor no está activo (usa 'with RunAuditor()').")
        return self._run_dir

    def __enter__(self) -> Self:
        """Crea la carpeta del run y reconfigura el logging raíz.

        Returns:
            El propio auditor.
        """
        self._run_dir = self._create_run_dir()

        root = logging.getLogger()
        self._previous_level = root.level
        self._previous_handlers = list(root.handlers)
        for handler in self._previous_handlers:
            root.removeHandler(handler)
        root.setLevel(logging.DEBUG)  # captura absolutamente todos los niveles

        formatter = logging.Formatter(_LOG_FORMAT)
        self._file_handler = logging.FileHandler(
            self._run_dir / "execution.log", encoding="utf-8"
        )
        self._file_handler.setLevel(logging.DEBUG)
        self._file_handler.setFormatter(formatter)

        self._console_handler = logging.StreamHandler(sys.stderr)
        self._console_handler.setLevel(logging.INFO)  # consola legible, sin ruido DEBUG
        self._console_handler.setFormatter(formatter)

        root.addHandler(self._file_handler)
        root.addHandler(self._console_handler)
        logger.info("Auditoría iniciada en %s", self._run_dir)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Registra excepciones no controladas y restaura el logging original.

        Args:
            exc_type: Tipo de excepción (si la hubo).
            exc: Instancia de la excepción.
            tb: Traza de la excepción.
        """
        if exc is not None:
            logger.error("Ejecución terminada por excepción", exc_info=exc)
        logger.info("Auditoría finalizada")

        root = logging.getLogger()
        for own_handler in (self._file_handler, self._console_handler):
            if own_handler is not None:
                root.removeHandler(own_handler)
                own_handler.close()
        for handler in self._previous_handlers:
            root.addHandler(handler)
        root.setLevel(self._previous_level)
        # No se suprime ninguna excepción: se devuelve None

    def save_config(self, config: dict[str, Any]) -> Path:
        """Guarda los parámetros de la ejecución en ``run_config.json``.

        Args:
            config: Parámetros (argumentos, umbrales, constantes...).

        Returns:
            Ruta del fichero escrito.
        """
        return self._write_json("run_config.json", config)

    def save_search_results(self, query: str, results: list[Any]) -> Path:
        """Añade una búsqueda a ``search_results.json`` (acumula todas las del run).

        El fichero contiene ``{"searches": [{"query": ..., "results": [...]}, ...]}``
        en orden cronológico y se reescribe tras cada búsqueda, de modo que no se
        pierde nada si la ejecución se interrumpe.

        Args:
            query: Texto de la consulta del usuario.
            results: Resultados devueltos por Qdrant (dataclasses o dicts).

        Returns:
            Ruta del fichero escrito.
        """
        serialized = [
            asdict(r) if is_dataclass(r) and not isinstance(r, type) else r
            for r in results
        ]
        self._searches.append({"query": query, "results": serialized})
        return self._write_json("search_results.json", {"searches": self._searches})

    def _create_run_dir(self) -> Path:
        """Crea una subcarpeta única ``run_YYYYMMDD_HHMMSS`` (con sufijo si colisiona).

        Returns:
            Ruta de la carpeta creada.
        """
        self._base_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        candidate = self._base_dir / stamp
        suffix = 1
        while True:
            try:
                candidate.mkdir()
                return candidate
            except FileExistsError:
                suffix += 1
                candidate = self._base_dir / f"{stamp}_{suffix}"

    def _write_json(self, filename: str, data: dict[str, Any]) -> Path:
        """Escribe un diccionario como JSON legible dentro de la carpeta del run.

        Args:
            filename: Nombre del fichero.
            data: Contenido serializable.

        Returns:
            Ruta del fichero escrito.
        """
        path = self.run_dir / filename
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        logger.debug("Escrito %s", path)
        return path
