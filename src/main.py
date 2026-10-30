"""Orquestador CLI de la PoC: indexación de vídeo y búsqueda semántica."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections.abc import Sequence
from typing import Any

from src.config import (
    AUDIT_DIR,
    DB_PATH,
    RAW_DIR,
    VECTOR_SIZE,
)
from src.config import (
    build_run_config as _build_run_config,
)
from src.models import (
    QuotaSafetyError,
    RerankedResult,
    SearchResult,
    VectorDBError,
    VectorizationError,
)
from src.rag_engine.evaluation import (
    BenchmarkSummary,
    load_ground_truth,
    run_benchmark,
)
from src.rag_engine.service import VideoRAGService, format_timestamp
from src.utils.audit import RunAuditor

logger = logging.getLogger(__name__)

EXIT_COMMANDS: tuple[str, ...] = ("salir", "exit", "quit", "q")


def print_results(results: Sequence[SearchResult | RerankedResult]) -> None:
    """Imprime por consola los resultados de búsqueda de forma legible."""
    if not results:
        print("No se encontraron resultados.")
        return
    print(f"\nTop {len(results)} resultados:")
    for rank, r in enumerate(results, start=1):
        if isinstance(r, RerankedResult):
            status = "VALIDADO" if r.is_relevant else "DESCARTADO"
            print(
                f"  {rank}. [{r.video_name}] {format_timestamp(r.start_time)} - "
                f"{format_timestamp(r.end_time)}  "
                f"({r.start_time:.1f}s - {r.end_time:.1f}s)\n"
                f"      Relevancia VLM: {r.relevance_score:.2f} ({status}) | "
                f"Coseno Qdrant: {r.vector_score:.4f}\n"
                f"      Análisis visual: {r.reasoning}"
            )
        else:
            print(
                f"  {rank}. [{r.video_name}] {format_timestamp(r.start_time)} - "
                f"{format_timestamp(r.end_time)}  "
                f"({r.start_time:.1f}s - {r.end_time:.1f}s)  score={r.score:.4f}"
            )


def print_benchmark_summary(summary: BenchmarkSummary) -> None:
    """Imprime las métricas agregadas de una ejecución de benchmark."""
    print(
        f"\nResultados del Benchmark ({summary.total_queries} consultas, "
        f"top-K={summary.top_k}, rerank={summary.rerank}):\n"
        f"  Recall@1:       {summary.recall_at_1:.2%}\n"
        f"  Recall@{summary.top_k}:       {summary.recall_at_k:.2%}\n"
        f"  MRR:            {summary.mrr:.4f}\n"
        f"  Latencia media: {summary.mean_latency_seconds:.2f}s"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define y procesa los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="PoC Video RAG multimodal")
    parser.add_argument(
        "--index", action="store_true", help="Indexar el vídeo antes de buscar"
    )
    parser.add_argument(
        "--video", default="test_video.mp4", help="Vídeo dentro de data/raw/"
    )
    parser.add_argument("--top-k", type=int, default=5, help="Número de resultados")
    parser.add_argument(
        "--max-concurrent", type=int, default=3, help="Peticiones simultáneas"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Umbral SSIM para corte de planos"
    )
    parser.add_argument(
        "--min-shot-frames", type=int, default=12, help="Fotogramas mínimos por plano"
    )
    parser.add_argument(
        "--entropy-stride",
        type=int,
        default=1,
        help="Paso de muestreo para entropía",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=80,
        help="Calidad JPEG de los fotogramas (1-100)",
    )
    parser.add_argument(
        "--target-width",
        type=int,
        default=512,
        help="Ancho máximo de los fotogramas en píxeles (default: 512)",
    )
    parser.add_argument(
        "--rerank",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Activar reordenación multimodal (Etapa 2 con Gemini 3.5 Flash Lite)",
    )
    parser.add_argument(
        "--benchmark",
        default=None,
        help="Ruta a archivo JSON con ground truth para evaluación automática",
    )
    return parser.parse_args(argv)


def build_run_config(args: argparse.Namespace) -> dict[str, Any]:
    """Construye el volcado de parámetros de la ejecución para auditoría."""
    return _build_run_config(
        cli_args=vars(args),
        raw_dir=RAW_DIR,
        db_path=DB_PATH,
        vector_size=VECTOR_SIZE,
    )


def run_search_loop(
    service: VideoRAGService,
    top_k: int,
    auditor: RunAuditor,
    rerank: bool = True,
) -> int:
    """Bucle interactivo de búsqueda hasta que el usuario decida salir."""
    completed = 0
    print(f"Escribe tu búsqueda ({', '.join(EXIT_COMMANDS)} para terminar).")
    while True:
        try:
            query = input("\nIntroduce tu búsqueda: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            logger.info("Entrada interrumpida por el usuario")
            break

        if query.lower() in EXIT_COMMANDS:
            logger.info("El usuario terminó la sesión")
            break
        if not query:
            continue

        try:
            results = asyncio.run(
                service.search(query, limit=top_k, rerank=rerank, auditor=auditor)
            )
        except (ValueError, VectorizationError, VectorDBError) as exc:
            logger.error("Búsqueda fallida para %r: %s", query, exc)
            print(f"No se pudo completar la búsqueda: {exc}")
            continue

        print_results(results)
        completed += 1

    logger.info("Sesión finalizada: %d búsquedas completadas", completed)
    return completed


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada: indexación opcional y búsqueda interactiva."""
    args = parse_args(argv)

    with RunAuditor(AUDIT_DIR) as auditor:
        auditor.save_config(build_run_config(args))
        try:
            service = VideoRAGService(
                db_path=DB_PATH,
                raw_dir=RAW_DIR,
                vector_size=VECTOR_SIZE,
            )
            if args.index:
                asyncio.run(
                    service.index_video(
                        args.video,
                        max_concurrent=args.max_concurrent,
                        threshold=args.threshold,
                        min_shot_frames=args.min_shot_frames,
                        entropy_stride=args.entropy_stride,
                        jpeg_quality=args.jpeg_quality,
                        target_width=args.target_width,
                    )
                )
            if args.benchmark:
                dataset = load_ground_truth(args.benchmark)
                summary = asyncio.run(
                    run_benchmark(
                        service,
                        dataset,
                        top_k=args.top_k,
                        rerank=args.rerank,
                        auditor=auditor,
                    )
                )
                print_benchmark_summary(summary)
            else:
                run_search_loop(service, args.top_k, auditor, rerank=args.rerank)
        except (
            FileNotFoundError,
            TypeError,
            ValueError,
            QuotaSafetyError,
            VectorizationError,
            VectorDBError,
        ) as exc:
            logger.error("%s", exc)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
