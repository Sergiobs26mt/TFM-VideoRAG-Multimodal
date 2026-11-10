"""Arnés E2E: indexa animals.mp4 en Qdrant temporal y ejecuta los benchmarks con la API real."""

from __future__ import annotations

import asyncio
import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from src.config import PROJECT_ROOT, VECTOR_SIZE
from src.embeddings.gemini_vectorizer import GeminiMultimodalVectorizer
from src.models import RerankedResult, SearchResult
from src.rag_engine.evaluation import (
    BenchmarkSummary,
    GroundTruthQuery,
    MatryoshkaDimensionMetrics,
    MatryoshkaMultiTargetMetrics,
    MatryoshkaNegativeMetrics,
    MultiTargetEvaluation,
    MultiTargetQuery,
    NegativeQueryEvaluation,
    evaluate_matryoshka_dimensions,
    evaluate_matryoshka_multi_target,
    evaluate_matryoshka_negative,
    load_ground_truth_csv,
    run_benchmark,
    run_multi_target_benchmark,
    run_negative_benchmark,
    write_matryoshka_ablation_csv,
    write_matryoshka_multi_target_csv,
    write_matryoshka_negative_csv,
)
from src.rag_engine.reranker import MultimodalReranker
from src.rag_engine.service import VideoRAGService
from src.rag_engine.vector_db import QdrantSearchEngine
from src.utils.audit import RunAuditor

VIDEO_NAME = "animals.mp4"
PYTHON_VIDEO_NAME = "python_tutorial.mp4"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
GROUND_TRUTH_CSV = PROJECT_ROOT / "data" / "dataframe_gt" / "video_chunks_ground_truth.csv"
TOP_K = 5
MULTI_TARGET_TOP_K = 10
REPORT_FILENAME = "e2e_report.csv"
MULTI_TARGET_REPORT_FILENAME = "e2e_multi_target_report.csv"
NEGATIVE_REPORT_FILENAME = "e2e_negative_report.csv"
PYTHON_REPORT_FILENAME = "python_e2e_report.csv"
PYTHON_MULTI_TARGET_REPORT_FILENAME = "python_e2e_multi_target_report.csv"
PYTHON_NEGATIVE_REPORT_FILENAME = "python_e2e_negative_report.csv"
MATRYOSHKA_REPORT_FILENAME = "matryoshka_ablation_20261110.csv"
MATRYOSHKA_MULTI_TARGET_FILENAME = "matryoshka_multi_target_20261110.csv"
MATRYOSHKA_NEGATIVE_FILENAME = "matryoshka_negative_20261110.csv"
MATRYOSHKA_DIMENSIONS: tuple[int, ...] = (3072, 1536, 768)
EXPECTED_E2E_CHUNKS = 82

MULTI_TARGET_QUERIES: tuple[MultiTargetQuery, ...] = (
    MultiTargetQuery(
        query="Felinos",
        video_name=VIDEO_NAME,
        expected_intervals=((5.0, 10.0), (18.0, 23.0), (35.0, 39.0), (193.0, 196.0)),
    ),
    MultiTargetQuery(
        query="Reptiles",
        video_name=VIDEO_NAME,
        expected_intervals=((23.0, 29.0), (232.0, 241.0)),
    ),
    MultiTargetQuery(
        query="Aves",
        video_name=VIDEO_NAME,
        expected_intervals=((39.0, 45.0), (52.0, 59.0), (106.0, 110.0), (120.0, 124.0)),
    ),
    MultiTargetQuery(
        query="Manada o grupo de tres o más animales juntos",
        video_name=VIDEO_NAME,
        expected_intervals=(
            (14.0, 18.0),
            (29.0, 32.0),
            (39.0, 45.0),
            (96.0, 102.0),
            (106.0, 110.0),
            (114.0, 120.0),
            (124.0, 130.0),
            (136.0, 144.0),
            (144.0, 158.0),
            (208.0, 212.0),
        ),
    ),
    MultiTargetQuery(
        query="Animales de color blanco y negro",
        video_name=VIDEO_NAME,
        expected_intervals=(
            (1.0, 5.0),
            (45.0, 49.0),
            (49.0, 52.0),
            (106.0, 110.0),
            (114.0, 120.0),
        ),
    ),
    MultiTargetQuery(
        query="Animales marinos o acuáticos nadando o flotando en el agua",
        video_name=VIDEO_NAME,
        expected_intervals=(
            (32.0, 35.0),
            (39.0, 45.0),
            (59.0, 64.0),
            (196.0, 198.0),
            (212.0, 217.0),
            (223.0, 227.0),
            (232.0, 241.0),
            (241.0, 243.0),
        ),
    ),
    MultiTargetQuery(
        query="Animales comiendo, pastando o amamantándose",
        video_name=VIDEO_NAME,
        expected_intervals=(
            (49.0, 52.0),
            (76.0, 81.0),
            (81.0, 86.0),
            (86.0, 90.0),
            (90.0, 96.0),
            (110.0, 114.0),
            (114.0, 120.0),
            (227.0, 232.0),
        ),
    ),
    MultiTargetQuery(
        query="Crías de animales o animales jóvenes",
        video_name=VIDEO_NAME,
        expected_intervals=(
            (14.0, 18.0),
            (45.0, 49.0),
            (49.0, 52.0),
            (81.0, 86.0),
            (102.0, 106.0),
            (110.0, 114.0),
            (136.0, 144.0),
        ),
    ),
    MultiTargetQuery(
        query="Primerísimo plano macro de ojos, orejas o fosas nasales de un animal",
        video_name=VIDEO_NAME,
        expected_intervals=(
            (23.0, 29.0),
            (130.0, 136.0),
            (180.0, 186.0),
            (193.0, 196.0),
            (198.0, 201.0),
            (201.0, 208.0),
        ),
    ),
    MultiTargetQuery(
        query="Animación 3D microscópica de células o glóbulos rojos",
        video_name=VIDEO_NAME,
        expected_intervals=(
            (64.0, 76.0),
            (158.0, 170.0),
            (170.0, 174.0),
            (174.0, 180.0),
        ),
    ),
)

NEGATIVE_QUERIES: tuple[str, ...] = (
    "Tiburón blanco atacando en el océano",
    "Canguro saltando por el desierto australiano",
    "Serpiente cobra deslizándose por la arena",
    "Águila volando alto en el cielo azul",
    "Cocodrilo con las fauces abiertas en la orilla de un río",
    "Bandada de flamencos rosas en una laguna salada",
    "Camión de bomberos rojo con la sirena encendida por la calle",
    "Persona programando código en un ordenador portátil en una oficina",
)

PYTHON_MULTI_TARGET_QUERIES: tuple[MultiTargetQuery, ...] = (
    MultiTargetQuery(
        query="Planos del presentador en blanco y negro",
        video_name=PYTHON_VIDEO_NAME,
        expected_intervals=((0.0, 0.7), (51.4, 53.5), (109.0, 110.2)),
    ),
    MultiTargetQuery(
        query="Tipos de datos primitivos en Python (int, float, str, bool, None, complex)",
        video_name=PYTHON_VIDEO_NAME,
        expected_intervals=(
            (25.0, 27.9),
            (27.9, 30.5),
            (30.5, 33.5),
            (33.5, 37.0),
            (37.0, 40.0),
            (40.0, 44.8),
            (44.8, 47.5),
            (47.5, 49.8),
            (49.8, 51.4),
        ),
    ),
    MultiTargetQuery(
        query="Estructuras de datos y colecciones en Python (list, tuple, set, dict)",
        video_name=PYTHON_VIDEO_NAME,
        expected_intervals=(
            (53.5, 57.5),
            (57.5, 66.1),
            (66.1, 71.1),
            (71.1, 76.1),
            (76.1, 84.4),
            (84.4, 96.1),
        ),
    ),
    MultiTargetQuery(
        query="Rótulos o símbolos tachados con una cruz roja X en pantalla",
        video_name=PYTHON_VIDEO_NAME,
        expected_intervals=((110.2, 114.1), (201.8, 205.4)),
    ),
    MultiTargetQuery(
        query="Terminal o consola mostrando la salida de ejecución de un script con números o texto",
        video_name=PYTHON_VIDEO_NAME,
        expected_intervals=(
            (161.6, 166.5),
            (224.4, 225.3),
            (228.1, 230.2),
            (236.0, 238.3),
        ),
    ),
    MultiTargetQuery(
        query="Código de condicionales if, elif y else en el editor Visual Studio Code",
        video_name=PYTHON_VIDEO_NAME,
        expected_intervals=((185.0, 191.0), (191.0, 197.6), (205.4, 213.6)),
    ),
    MultiTargetQuery(
        query="Código de bucles for y while en el editor Visual Studio Code",
        video_name=PYTHON_VIDEO_NAME,
        expected_intervals=(
            (221.1, 224.4),
            (225.3, 228.1),
            (230.2, 232.5),
            (232.5, 236.0),
        ),
    ),
    MultiTargetQuery(
        query="Declaración y reasignación de variables en el editor Visual Studio Code",
        video_name=PYTHON_VIDEO_NAME,
        expected_intervals=((100.2, 107.7), (114.1, 122.3), (132.8, 139.2)),
    ),
    MultiTargetQuery(
        query="Uso de funciones integradas input, len y print en el editor con ventana emergente de documentación",
        video_name=PYTHON_VIDEO_NAME,
        expected_intervals=(
            (149.6, 153.1),
            (153.1, 156.7),
            (156.7, 158.7),
            (221.1, 224.4),
            (225.3, 228.1),
            (240.3, 242.5),
        ),
    ),
    MultiTargetQuery(
        query="Títulos de sección sobreimpresos en letras blancas grandes con el presentador hablando a cámara",
        video_name=PYTHON_VIDEO_NAME,
        expected_intervals=(
            (19.0, 25.0),
            (53.5, 57.5),
            (96.1, 100.2),
            (145.7, 149.6),
            (158.7, 161.6),
            (173.7, 175.6),
            (213.6, 221.1),
            (238.3, 240.3),
        ),
    ),
)

PYTHON_NEGATIVE_QUERIES: tuple[str, ...] = (
    "Definición de una clase orientada a objetos con class y el método __init__ en Python",
    "Gráfico de barras o visualización de datos con matplotlib y pandas en un Jupyter Notebook",
    "Consulta SQL con SELECT y FROM en una base de datos PostgreSQL",
    "Instalación de paquetes con el comando pip install en la terminal",
    "Diagrama de arquitectura de redes neuronales o entrenamiento con PyTorch",
    "Página web con código HTML y estilos CSS abierta en el navegador Chrome",
    "León macho con melena descansando tumbado sobre la hierba verde",
    "Partido de fútbol en un estadio lleno de espectadores",
)


@dataclass(frozen=True)
class E2ERun:
    """Resultado de una ejecución E2E completa."""

    indexed_chunks: int
    indexed_videos: list[str]
    dataset: list[GroundTruthQuery]
    retrieval: BenchmarkSummary
    reranked: BenchmarkSummary
    report_path: Path
    multi_target_retrieval: list[MultiTargetEvaluation]
    multi_target_reranked: list[MultiTargetEvaluation]
    multi_target_report_path: Path
    negative_retrieval: list[NegativeQueryEvaluation]
    negative_reranked: list[NegativeQueryEvaluation]
    negative_report_path: Path


def write_report(
    path: Path,
    dataset: list[GroundTruthQuery],
    retrieval: BenchmarkSummary,
    reranked: BenchmarkSummary,
) -> None:
    """Escribe un CSV con el rango del acierto de cada consulta, con y sin reordenación."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "query",
                "video_name",
                "expected_start",
                "expected_end",
                "hit_rank_retrieval",
                "hit_rank_rerank",
            ]
        )
        for item, plain, judged in zip(
            dataset, retrieval.evaluations, reranked.evaluations, strict=True
        ):
            writer.writerow(
                [
                    item.query,
                    item.video_name,
                    item.expected_start,
                    item.expected_end,
                    plain.hit_rank,
                    judged.hit_rank,
                ]
            )


def write_multi_target_report(
    path: Path,
    retrieval: Sequence[MultiTargetEvaluation],
    reranked: Sequence[MultiTargetEvaluation],
) -> None:
    """Escribe un CSV con la cobertura y precisión de cada consulta multi-objetivo."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "query",
                "video_name",
                "expected_count",
                "matched_retrieval",
                "recall_retrieval",
                "precision_retrieval",
                "first_hit_retrieval",
                "matched_rerank",
                "recall_rerank",
                "precision_rerank",
                "first_hit_rerank",
            ]
        )
        for plain, judged in zip(retrieval, reranked, strict=True):
            writer.writerow(
                [
                    plain.query,
                    plain.video_name,
                    plain.expected_count,
                    plain.matched_intervals,
                    f"{plain.interval_recall:.4f}",
                    f"{plain.precision:.4f}",
                    plain.first_hit_rank,
                    judged.matched_intervals,
                    f"{judged.interval_recall:.4f}",
                    f"{judged.precision:.4f}",
                    judged.first_hit_rank,
                ]
            )


def write_negative_report(
    path: Path,
    retrieval: Sequence[NegativeQueryEvaluation],
    reranked: Sequence[NegativeQueryEvaluation],
) -> None:
    """Escribe un CSV con los falsos positivos de cada consulta negativa antes y después del juez."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "query",
                "total_candidates",
                "false_positives_retrieval",
                "false_positives_rerank",
                "correctly_rejected_rerank",
            ]
        )
        for plain, judged in zip(retrieval, reranked, strict=True):
            writer.writerow(
                [
                    plain.query,
                    plain.total_candidates,
                    plain.false_positives,
                    judged.false_positives,
                    judged.correctly_rejected,
                ]
            )


async def execute_video_e2e_run(
    auditor: RunAuditor,
    *,
    video_name: str,
    multi_target_queries: Sequence[MultiTargetQuery],
    negative_queries: Sequence[str],
    report_filename: str,
    multi_target_report_filename: str,
    negative_report_filename: str,
    threshold: float = 0.5,
    max_shot_duration: float = 8.0,
) -> E2ERun:
    """Indexa un vídeo y evalúa sus baterías literal, multi-objetivo y negativa."""
    dataset = load_ground_truth_csv(GROUND_TRUTH_CSV, video_name=video_name)
    engine = QdrantSearchEngine(path=None)
    service = VideoRAGService(
        engine=engine,
        raw_dir=RAW_DIR,
        processed_dir=PROCESSED_DIR,
        vector_size=VECTOR_SIZE,
    )

    indexed_chunks = await service.index_video(
        video_name, threshold=threshold, max_shot_duration=max_shot_duration
    )
    retrieval = await run_benchmark(service, dataset, top_k=TOP_K, rerank=False)
    reranked = await run_benchmark(
        service, dataset, top_k=TOP_K, rerank=True, auditor=auditor
    )

    multi_retrieval = await run_multi_target_benchmark(
        service, multi_target_queries, top_k=MULTI_TARGET_TOP_K, rerank=False
    )
    multi_reranked = await run_multi_target_benchmark(
        service,
        multi_target_queries,
        top_k=MULTI_TARGET_TOP_K,
        rerank=True,
        auditor=auditor,
    )

    negative_retrieval = await run_negative_benchmark(
        service, negative_queries, top_k=TOP_K, rerank=False
    )
    negative_reranked = await run_negative_benchmark(
        service, negative_queries, top_k=TOP_K, rerank=True, auditor=auditor
    )

    report_path = auditor.run_dir / report_filename
    write_report(report_path, dataset, retrieval, reranked)

    multi_report_path = auditor.run_dir / multi_target_report_filename
    write_multi_target_report(multi_report_path, multi_retrieval, multi_reranked)

    negative_report_path = auditor.run_dir / negative_report_filename
    write_negative_report(negative_report_path, negative_retrieval, negative_reranked)

    return E2ERun(
        indexed_chunks=indexed_chunks,
        indexed_videos=service.get_indexed_videos(),
        dataset=dataset,
        retrieval=retrieval,
        reranked=reranked,
        report_path=report_path,
        multi_target_retrieval=multi_retrieval,
        multi_target_reranked=multi_reranked,
        multi_target_report_path=multi_report_path,
        negative_retrieval=negative_retrieval,
        negative_reranked=negative_reranked,
        negative_report_path=negative_report_path,
    )


async def execute_e2e_run(auditor: RunAuditor) -> E2ERun:
    """Indexa animals.mp4 y evalúa las baterías literal, multi-objetivo y negativa."""
    return await execute_video_e2e_run(
        auditor,
        video_name=VIDEO_NAME,
        multi_target_queries=MULTI_TARGET_QUERIES,
        negative_queries=NEGATIVE_QUERIES,
        report_filename=REPORT_FILENAME,
        multi_target_report_filename=MULTI_TARGET_REPORT_FILENAME,
        negative_report_filename=NEGATIVE_REPORT_FILENAME,
    )


async def execute_python_e2e_run(auditor: RunAuditor) -> E2ERun:
    """Indexa python_tutorial.mp4 y evalúa las baterías literal, multi-objetivo y negativa."""
    return await execute_video_e2e_run(
        auditor,
        video_name=PYTHON_VIDEO_NAME,
        multi_target_queries=PYTHON_MULTI_TARGET_QUERIES,
        negative_queries=PYTHON_NEGATIVE_QUERIES,
        report_filename=PYTHON_REPORT_FILENAME,
        multi_target_report_filename=PYTHON_MULTI_TARGET_REPORT_FILENAME,
        negative_report_filename=PYTHON_NEGATIVE_REPORT_FILENAME,
        threshold=0.6,
        max_shot_duration=5.0,
    )


@dataclass(frozen=True)
class MatryoshkaAblationRun:
    """Resultado de la evaluación empírica Matryoshka en 3072, 1536 y 768 dimensiones."""

    python_chunks: int
    animals_chunks: int
    python_metrics: list[MatryoshkaDimensionMetrics]
    animals_metrics: list[MatryoshkaDimensionMetrics]
    python_multi_target: list[MatryoshkaMultiTargetMetrics]
    animals_multi_target: list[MatryoshkaMultiTargetMetrics]
    python_negative: list[MatryoshkaNegativeMetrics]
    animals_negative: list[MatryoshkaNegativeMetrics]
    csv_path: Path
    multi_target_csv_path: Path
    negative_csv_path: Path
    detail_csv_path: Path


def write_matryoshka_detail_csv(
    path: Path,
    dataset: Sequence[GroundTruthQuery],
    metrics_by_dim: Sequence[MatryoshkaDimensionMetrics],
) -> None:
    """Escribe o añade el desglose por consulta de hit_rank en 3072, 1536 y 768 dimensiones."""
    by_dim = {m.dimension: m.evaluations for m in metrics_by_dim}
    write_header = not path.is_file()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(
                [
                    "video_name",
                    "query",
                    "expected_start",
                    "expected_end",
                    "hit_rank_3072",
                    "hit_rank_1536",
                    "hit_rank_768",
                ]
            )
        for idx, item in enumerate(dataset):
            r3072 = by_dim[3072][idx].hit_rank
            r1536 = by_dim[1536][idx].hit_rank
            r768 = by_dim[768][idx].hit_rank
            writer.writerow(
                [
                    item.video_name,
                    item.query,
                    item.expected_start,
                    item.expected_end,
                    r3072,
                    r1536,
                    r768,
                ]
            )


async def execute_matryoshka_ablation_run(
    auditor: RunAuditor,
    *,
    force_reindex: bool = False,
) -> MatryoshkaAblationRun:
    """Garantiza la indexación de ambos vídeos (82 chunks c/u a 3072-d) y evalúa MRL (3072, 1536, 768)."""
    db_dir = str(PROJECT_ROOT / "data" / "vector_db")
    engine = QdrantSearchEngine(path=db_dir, processed_dir=PROCESSED_DIR)
    try:
        service = VideoRAGService(
            engine=engine,
            raw_dir=RAW_DIR,
            processed_dir=PROCESSED_DIR,
            vector_size=VECTOR_SIZE,
        )

        py_chunks = engine.get_video_chunks_with_vectors(PYTHON_VIDEO_NAME)
        if (
            force_reindex
            or len(py_chunks) != EXPECTED_E2E_CHUNKS
            or len(py_chunks[0].embedding) != VECTOR_SIZE
            or not any(c.transcript for c in py_chunks)
        ):
            await service.index_video(
                PYTHON_VIDEO_NAME,
                threshold=0.6,
                max_shot_duration=5.0,
                output_dimensionality=VECTOR_SIZE,
                enable_audio=True,
            )
            py_chunks = engine.get_video_chunks_with_vectors(PYTHON_VIDEO_NAME)

        an_chunks = engine.get_video_chunks_with_vectors(VIDEO_NAME)
        if (
            force_reindex
            or len(an_chunks) != EXPECTED_E2E_CHUNKS
            or len(an_chunks[0].embedding) != VECTOR_SIZE
            or not any(c.transcript for c in an_chunks)
        ):
            await service.index_video(
                VIDEO_NAME,
                threshold=0.5,
                max_shot_duration=8.0,
                output_dimensionality=VECTOR_SIZE,
                enable_audio=True,
            )
            an_chunks = engine.get_video_chunks_with_vectors(VIDEO_NAME)
    finally:
        engine._client.close()

    python_dataset = load_ground_truth_csv(
        GROUND_TRUTH_CSV, video_name=PYTHON_VIDEO_NAME
    )
    animals_dataset = load_ground_truth_csv(GROUND_TRUTH_CSV, video_name=VIDEO_NAME)

    all_queries = list(
        dict.fromkeys(
            [item.query for item in python_dataset]
            + [item.query for item in animals_dataset]
            + [item.query for item in PYTHON_MULTI_TARGET_QUERIES]
            + [item.query for item in MULTI_TARGET_QUERIES]
            + list(PYTHON_NEGATIVE_QUERIES)
            + list(NEGATIVE_QUERIES)
        )
    )
    vectorizer = GeminiMultimodalVectorizer(output_dimensionality=VECTOR_SIZE)
    semaphore = asyncio.Semaphore(3)
    query_vectors_3072: dict[str, list[float]] = {}

    async def _embed_q(q_text: str) -> tuple[str, list[float]]:
        async with semaphore:
            return q_text, await vectorizer.embed_query(q_text)

    pairs = await asyncio.gather(*(_embed_q(q) for q in all_queries))
    for q_text, vec in pairs:
        query_vectors_3072[q_text] = vec

    reranker = MultimodalReranker()
    verdict_cache: dict[
        tuple[str, str, int, float, float], tuple[float, bool, str]
    ] = {}

    async def _cached_rerank(
        query: str, candidates: list[SearchResult]
    ) -> list[RerankedResult]:
        missing: list[SearchResult] = []
        for cand in candidates:
            key = (
                query,
                cand.video_name,
                cand.chunk_index,
                round(cand.start_time, 2),
                round(cand.end_time, 2),
            )
            if key not in verdict_cache:
                missing.append(cand)
        if missing:
            fresh = await reranker.rerank(
                query,
                missing,
                raw_dir=RAW_DIR,
                processed_dir=PROCESSED_DIR,
            )
            for item in fresh:
                k = (
                    query,
                    item.video_name,
                    item.chunk_index,
                    round(item.start_time, 2),
                    round(item.end_time, 2),
                )
                verdict_cache[k] = (
                    item.relevance_score,
                    item.is_relevant,
                    item.reasoning,
                )
        evaluated: list[RerankedResult] = []
        for cand in candidates:
            key = (
                query,
                cand.video_name,
                cand.chunk_index,
                round(cand.start_time, 2),
                round(cand.end_time, 2),
            )
            rel_score, is_rel, reason = verdict_cache[key]
            evaluated.append(
                RerankedResult(
                    result=cand,
                    relevance_score=rel_score,
                    is_relevant=is_rel,
                    reasoning=reason,
                )
            )
        evaluated.sort(
            key=lambda r: (not r.is_relevant, -r.relevance_score, -r.vector_score)
        )
        return evaluated

    python_metrics = evaluate_matryoshka_dimensions(
        video_name=PYTHON_VIDEO_NAME,
        chunks_3072=py_chunks,
        dataset=python_dataset,
        query_vectors_3072=query_vectors_3072,
        dimensions=MATRYOSHKA_DIMENSIONS,
        top_k=TOP_K,
    )
    animals_metrics = evaluate_matryoshka_dimensions(
        video_name=VIDEO_NAME,
        chunks_3072=an_chunks,
        dataset=animals_dataset,
        query_vectors_3072=query_vectors_3072,
        dimensions=MATRYOSHKA_DIMENSIONS,
        top_k=TOP_K,
    )

    python_multi_target = await evaluate_matryoshka_multi_target(
        video_name=PYTHON_VIDEO_NAME,
        chunks_3072=py_chunks,
        queries=PYTHON_MULTI_TARGET_QUERIES,
        query_vectors_3072=query_vectors_3072,
        rerank_fn=_cached_rerank,
        dimensions=MATRYOSHKA_DIMENSIONS,
        top_k=MULTI_TARGET_TOP_K,
    )
    animals_multi_target = await evaluate_matryoshka_multi_target(
        video_name=VIDEO_NAME,
        chunks_3072=an_chunks,
        queries=MULTI_TARGET_QUERIES,
        query_vectors_3072=query_vectors_3072,
        rerank_fn=_cached_rerank,
        dimensions=MATRYOSHKA_DIMENSIONS,
        top_k=MULTI_TARGET_TOP_K,
    )

    python_negative = await evaluate_matryoshka_negative(
        video_name=PYTHON_VIDEO_NAME,
        chunks_3072=py_chunks,
        queries=PYTHON_NEGATIVE_QUERIES,
        query_vectors_3072=query_vectors_3072,
        rerank_fn=_cached_rerank,
        dimensions=MATRYOSHKA_DIMENSIONS,
        top_k=TOP_K,
    )
    animals_negative = await evaluate_matryoshka_negative(
        video_name=VIDEO_NAME,
        chunks_3072=an_chunks,
        queries=NEGATIVE_QUERIES,
        query_vectors_3072=query_vectors_3072,
        rerank_fn=_cached_rerank,
        dimensions=MATRYOSHKA_DIMENSIONS,
        top_k=TOP_K,
    )

    all_metrics = [*python_metrics, *animals_metrics]
    all_multi_target = [*python_multi_target, *animals_multi_target]
    all_negative = [*python_negative, *animals_negative]

    audit_csv = auditor.run_dir / MATRYOSHKA_REPORT_FILENAME
    write_matryoshka_ablation_csv(audit_csv, all_metrics)
    write_matryoshka_multi_target_csv(
        auditor.run_dir / MATRYOSHKA_MULTI_TARGET_FILENAME, all_multi_target
    )
    write_matryoshka_negative_csv(
        auditor.run_dir / MATRYOSHKA_NEGATIVE_FILENAME, all_negative
    )

    results_dir = PROJECT_ROOT / "data" / "e2e_results"
    results_csv = results_dir / MATRYOSHKA_REPORT_FILENAME
    multi_csv = results_dir / MATRYOSHKA_MULTI_TARGET_FILENAME
    neg_csv = results_dir / MATRYOSHKA_NEGATIVE_FILENAME
    write_matryoshka_ablation_csv(results_csv, all_metrics)
    write_matryoshka_multi_target_csv(multi_csv, all_multi_target)
    write_matryoshka_negative_csv(neg_csv, all_negative)

    detail_csv = auditor.run_dir / "matryoshka_queries_detail_20261110.csv"
    write_matryoshka_detail_csv(detail_csv, python_dataset, python_metrics)
    write_matryoshka_detail_csv(detail_csv, animals_dataset, animals_metrics)

    return MatryoshkaAblationRun(
        python_chunks=len(py_chunks),
        animals_chunks=len(an_chunks),
        python_metrics=python_metrics,
        animals_metrics=animals_metrics,
        python_multi_target=python_multi_target,
        animals_multi_target=animals_multi_target,
        python_negative=python_negative,
        animals_negative=animals_negative,
        csv_path=results_csv,
        multi_target_csv_path=multi_csv,
        negative_csv_path=neg_csv,
        detail_csv_path=detail_csv,
    )


