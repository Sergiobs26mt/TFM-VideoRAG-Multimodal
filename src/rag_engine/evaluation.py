"""Evaluación empírica y benchmark automatizado (Recall@K, MRR y latencia)."""

from __future__ import annotations

import csv
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from src.models import RerankedResult, SearchResult
from src.rag_engine.service import VideoRAGService
from src.utils.audit import RunAuditor


@dataclass(frozen=True)
class GroundTruthQuery:
    """Consulta de prueba con su vídeo e intervalo temporal esperado."""

    query: str
    video_name: str
    expected_start: float
    expected_end: float


@dataclass(frozen=True)
class QueryEvaluation:
    """Resultado de evaluar una consulta individual frente a su ground truth."""

    query: str
    video_name: str
    hit_rank: int | None
    reciprocal_rank: float
    latency_seconds: float


@dataclass(frozen=True)
class BenchmarkSummary:
    """Métricas agregadas de una batería de pruebas empíricas."""

    total_queries: int
    top_k: int
    rerank: bool
    recall_at_1: float
    recall_at_k: float
    mrr: float
    mean_latency_seconds: float
    evaluations: list[QueryEvaluation]


@dataclass(frozen=True)
class MultiTargetQuery:
    """Consulta abstracta que corresponde a varios intervalos temporales de un vídeo."""

    query: str
    video_name: str
    expected_intervals: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class MultiTargetEvaluation:
    """Resultado de evaluar una consulta multi-objetivo frente a sus intervalos esperados."""

    query: str
    video_name: str
    expected_count: int
    matched_intervals: int
    interval_recall: float
    relevant_returned: int
    true_positives: int
    precision: float
    first_hit_rank: int | None
    latency_seconds: float


@dataclass(frozen=True)
class NegativeQueryEvaluation:
    """Resultado de evaluar una consulta sobre contenido inexistente en el vídeo."""

    query: str
    total_candidates: int
    false_positives: int
    correctly_rejected: bool
    latency_seconds: float


def intervals_overlap(
    start_a: float,
    end_a: float,
    start_b: float,
    end_b: float,
    min_overlap_seconds: float = 0.0,
) -> bool:
    """Comprueba si dos intervalos temporales solapan más de ``min_overlap_seconds``."""
    overlap = min(end_a, end_b) - max(start_a, start_b)
    return overlap > min_overlap_seconds


def evaluate_search_results(
    target: GroundTruthQuery,
    results: Sequence[SearchResult | RerankedResult],
    latency_seconds: float = 0.0,
) -> QueryEvaluation:
    """Localiza el rango del primer candidato relevante que solapa con el intervalo esperado."""
    hit_rank: int | None = None
    for rank, res in enumerate(results, start=1):
        if isinstance(res, RerankedResult) and not res.is_relevant:
            continue
        if res.video_name == target.video_name and intervals_overlap(
            res.start_time,
            res.end_time,
            target.expected_start,
            target.expected_end,
        ):
            hit_rank = rank
            break

    rr = (1.0 / hit_rank) if hit_rank is not None else 0.0
    return QueryEvaluation(
        query=target.query,
        video_name=target.video_name,
        hit_rank=hit_rank,
        reciprocal_rank=rr,
        latency_seconds=latency_seconds,
    )


def evaluate_multi_target_results(
    target: MultiTargetQuery,
    results: Sequence[SearchResult | RerankedResult],
    latency_seconds: float = 0.0,
) -> MultiTargetEvaluation:
    """Calcula intervalos cubiertos, precisión y primer acierto para una consulta multi-objetivo."""
    first_hit_rank: int | None = None
    relevant_returned = 0
    true_positives = 0
    covered_indices: set[int] = set()

    for rank, res in enumerate(results, start=1):
        if isinstance(res, RerankedResult) and not res.is_relevant:
            continue
        relevant_returned += 1
        if res.video_name != target.video_name:
            continue

        matched_any = False
        for idx, (exp_start, exp_end) in enumerate(target.expected_intervals):
            if intervals_overlap(res.start_time, res.end_time, exp_start, exp_end):
                covered_indices.add(idx)
                matched_any = True

        if matched_any:
            true_positives += 1
            if first_hit_rank is None:
                first_hit_rank = rank

    expected_count = len(target.expected_intervals)
    matched_intervals = len(covered_indices)
    interval_recall = (
        matched_intervals / expected_count if expected_count > 0 else 0.0
    )
    precision = (
        true_positives / relevant_returned if relevant_returned > 0 else 0.0
    )
    return MultiTargetEvaluation(
        query=target.query,
        video_name=target.video_name,
        expected_count=expected_count,
        matched_intervals=matched_intervals,
        interval_recall=interval_recall,
        relevant_returned=relevant_returned,
        true_positives=true_positives,
        precision=precision,
        first_hit_rank=first_hit_rank,
        latency_seconds=latency_seconds,
    )


def evaluate_negative_results(
    query: str,
    results: Sequence[SearchResult | RerankedResult],
    latency_seconds: float = 0.0,
) -> NegativeQueryEvaluation:
    """Cuenta cuántos candidatos se dan por relevantes ante una consulta sin respuesta en el vídeo."""
    false_positives = sum(
        1 for res in results if not isinstance(res, RerankedResult) or res.is_relevant
    )
    return NegativeQueryEvaluation(
        query=query,
        total_candidates=len(results),
        false_positives=false_positives,
        correctly_rejected=false_positives == 0,
        latency_seconds=latency_seconds,
    )


def compute_benchmark_summary(
    evaluations: Sequence[QueryEvaluation],
    *,
    top_k: int,
    rerank: bool,
) -> BenchmarkSummary:
    """Calcula Recall@1, Recall@K, MRR y latencia media sobre una colección de evaluaciones."""
    total = len(evaluations)
    if total == 0:
        return BenchmarkSummary(
            total_queries=0,
            top_k=top_k,
            rerank=rerank,
            recall_at_1=0.0,
            recall_at_k=0.0,
            mrr=0.0,
            mean_latency_seconds=0.0,
            evaluations=[],
        )

    hits_at_1 = sum(1 for e in evaluations if e.hit_rank == 1)
    hits_at_k = sum(
        1 for e in evaluations if e.hit_rank is not None and e.hit_rank <= top_k
    )
    mrr = sum(e.reciprocal_rank for e in evaluations) / total
    mean_latency = sum(e.latency_seconds for e in evaluations) / total

    return BenchmarkSummary(
        total_queries=total,
        top_k=top_k,
        rerank=rerank,
        recall_at_1=hits_at_1 / total,
        recall_at_k=hits_at_k / total,
        mrr=mrr,
        mean_latency_seconds=mean_latency,
        evaluations=list(evaluations),
    )


def load_ground_truth(path: str | Path) -> list[GroundTruthQuery]:
    """Carga un conjunto de consultas con ground truth desde un fichero JSON."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"No existe el archivo de ground truth: {file_path}")

    raw = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise TypeError("El archivo de ground truth debe contener una lista JSON.")

    return [
        GroundTruthQuery(
            query=str(item["query"]),
            video_name=str(item["video_name"]),
            expected_start=float(item["expected_start"]),
            expected_end=float(item["expected_end"]),
        )
        for item in raw
    ]


def load_ground_truth_csv(
    path: str | Path,
    *,
    video_name: str | None = None,
) -> list[GroundTruthQuery]:
    """Carga consultas con ground truth desde un CSV, filtrando opcionalmente por ``video_name``."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"No existe el archivo de ground truth: {file_path}")

    with file_path.open(encoding="utf-8", newline="") as handle:
        return [
            GroundTruthQuery(
                query=row["test_query"],
                video_name=row["video_name"],
                expected_start=float(row["start_time"]),
                expected_end=float(row["end_time"]),
            )
            for row in csv.DictReader(handle)
            if video_name is None or row["video_name"] == video_name
        ]


async def run_benchmark(
    service: VideoRAGService,
    dataset: Sequence[GroundTruthQuery],
    *,
    top_k: int = 5,
    rerank: bool = True,
    auditor: RunAuditor | None = None,
) -> BenchmarkSummary:
    """Ejecuta todas las consultas del dataset y registra ``benchmark_summary.json``."""
    evaluations: list[QueryEvaluation] = []
    for item in dataset:
        t0 = time.perf_counter()
        results = await service.search(
            item.query,
            limit=top_k,
            rerank=rerank,
            auditor=auditor,
        )
        elapsed = time.perf_counter() - t0
        evaluations.append(evaluate_search_results(item, results, elapsed))

    summary = compute_benchmark_summary(evaluations, top_k=top_k, rerank=rerank)
    if auditor is not None:
        out_path = auditor.run_dir / "benchmark_summary.json"
        out_path.write_text(
            json.dumps(asdict(summary), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return summary


async def run_multi_target_benchmark(
    service: VideoRAGService,
    dataset: Sequence[MultiTargetQuery],
    *,
    top_k: int = 10,
    rerank: bool = True,
    auditor: RunAuditor | None = None,
) -> list[MultiTargetEvaluation]:
    """Ejecuta una batería de consultas multi-objetivo y devuelve sus evaluaciones."""
    evaluations: list[MultiTargetEvaluation] = []
    for item in dataset:
        t0 = time.perf_counter()
        results = await service.search(
            item.query,
            limit=top_k,
            rerank=rerank,
            auditor=auditor,
        )
        elapsed = time.perf_counter() - t0
        evaluations.append(evaluate_multi_target_results(item, results, elapsed))
    return evaluations


async def run_negative_benchmark(
    service: VideoRAGService,
    queries: Sequence[str],
    *,
    top_k: int = 5,
    rerank: bool = True,
    auditor: RunAuditor | None = None,
) -> list[NegativeQueryEvaluation]:
    """Ejecuta una batería de consultas negativas y evalúa el rechazo de falsos positivos."""
    evaluations: list[NegativeQueryEvaluation] = []
    for query in queries:
        t0 = time.perf_counter()
        results = await service.search(
            query,
            limit=top_k,
            rerank=rerank,
            auditor=auditor,
        )
        elapsed = time.perf_counter() - t0
        evaluations.append(evaluate_negative_results(query, results, elapsed))
    return evaluations

