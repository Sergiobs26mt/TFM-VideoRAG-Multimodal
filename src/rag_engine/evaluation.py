"""Evaluación empírica y benchmark automatizado (Recall@K, MRR y latencia)."""

from __future__ import annotations

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
