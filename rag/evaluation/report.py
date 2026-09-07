"""Assembly of every evaluation artifact into one markdown report for the memoria."""

import logging
import os
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Sequence

import pandas as pd

from rag.evaluation import generation, query_processing, retrieval
from rag.evaluation.utils import EvalConfig, EvalRow, load_completed

logger = logging.getLogger(__name__)


def build_report(config: EvalConfig, rows: Sequence[EvalRow], out_path: str) -> None:
    """Read every artifact and write the report plus its backing tables."""
    retriever_records = list(load_completed(config.path(config.retriever_results)).values())
    generation_records = list(load_completed(config.path(config.generation_results)).values())
    query_processing_records = list(load_completed(config.path(config.query_processing_results)).values())

    sections = [
        f"# Evaluación del RAG\n\nGenerado: {datetime.now(UTC).isoformat()}\n",
        _dataset_section(rows),
        _retrieval_section(retriever_records, config),
        _generation_section(generation_records, config),
        _query_processing_section(query_processing_records),
        _latency_section(query_processing_records, generation_records),
    ]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("\n\n".join(section for section in sections if section))
    logger.info("Report written to '%s'.", out_path)


def _dataset_section(rows: Sequence[EvalRow]) -> str:
    """Row counts per exam and question type."""
    if not rows:
        return "## Dataset\n\nNo se han encontrado filas de dataset."

    frame = pd.DataFrame(
        [{"exam": row.exam, "question_type": row.question_type} for row in rows]
    )
    counts = frame.pivot_table(index="exam", columns="question_type", aggfunc="size", fill_value=0)
    return f"## Dataset\n\n{counts.to_markdown()}\n"


def _retrieval_section(records: Sequence[dict], config: EvalConfig) -> str:
    """Retriever sweep over candidate depth k. No reranker in the pipeline anymore."""
    if not records:
        return ""

    overall = pd.DataFrame([asdict(m) for m in retrieval.summarize_retriever(records, config.k_grid)])
    section = (
        "## Retrieval\n\nHit Rate@k del retriever, sin ninguna reordenación posterior.\n\n"
        f"{overall.to_markdown(index=False)}\n"
    )
    by_type = [
        {"question_type": question_type, **asdict(metric)}
        for question_type in sorted({r["question_type"] for r in records})
        for metric in retrieval.summarize_retriever(
            [r for r in records if r["question_type"] == question_type], config.k_grid
        )
    ]
    if by_type:
        section += f"\n### Por tipo de pregunta\n\n{pd.DataFrame(by_type).to_markdown(index=False)}\n"
    return section


def _generation_section(records: Sequence[dict], config: EvalConfig) -> str:
    """Judge scores overall, each with its own per-question-type score-histogram table."""
    if not records:
        return ""

    overall = generation.summarize(records, config.judge_max_score)
    correctness_table = pd.DataFrame(
        generation.score_breakdown_rows(records, "correctness", config.judge_max_score)
    )
    faithfulness_table = pd.DataFrame(
        generation.score_breakdown_rows(records, "faithfulness", config.judge_max_score)
    )

    section = (
        f"## Generación (LLM como juez: {config.judge_model})\n\n"
        f"Puntuaciones de 0 a {config.judge_max_score}. Las filas que el pipeline no llegó a "
        "responder puntúan 0 y siguen contando en el denominador.\n\n"
        f"- Filas evaluadas: **{overall.n_rows}**\n"
        f"- Puntuadas a cero por fallo del pipeline: **{overall.n_zeroed}**\n"
        f"- Correctness: **{overall.correctness:.2f}** / {config.judge_max_score} "
        f"({overall.correctness_normalized:.3f})\n\n"
        f"{correctness_table.to_markdown(index=False)}\n\n"
        f"- Faithfulness: **{overall.faithfulness:.2f}** / {config.judge_max_score} "
        f"({overall.faithfulness_normalized:.3f})\n\n"
        f"{faithfulness_table.to_markdown(index=False)}\n"
    )

    reasons = pd.Series(
        [record["zeroed_reason"] for record in records if record.get("zeroed_reason")]
    ).value_counts()
    if not reasons.empty:
        section += f"\n### Motivos de puntuación cero\n\n{reasons.to_markdown()}\n"

    statuses = pd.Series([record["status"] for record in records]).value_counts()
    section += f"\n### Estados del pipeline\n\n{statuses.to_markdown()}\n"
    return section


def _query_processing_section(records: Sequence[dict]) -> str:
    """Confusion matrix of the off-topic filter."""
    if not records:
        return ""

    confusion = query_processing.summarize(records)
    return (
        "## Query processing (filtrado off-topic)\n\n"
        "Clase positiva: off-topic. `UNKNOWN` cuenta como *predicho relevante*, igual que en "
        "producción (fail-open).\n\n"
        "|                    | pred. off-topic | pred. relevante |\n"
        "|--------------------|-----------------|-----------------|\n"
        f"| **real off-topic** | {confusion.tp} | {confusion.fn} |\n"
        f"| **real relevante** | {confusion.fp} | {confusion.tn} |\n\n"
        f"- Precision: **{confusion.precision:.3f}**\n"
        f"- Accuracy: **{confusion.accuracy:.3f}**\n"
        f"- Veredictos `UNKNOWN`: **{confusion.n_unknown}**\n"
    )


def _latency_section(query_processing_records: Sequence[dict], generation_records: Sequence[dict]) -> str:
    """Per-stage latency of the full pipeline, over the rows that actually reached each stage."""
    rows: list[dict] = []
    processing = query_processing.processing_latency(query_processing_records)
    if processing:
        rows.append(processing)
    rows.extend(generation.stage_latencies(generation_records))

    if not rows:
        return ""
    return (
        "## Latencia\n\nEtapas del pipeline completo, solo sobre las filas que alcanzan cada una:"
        f"\n\n{pd.DataFrame(rows).to_markdown(index=False)}\n"
    )
