"""Assembly of every evaluation artifact into one markdown report for the memoria."""

import logging
import os
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Sequence

import pandas as pd

from rag.evaluation import generation, relevance, retrieval
from rag.evaluation.utils import EvalConfig, EvalRow, load_completed

logger = logging.getLogger(__name__)


def build_report(config: EvalConfig, rows: Sequence[EvalRow], out_path: str) -> None:
    """Read every artifact and write the report plus its backing tables."""
    retriever_records = list(load_completed(config.path(config.retriever_results)).values())
    reranker_records = list(load_completed(config.path(config.reranker_results)).values())
    generation_records = list(load_completed(config.path(config.generation_results)).values())
    relevance_records = list(load_completed(config.path(config.relevance_results)).values())

    sections = [
        f"# Evaluación del RAG\n\nGenerado: {datetime.now(UTC).isoformat()}\n",
        _dataset_section(rows),
        _retrieval_section(retriever_records, reranker_records, config),
        _generation_section(generation_records, config),
        _relevance_section(relevance_records),
        _latency_section(retriever_records, reranker_records, generation_records),
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


def _retrieval_section(
    retriever_records: Sequence[dict], reranker_records: Sequence[dict], config: EvalConfig
) -> str:
    """Retriever sweep and reranker sweep, reported separately - they answer different questions.

    The retriever table sweeps candidate depth k with no reranking, answering "how many
    candidates should the retriever hand to the reranker?". The reranker table is computed at
    the single fixed `config.retriever_k`, answering "of those, how many should the reranker
    keep?" - its k grid is bounded by `retriever_k` and its numbers are only meaningful once
    that depth has actually been chosen from the retriever table below.
    """
    sections: list[str] = []

    if retriever_records:
        overall = pd.DataFrame(
            [asdict(m) for m in retrieval.summarize_retriever(retriever_records, config.k_grid)]
        )
        section = (
            "### Retriever (sin reranking)\n\nHit Rate@k marca el techo: el reranker solo "
            "reordena lo que el retriever ya ha devuelto, nunca puede superarlo.\n\n"
            f"{overall.to_markdown(index=False)}\n"
        )
        by_type = [
            {"question_type": question_type, **asdict(metric)}
            for question_type in sorted({r["question_type"] for r in retriever_records})
            for metric in retrieval.summarize_retriever(
                [r for r in retriever_records if r["question_type"] == question_type], config.k_grid
            )
        ]
        if by_type:
            section += (
                "\n#### Por tipo de pregunta\n\nLa diferencia entre `with_ref` y `standard` "
                "mide cuánto aporta nombrar la fuente en la pregunta.\n\n"
                f"{pd.DataFrame(by_type).to_markdown(index=False)}\n"
            )
        sections.append(section)

    if reranker_records:
        overall = pd.DataFrame(
            [asdict(m) for m in retrieval.summarize_reranker(reranker_records, config.reranker_k_grid)]
        )
        sections.append(
            f"### Reranker (retriever_k = {config.retriever_k})\n\nSweep de cuántos chunks "
            "reordenados conservar, con la profundidad del retriever ya fijada.\n\n"
            f"{overall.to_markdown(index=False)}\n"
        )

    return "## Retrieval\n\n" + "\n".join(sections) if sections else ""


def _generation_section(records: Sequence[dict], config: EvalConfig) -> str:
    """Judge scores overall and grouped, with the hard-zero breakdown."""
    if not records:
        return ""

    overall = generation.summarize(records, config.judge_max_score)
    section = (
        f"## Generación (LLM como juez: {config.judge_model})\n\n"
        f"Puntuaciones de 0 a {config.judge_max_score}. Las filas que el pipeline no llegó a "
        "responder puntúan 0 y siguen contando en el denominador.\n\n"
        f"- Filas evaluadas: **{overall.n_rows}**\n"
        f"- Puntuadas a cero por fallo del pipeline: **{overall.n_zeroed}**\n"
        f"- Correctness: **{overall.correctness:.2f}** / {config.judge_max_score} "
        f"({overall.correctness_normalized:.3f})\n"
        f"- Faithfulness: **{overall.faithfulness:.2f}** / {config.judge_max_score} "
        f"({overall.faithfulness_normalized:.3f})\n"
    )

    grouped: list[dict] = []
    for key in ("exam", "question_type"):
        for value in sorted({record[key] for record in records}):
            subset = [record for record in records if record[key] == value]
            grouped.append(
                {"grupo": f"{key}={value}", **asdict(generation.summarize(subset, config.judge_max_score))}
            )
    if grouped:
        section += f"\n{pd.DataFrame(grouped).to_markdown(index=False)}\n"

    reasons = pd.Series(
        [record["zeroed_reason"] for record in records if record.get("zeroed_reason")]
    ).value_counts()
    if not reasons.empty:
        section += f"\n### Motivos de puntuación cero\n\n{reasons.to_markdown()}\n"

    statuses = pd.Series([record["status"] for record in records]).value_counts()
    section += f"\n### Estados del pipeline\n\n{statuses.to_markdown()}\n"
    return section


def _relevance_section(records: Sequence[dict]) -> str:
    """Confusion matrix of the off-topic filter."""
    if not records:
        return ""

    confusion = relevance.summarize(records)
    return (
        "## Relevance check (filtrado off-topic)\n\n"
        "Clase positiva: off-topic. `UNKNOWN` cuenta como *predicho relevante*, igual que en "
        "producción (fail-open).\n\n"
        "|                    | pred. off-topic | pred. relevante |\n"
        "|--------------------|-----------------|-----------------|\n"
        f"| **real off-topic** | {confusion.tp} | {confusion.fn} |\n"
        f"| **real relevante** | {confusion.fp} | {confusion.tn} |\n\n"
        f"- Precision: **{confusion.precision:.3f}**\n"
        f"- Recall: **{confusion.recall:.3f}**\n"
        f"- F1: **{confusion.f1:.3f}**\n"
        f"- Accuracy: **{confusion.accuracy:.3f}**\n"
        f"- Veredictos `UNKNOWN`: **{confusion.n_unknown}**\n"
    )


def _latency_section(
    retriever_records: Sequence[dict], reranker_records: Sequence[dict], generation_records: Sequence[dict]
) -> str:
    """Per-stage latency, over the rows that actually reached each stage."""
    sections: list[str] = []

    stages = generation.stage_latencies(generation_records)
    if stages:
        sections.append(
            "Etapas del pipeline completo, solo sobre las filas que alcanzan cada una:\n\n"
            f"{pd.DataFrame(stages).to_markdown(index=False)}"
        )

    retriever_latencies = retrieval.retriever_latencies(retriever_records)
    if retriever_latencies:
        frame = pd.DataFrame([{k: round(v, 1) for k, v in retriever_latencies.items()}])
        sections.append(
            f"Sweep del retriever (una llamada por k del grid):\n\n{frame.to_markdown(index=False)}"
        )

    reranker_latencies = retrieval.reranker_latencies(reranker_records)
    if reranker_latencies:
        frame = pd.DataFrame([{k: round(v, 1) for k, v in reranker_latencies.items()}])
        sections.append(
            f"Sweep del reranker (una recuperación + un reranking por fila):\n\n"
            f"{frame.to_markdown(index=False)}"
        )

    return "## Latencia\n\n" + "\n\n".join(sections) + "\n" if sections else ""
