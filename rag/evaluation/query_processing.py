"""Evaluation of the query processor: does it tell domain questions from off-topic ones, and how
does it condense each query?

A standalone pass rather than a by-product of the generation run. The stage only needs
`QueryProcessor.process(query)` - no retrieval, no generation - so all questions cost a few
seconds each and this phase is independent of the multi-hour generation pass. It is also the
more honest experimental design: the component is evaluated in isolation.
"""

import logging
import statistics
import time
from dataclasses import dataclass
from typing import Sequence

from tqdm import tqdm

from rag.evaluation.utils import EvalRow, append_record, reset
from rag.pipeline.query_processor import QueryProcessor

logger = logging.getLogger(__name__)


def evaluate_query_processing(rows: Sequence[EvalRow], processor: QueryProcessor, path: str) -> None:
    """Classify every question and record the verdict + condensation against its dataset label."""
    reset(path)
    logger.info("Running query processing for %d rows.", len(rows))

    for row in tqdm(rows, desc="query_processing", unit="row"):
        started = time.perf_counter()
        result = processor.process(row.question)
        append_record(
            path,
            {
                "row_id": row.row_id,
                "exam": row.exam,
                "question_type": row.question_type,
                "question": row.question,
                "condensed_query": result.condensed_query,
                "actual_off_topic": row.is_off_topic,
                "verdict": result.relevance.value,
                "duration_ms": (time.perf_counter() - started) * 1000,
            },
        )


def processing_latency(records: Sequence[dict]) -> dict | None:
    """Mean/max query-processing latency, in the same row shape as `generation.stage_latencies`.

    Sourced from this phase's own `duration_ms`, not from `generation.jsonl`: `run_generate`
    reuses this phase's condensed queries instead of running the query processor again, so
    `generation.jsonl` rows never carry a "processing" duration of their own anymore.
    """
    if not records:
        return None
    values = [record["duration_ms"] for record in records]
    return {
        "stage": "processing",
        "n": len(values),
        "avg_s": round(statistics.fmean(values) / 1000, 2),
        "max_s": round(max(values) / 1000, 2),
    }


@dataclass(frozen=True)
class ConfusionMetrics:
    """Confusion matrix of the off-topic filter. The positive class is off-topic."""

    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    accuracy: float
    n_unknown: int


def summarize(records: Sequence[dict]) -> ConfusionMetrics:
    """Score the relevance check against the dataset labels.

    A row counts as predicted-positive only when the verdict is OFF_TOPIC. UNKNOWN means the
    classifier failed and the pipeline falls open and answers anyway, so it is counted as
    predicted-relevant, matching production behaviour.
    """
    tp = fp = fn = tn = n_unknown = 0
    for record in records:
        if record["verdict"] == "UNKNOWN":
            n_unknown += 1

        actual_off_topic = record["actual_off_topic"]
        predicted_off_topic = record["verdict"] == "OFF_TOPIC"

        if actual_off_topic and predicted_off_topic:
            tp += 1
        elif actual_off_topic:
            fn += 1
        elif predicted_off_topic:
            fp += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    total = tp + fp + fn + tn
    return ConfusionMetrics(
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        precision=precision,
        recall=recall,
        f1=2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        accuracy=(tp + tn) / total if total else 0.0,
        n_unknown=n_unknown,
    )
