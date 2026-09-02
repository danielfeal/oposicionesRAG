"""Evaluation of the relevance check: does it tell domain questions from off-topic ones?

A standalone pass rather than a by-product of the generation run. The stage only needs
`RelevanceChecker.check(query)` - no retrieval, no generation - so all questions cost a few
seconds each and this phase is independent of the multi-hour generation pass. It is also the
more honest experimental design: the component is evaluated in isolation.
"""

import logging
import time
from dataclasses import dataclass
from typing import Sequence

from tqdm import tqdm

from rag.evaluation.utils import EvalRow, append_record, load_completed
from rag.rag_pipeline.relevance import RelevanceChecker

logger = logging.getLogger(__name__)


def evaluate_relevance(
    rows: Sequence[EvalRow], checker: RelevanceChecker, path: str, resume: bool
) -> None:
    """Classify every question and record the verdict against its dataset label."""
    completed = load_completed(path) if resume else {}
    todo = [row for row in rows if row.row_id not in completed] if resume else list(rows)
    logger.info("Checking relevance for %d rows (%d already done).", len(todo), len(completed))

    for row in tqdm(todo, desc="relevance", unit="row"):
        started = time.perf_counter()
        verdict = checker.check(row.question)
        append_record(
            path,
            {
                "row_id": row.row_id,
                "exam": row.exam,
                "question_type": row.question_type,
                "question": row.question,
                "actual_off_topic": row.is_off_topic,
                "verdict": verdict.value,
                "duration_ms": (time.perf_counter() - started) * 1000,
            },
        )


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
