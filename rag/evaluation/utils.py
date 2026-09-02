"""Shared support for the evaluation stages: configuration, the dataset, and artifact storage.

Only code used by more than one stage belongs here. Anything specific to a single stage - the
ground-truth matching rule, the retrieval metrics, the confusion matrix - lives in that stage's
module instead.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

QUESTION_TYPES = ("with_ref", "standard", "with_options", "off_topic")


@dataclass(frozen=True)
class EvalConfig:
    """Paths, the retrieval sweep grid, and judge settings for the evaluation harness."""

    dataset_dir: str = "./documentation"
    dataset_pattern: str = "evaluation_dataset_{exam}.csv"
    exams: tuple[str, ...] = ("A1", "A2", "C1", "C2")
    out_dir: str = "./documentation/evaluation"

    retriever_results: str = "retriever.jsonl"
    reranker_results: str = "reranker.jsonl"
    generation_results: str = "generation.jsonl"
    relevance_results: str = "relevance.jsonl"
    trace_db: str = "./documentation/evaluation/evaluation_traces.db"  # never interactions.db

    k_grid: tuple[int, ...] = (3, 5, 10, 20, 50)  # retriever sweep: how many candidates to fetch
    retriever_k: int = 20  # set by hand after reading the retriever sweep; feeds the reranker sweep
    reranker_k_grid: tuple[int, ...] = (1, 2, 3, 5, 10)  # how many reranked chunks to keep
    judge_model: str = "gemini-3.7-flash"
    judge_temperature: float = 0.0
    judge_max_score: int = 3

    def path(self, artifact: str) -> str:
        """Path of an artifact inside `out_dir`."""
        return os.path.join(self.out_dir, artifact)

    def dataset_path(self, exam: str) -> str:
        """Path of one exam's evaluation CSV."""
        return os.path.join(self.dataset_dir, self.dataset_pattern.format(exam=exam))


@dataclass(frozen=True)
class EvalRow:
    """One row of an evaluation CSV."""

    row_id: str  # the CSV `id`, e.g. "A2_001" - primary key across every artifact
    exam: str
    question_type: str
    question: str
    ground_truth_answer: str
    ground_truth_ref: str
    ground_truth_doc_id: str

    @property
    def is_off_topic(self) -> bool:
        """True for rows that only exercise the relevance check."""
        return self.question_type == "off_topic"


def load_dataset(config: EvalConfig, exams: Sequence[str] | None = None) -> list[EvalRow]:
    """Load and concatenate the per-exam CSVs, skipping any that do not exist yet.

    A missing exam file is a warning, not an error.
    """
    rows: list[EvalRow] = []
    for exam in exams or config.exams:
        path = config.dataset_path(exam)
        if not os.path.exists(path):
            logger.warning("No evaluation dataset for exam '%s' at '%s'; skipping.", exam, path)
            continue

        frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        rows.extend(
            EvalRow(
                row_id=record["id"].strip(),
                exam=record["exam"].strip(),
                question_type=record["question_type"].strip(),
                question=record["question"].strip(),
                ground_truth_answer=record["ground_truth_answer"].strip(),
                ground_truth_ref=record["ground_truth_ref"].strip(),
                ground_truth_doc_id=record["ground_truth_doc_id"].strip(),
            )
            for record in frame.to_dict("records")
        )
        logger.info("Loaded %d rows for exam '%s'.", len(frame), exam)

    if not rows:
        raise RuntimeError(f"No evaluation rows found for exams={list(exams or config.exams)}.")
    return rows


def append_record(path: str, record: Any) -> None:
    """Append one dataclass (or mapping) as a JSON line, flushed and fsynced.

    Each phase's JSONL is its output artifact, so writing it per row rather than buffering costs
    nothing and makes the phase resumable: the fsync costs ~1ms against rows that take seconds to
    produce, so an interrupted run loses at most the row in flight.
    """
    payload = record if isinstance(record, Mapping) else asdict(record)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_completed(path: str) -> dict[str, dict]:
    """Read a JSONL artifact into `{row_id: record}`; a missing file yields `{}`.

    A truncated final line (a run killed mid-write) is discarded rather than raising, so an
    interrupted phase always resumes cleanly.
    """
    if not os.path.exists(path):
        return {}

    completed: dict[str, dict] = {}
    with open(path, encoding="utf-8") as handle:
        lines = handle.readlines()

    for number, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if number == len(lines):
                logger.warning("Discarding truncated final line of '%s'.", path)
                continue
            raise
        completed[record["row_id"]] = record

    return completed
