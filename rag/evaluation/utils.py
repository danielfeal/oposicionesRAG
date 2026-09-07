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

from rag.pipeline.query_processor import QueryProcessingResult
from rag.pipeline.types import RelevanceVerdict

logger = logging.getLogger(__name__)

QUESTION_TYPES = ("with_ref", "standard", "with_options", "off_topic")


@dataclass(frozen=True)
class EvalConfig:
    """Paths, the retrieval sweep grid, and judge settings for the evaluation harness."""

    dataset_dir: str = "./documentation"
    dataset_filename: str = "evaluation_dataset_v2.csv"
    out_dir: str = "./documentation/evaluation"

    retriever_results: str = "retriever.jsonl"
    generation_results: str = "generation.jsonl"
    query_processing_results: str = "query_processing.jsonl"

    k_grid: tuple[int, ...] = (1, 2, 3, 4, 5, 10, 20)  # how many candidates to fetch
    judge_model: str = "gemini-3.7-flash"
    judge_temperature: float = 0.0
    judge_max_score: int = 3

    def path(self, artifact: str) -> str:
        """Path of an artifact inside `out_dir`."""
        return os.path.join(self.out_dir, artifact)

    def dataset_path(self) -> str:
        """Path of the evaluation dataset CSV."""
        return os.path.join(self.dataset_dir, self.dataset_filename)


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


def load_dataset(config: EvalConfig) -> list[EvalRow]:
    """Load the evaluation dataset CSV. Always the full set - no per-exam subsetting."""
    path = config.dataset_path()
    if not os.path.exists(path):
        raise RuntimeError(f"No evaluation dataset at '{path}'.")

    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    rows = [
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
    ]
    logger.info("Loaded %d rows from '%s'.", len(rows), path)
    return rows


def append_record(path: str, record: Any) -> None:
    """Append one dataclass (or mapping) as a JSON line, flushed and fsynced.

    Each phase's JSONL is its output artifact, so writing it per row rather than buffering costs
    nothing and keeps partial results on disk if the run is interrupted: the fsync costs ~1ms
    against rows that take seconds to produce, so a crash loses at most the row in flight.
    """
    payload = record if isinstance(record, Mapping) else asdict(record)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def reset(path: str) -> None:
    """Delete `path` if it exists, so a fresh full run doesn't append onto stale rows."""
    if os.path.exists(path):
        os.remove(path)


def load_completed(path: str) -> dict[str, dict]:
    """Read a JSONL artifact into `{row_id: record}`; a missing file yields `{}`.

    A truncated final line (a run killed mid-write) is discarded rather than raising, so a
    crashed phase's partial results are still readable.
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


def load_query_results(config: EvalConfig, rows: Sequence[EvalRow]) -> dict[str, QueryProcessingResult]:
    """Load each row's already-computed relevance verdict + condensed query.

    Lets `run_retrieve` and `run_generate` reuse `run_query_processing`'s output instead of
    paying for a second query-processing pass over queries that haven't changed - production only
    ever runs query processing once per real query, so evaluating it twice per row would test
    something production never does. Raises if any row is missing - `run_query_processing` must
    run first.
    """
    records = load_completed(config.path(config.query_processing_results))
    missing = [row.row_id for row in rows if row.row_id not in records]
    if missing:
        raise RuntimeError(
            f"No query-processing results for {len(missing)} rows (e.g. {missing[:5]}); "
            "run_query_processing must run first."
        )
    return {
        row.row_id: QueryProcessingResult(
            condensed_query=records[row.row_id]["condensed_query"],
            relevance=RelevanceVerdict(records[row.row_id]["verdict"]),
        )
        for row in rows
    }
