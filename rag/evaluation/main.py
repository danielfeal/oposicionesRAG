"""Offline evaluation harness: one phase per component of the RAG pipeline.

Each phase writes its own JSONL artifact and is resumable; `report` reads all of them. They are
kept as separate functions rather than one linear run because their costs differ by orders of
magnitude - regenerating a report must not mean re-running generation. Edit the constants below
and comment out whichever `run_*` call in `main()` you don't want to execute.
"""

import asyncio
import logging
from dataclasses import replace
from typing import Sequence

from dotenv import load_dotenv
from qdrant_client import QdrantClient

from rag.config import AppConfig
from rag.evaluation import generation, retrieval
from rag.evaluation.relevance import evaluate_relevance
from rag.evaluation.report import build_report
from rag.evaluation.utils import EvalConfig, EvalRow, load_dataset
from rag.rag_pipeline.main import build_components, build_ollama, build_retrieval_only

logger = logging.getLogger(__name__)

# Edit these to control what the phases called in main() run on.
EXAMS: list[str] | None = ["A2"]  # None = all four
RESUME = False


def run_retrieve_retriever(
    config: EvalConfig, app_config: AppConfig, rows: Sequence[EvalRow], *, resume: bool
) -> None:
    """Sweep the retriever alone over `config.k_grid`. Run this first.

    Inspect the resulting report, then set `EvalConfig.retriever_k` to the chosen k before
    running `run_retrieve_reranker` - the reranker sweep needs that depth fixed, not swept.
    """
    graded_rows = [row for row in rows if not row.is_off_topic]

    client = QdrantClient(host=app_config.qdrant.host, port=app_config.qdrant.port)
    if not retrieval.valid_ground_truth(graded_rows, client, app_config):
        raise RuntimeError("Ground truth validation failed; see the logged errors above.")

    retriever, _reranker = build_retrieval_only(app_config)
    retrieval.evaluate_retriever(
        graded_rows, retriever, config.k_grid, config.path(config.retriever_results), resume
    )


def run_retrieve_reranker(
    config: EvalConfig, app_config: AppConfig, rows: Sequence[EvalRow], *, resume: bool
) -> None:
    """Sweep the reranker over `config.reranker_k_grid`, at the fixed `config.retriever_k`.

    Run only after `run_retrieve_retriever` and after choosing `retriever_k` from its report.
    """
    graded_rows = [row for row in rows if not row.is_off_topic]

    client = QdrantClient(host=app_config.qdrant.host, port=app_config.qdrant.port)
    if not retrieval.valid_ground_truth(graded_rows, client, app_config):
        raise RuntimeError("Ground truth validation failed; see the logged errors above.")

    retriever, reranker = build_retrieval_only(app_config)
    retrieval.evaluate_reranker(
        graded_rows,
        retriever,
        reranker,
        config.retriever_k,
        config.path(config.reranker_results),
        resume,
    )


def run_generate(
    config: EvalConfig, app_config: AppConfig, rows: Sequence[EvalRow], *, resume: bool
) -> None:
    """Run the real pipeline on every graded row and judge the answers it produces."""
    graded_rows = [row for row in rows if not row.is_off_topic]

    app_config = replace(app_config, trace_db=config.trace_db)  # never the production interactions.db
    components = build_components(app_config)

    asyncio.run(
        generation.evaluate_generation(
            graded_rows,
            components,
            app_config,
            config,
            config.path(config.generation_results),
            resume,
        )
    )


def run_relevance(
    config: EvalConfig, app_config: AppConfig, rows: Sequence[EvalRow], *, resume: bool
) -> None:
    """Classify every question with the relevance checker, on- and off-topic alike."""
    checker, _generator = build_ollama(app_config)
    evaluate_relevance(rows, checker, config.path(config.relevance_results), resume)


def run_report(config: EvalConfig, rows: Sequence[EvalRow]) -> None:
    """Assemble every artifact into one markdown report."""
    build_report(config, rows, config.path("report.md"))


def main() -> None:
    """Run the evaluation phases below in order; comment out any phase you don't want."""
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    config = EvalConfig()
    app_config = AppConfig()
    rows = load_dataset(config, EXAMS)

    # Retrieval is two sequential sweeps, not one run: (1) sweep the retriever alone, (2) read
    # the report, set EvalConfig.retriever_k to the chosen k, THEN sweep the reranker.
    # run_retrieve_retriever(config, app_config, rows, resume=RESUME)
    # run_retrieve_reranker(config, app_config, rows, resume=RESUME)
    # run_generate(config, app_config, rows, resume=RESUME)
    # run_relevance(config, app_config, rows, resume=RESUME)
    run_report(config, rows)


if __name__ == "__main__":
    main()
