"""Offline evaluation harness: one phase per component of the RAG pipeline.

Each phase writes its own JSONL artifact; `report` reads all of them. They are kept as separate
functions rather than one linear run because their costs differ by orders of magnitude -
regenerating a report must not mean re-running generation. Comment out whichever `run_*` call in
main() you don't want to execute; every phase always runs against the full dataset.
"""

import asyncio
import logging
from dataclasses import replace
from typing import Sequence

from dotenv import load_dotenv
from qdrant_client import QdrantClient

from rag.config import AppConfig
from rag.evaluation import generation, retrieval
from rag.evaluation.query_processing import evaluate_query_processing
from rag.evaluation.report import build_report
from rag.evaluation.utils import EvalConfig, EvalRow, load_dataset, load_query_results
from rag.rag_pipeline.main import build_components, build_llm, build_retrieval_only

logger = logging.getLogger(__name__)


def run_retrieve(config: EvalConfig, app_config: AppConfig, rows: Sequence[EvalRow]) -> None:
    """Sweep the retriever alone over `config.k_grid`, retrieving on each row's condensed query."""
    graded_rows = [row for row in rows if not row.is_off_topic]
    condensed_queries = {
        row_id: result.condensed_query for row_id, result in load_query_results(config, graded_rows).items()
    }

    client = QdrantClient(host=app_config.qdrant.host, port=app_config.qdrant.port)
    if not retrieval.valid_ground_truth(graded_rows, client, app_config):
        raise RuntimeError("Ground truth validation failed; see the logged errors above.")

    retriever = build_retrieval_only(app_config)
    retrieval.evaluate_retriever(
        graded_rows, condensed_queries, retriever, config.k_grid, config.path(config.retriever_results)
    )


def run_generate(config: EvalConfig, app_config: AppConfig, rows: Sequence[EvalRow]) -> None:
    """Run the real pipeline on every graded row and judge the answers it produces.

    Reuses `run_query_processing`'s condensed queries and verdicts instead of running the query
    processor a second time - see `load_query_results`.
    """
    graded_rows = [row for row in rows if not row.is_off_topic]
    query_results = load_query_results(config, graded_rows)

    app_config = replace(app_config, trace_db=config.trace_db)  # never the production interactions.db
    components = build_components(app_config)

    asyncio.run(
        generation.evaluate_generation(
            graded_rows, query_results, components, app_config, config, config.path(config.generation_results)
        )
    )


def run_query_processing(config: EvalConfig, app_config: AppConfig, rows: Sequence[EvalRow]) -> None:
    """Classify every question with the query processor, on- and off-topic alike."""
    processor, _generator = build_llm(app_config)
    evaluate_query_processing(rows, processor, config.path(config.query_processing_results))


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
    rows = load_dataset(config)

    run_query_processing(config, app_config, rows)
    run_retrieve(config, app_config, rows)  # needs run_query_processing's condensed queries
    run_generate(config, app_config, rows)
    run_report(config, rows)


if __name__ == "__main__":
    main()
