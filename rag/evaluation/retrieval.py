"""Evaluation of the retriever: sweeps candidate depth k with no reranking involved.

Owns the definition of a correct retrieval (`chunk_matches`) and the ground-truth validation
that proves every reference exists in the corpus.
"""

import logging
import statistics
import time
import unicodedata
from dataclasses import dataclass
from typing import Mapping, Sequence

from qdrant_client import QdrantClient, models

from rag.config import AppConfig
from rag.evaluation.utils import EvalRow, append_record, reset
from rag.rag_pipeline.retriever import HybridRetriever
from rag.rag_pipeline.types import RetrievedChunk

logger = logging.getLogger(__name__)


def normalize_ref(value: str | None) -> str:
    """NFC-normalize, strip and collapse internal whitespace. Case and accents preserved.

    Deliberately not case- or accent-folding: corpus headings and the CSV references derive from
    the same BOE strings, so they match literally, and folding would silently mask the
    ground-truth typos that `valid_ground_truth` exists to surface. NFC is required because a
    spreadsheet export can emit an accented vowel as a base letter plus a combining accent.
    """
    if not value:
        return ""
    return " ".join(unicodedata.normalize("NFC", value).split())


def chunk_matches(
    chunk_doc_id: str, chunk_seccion: str | None, chunk_tipo_seccion: str, row: EvalRow
) -> bool:
    """True if a corpus chunk is the ground-truth chunk for `row`.

    Matching uses the composite key `(doc_id, ref)`: `ground_truth_ref` alone is not unique across
    the corpus - the same article number exists in many different laws - so a reference-only
    comparison would match across documents. Preamble chunks carry `seccion is None` (the chunker
    never assigns one), so the section comparison falls back to `tipo_seccion`, which the CSV
    writes as the literal "preambulo".
    """
    expected = normalize_ref(row.ground_truth_ref)
    actual = normalize_ref(chunk_seccion) if chunk_seccion else normalize_ref(chunk_tipo_seccion)
    return chunk_doc_id == row.ground_truth_doc_id and bool(expected) and actual == expected


def rank_of_ground_truth(chunks: Sequence[RetrievedChunk], row: EvalRow) -> int | None:
    """1-based position of the ground-truth chunk in `chunks`, or None if absent."""
    for rank, chunk in enumerate(chunks, start=1):
        if chunk_matches(
            chunk.metadata.doc_id, chunk.metadata.seccion, chunk.metadata.tipo_seccion, row
        ):
            return rank
    return None


def hit_rate(ranks: Sequence[int | None], k: int) -> float:
    """Fraction of rows whose ground-truth chunk appears at rank <= k."""
    if not ranks:
        return 0.0
    return sum(1 for rank in ranks if rank is not None and rank <= k) / len(ranks)


def valid_ground_truth(rows: Sequence[EvalRow], client: QdrantClient, config: AppConfig) -> bool:
    """Check every graded row's `(doc_id, ref)` resolves to a real chunk in Qdrant.

    Runs before the retrieval pass: without it a retrieval miss is ambiguous between "the
    retriever failed" and "the ground-truth reference does not exist in the corpus" - opposite
    problems with opposite fixes. Logs every unresolved row before returning, so one run surfaces
    every problem instead of a fix-one-rerun-find-the-next loop.
    """
    ok = True
    sections_by_doc: dict[str, list[tuple[str | None, str]]] = {}

    for row in rows:
        if row.is_off_topic:
            continue
        if not row.ground_truth_doc_id or not row.ground_truth_ref:
            logger.error("Row '%s': empty ground_truth_doc_id or ground_truth_ref.", row.row_id)
            ok = False
            continue

        if row.ground_truth_doc_id not in sections_by_doc:
            sections_by_doc[row.ground_truth_doc_id] = _scroll_sections(
                client, config, row.ground_truth_doc_id
            )
        sections = sections_by_doc[row.ground_truth_doc_id]

        if not any(
            chunk_matches(row.ground_truth_doc_id, seccion, tipo, row) for seccion, tipo in sections
        ):
            logger.error(
                "Row '%s': ref '%s' in doc '%s' does not match any actual chunk.",
                row.row_id,
                row.ground_truth_ref,
                row.ground_truth_doc_id,
            )
            ok = False

    return ok


def _scroll_sections(
    client: QdrantClient, config: AppConfig, doc_id: str
) -> list[tuple[str | None, str]]:
    """Return every `(seccion, tipo_seccion)` pair stored for one document."""
    doc_filter = models.Filter(
        must=[
            models.FieldCondition(
                key=config.qdrant.doc_id_field, match=models.MatchValue(value=doc_id)
            )
        ]
    )

    pairs: list[tuple[str | None, str]] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=config.qdrant.collection_name,
            scroll_filter=doc_filter,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            metadata = (point.payload or {}).get("metadata", {})
            pairs.append((metadata.get("sección"), metadata.get("tipo_seccion", "none")))
        if offset is None:
            break

    return pairs


def evaluate_retriever_row(
    row: EvalRow, condensed_query: str, retriever: HybridRetriever, k_grid: Sequence[int]
) -> dict:
    """Retrieve at every k in the grid and rank the ground truth in each, independently.

    Retrieves on `condensed_query`, not `row.question`: production always retrieves on the query
    processor's output, so evaluating on the raw question would score a different pipeline.

    One retrieve call per k, not a slice of one k_max call: hybrid RRF fuses per-branch
    prefetches whose limits scale with k, so the top-10 of a k=50 query is not the same as a
    real k=10 query.
    """
    ranks: dict[int, int | None] = {}
    times: list[float] = []
    for k in k_grid:
        started = time.perf_counter()
        chunks = retriever.retrieve(condensed_query, row.exam, [], k)
        times.append((time.perf_counter() - started) * 1000)
        ranks[k] = rank_of_ground_truth(chunks, row)

    return {
        "row_id": row.row_id,
        "exam": row.exam,
        "question_type": row.question_type,
        "condensed_query": condensed_query,
        "ranks": ranks,
        "retrieve_ms": statistics.fmean(times),
    }


def evaluate_retriever(
    rows: Sequence[EvalRow],
    condensed_queries: Mapping[str, str],
    retriever: HybridRetriever,
    k_grid: Sequence[int],
    path: str,
) -> None:
    """Rank the ground truth in the retriever's own output, for every graded row."""
    reset(path)
    logger.info("Evaluating the retriever for %d rows.", len(rows))

    for row in rows:
        append_record(path, evaluate_retriever_row(row, condensed_queries[row.row_id], retriever, k_grid))


@dataclass(frozen=True)
class StageMetrics:
    """Hit rate for the retriever at one k."""

    k: int
    n_rows: int
    hit_rate: float


def summarize_retriever(records: Sequence[dict], k_grid: Sequence[int]) -> list[StageMetrics]:
    """Hit Rate@k for the retriever sweep: each k has its own independently-computed rank."""
    metrics: list[StageMetrics] = []
    for k in k_grid:
        ranks = [record["ranks"].get(str(k), record["ranks"].get(k)) for record in records]
        metrics.append(StageMetrics(k=k, n_rows=len(records), hit_rate=hit_rate(ranks, k)))
    return metrics
