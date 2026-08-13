"""Hybrid retrieval over the ingestion collection.

The vector store is built by the same factory the ingestor uses
(`rag.ingest_documents.qdrant_ingestor.build_vector_store`), so model names,
vector names and payload keys cannot drift between ingest and query. In
particular, e5 "query:"/"passage:" prefixes are NOT applied on either side -
adding them here alone would silently degrade recall.
"""

import logging
from typing import Sequence

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import models

from rag.config import AppConfig, QdrantConfig
from rag.rag_pipeline.types import ChunkMetadata, RetrievedChunk

logger = logging.getLogger(__name__)


def build_filter(
    config: QdrantConfig, exam: str, temas: Sequence[str] | None = None
) -> models.Filter:
    """Build the retrieval filter.

    Args:
        config: Qdrant config, giving the payload paths for exams_id and tema.
        exam: Exactly one exam id (A1 | A2 | C1 | C2). Always applied.
        temas: Optional temas; only meaningful for A1, ignored when empty.

    Returns:
        Filter matching points whose exams_id array contains `exam` and, when
        `temas` is given, whose tema array shares at least one value with it.
    """
    must: list[models.Condition] = [
        models.FieldCondition(key=config.exams_id_field, match=models.MatchValue(value=exam))
    ]
    if temas:
        must.append(
            models.FieldCondition(key=config.tema_field, match=models.MatchAny(any=list(temas)))
        )
    return models.Filter(must=must)


class HybridRetriever:
    """Dense + sparse hybrid retrieval, filtered by exam and optionally tema."""

    def __init__(self, vector_store: QdrantVectorStore, config: AppConfig) -> None:
        """
        Args:
            vector_store: Hybrid vector store, built via
                `rag.ingest_documents.qdrant_ingestor.build_vector_store` so it
                is guaranteed to match ingestion.
            config: Shared application config.
        """
        self._vector_store = vector_store
        self._config = config

    def retrieve(
        self, query: str, exam: str, temas: Sequence[str], k: int
    ) -> list[RetrievedChunk]:
        """Return up to `k` full-length passages matching the filter.

        Args:
            query: User query (already condensed if applicable).
            exam: Exam id to filter by (A1 | A2 | C1 | C2).
            temas: Optional tema filter; dropped for non-A1 exams (see
                `_effective_temas`).
            k: Maximum number of candidates to return.

        Returns:
            Retrieved chunks with their (uncalibrated) hybrid RRF score.
        """
        effective_temas = self._effective_temas(exam, temas)
        qdrant_filter = build_filter(self._config.qdrant, exam, effective_temas)
        results = self._vector_store.similarity_search_with_score(query=query, k=k, filter=qdrant_filter)
        return [self._to_chunk(document, score) for document, score in results]

    def _effective_temas(self, exam: str, temas: Sequence[str]) -> list[str]:
        """Drop the tema filter for non-A1 exams, where tema is always empty.

        Filtering by tema on a non-A1 exam would match zero points (every
        non-A1 document has `tema == []`), which is the worst failure mode
        here: a silent empty result set instead of an unfiltered search.
        """
        if exam != "A1" or not temas:
            if temas:
                logger.debug("Ignoring tema filter %s for non-A1 exam '%s'.", list(temas), exam)
            return []
        return list(temas)

    def _to_chunk(self, document: Document, score: float) -> RetrievedChunk:
        """Map a langchain `Document` to a `RetrievedChunk`, normalizing metadata."""
        point_id = document.metadata.get("_id")
        return RetrievedChunk(
            point_id=str(point_id),
            content=document.page_content,
            metadata=ChunkMetadata.from_payload(document.metadata),
            retrieval_score=score,
        )
