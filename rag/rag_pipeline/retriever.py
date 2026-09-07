"""Hybrid retrieval over the ingestion collection."""

from typing import Sequence

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import models

from rag.config import AppConfig, QdrantConfig
from rag.rag_pipeline.types import ChunkMetadata, RetrievedChunk


def build_filter(
    config: QdrantConfig, exam: str, temas: Sequence[str] | None = None
) -> models.Filter:
    """Build the retrieval filter."""
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
        self._vector_store = vector_store
        self._config = config

    def retrieve(
        self, query: str, exam: str, temas: Sequence[str], k: int
    ) -> list[RetrievedChunk]:
        """Return up to `k` full-length passages matching the filter."""

        effective_temas = self._effective_temas(exam, temas)
        qdrant_filter = build_filter(self._config.qdrant, exam, effective_temas)
        results = self._vector_store.similarity_search_with_score(query=query, k=k, filter=qdrant_filter)
        return [self._to_chunk(document, score) for document, score in results]

    def _effective_temas(self, exam: str, temas: Sequence[str]) -> list[str]:
        """Drop the tema filter for non-A1 exams, where tema is always empty."""
        if exam != "A1" or not temas:
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
