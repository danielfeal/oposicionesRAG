"""Shared configuration for ingestion and query.

Model names, collection name and vector names must be identical on both sides -
a mismatch degrades retrieval silently instead of raising. That is why they
live here and not in per-class constructor defaults.
"""

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class QdrantConfig:
    """Connection and payload-layout settings for the Qdrant collection."""

    host: str = os.getenv("RAG_QDRANT_HOST", "localhost")
    port: int = int(os.getenv("RAG_QDRANT_PORT", "6333"))
    collection_name: str = "oposiciones_rag"
    dense_vector_name: str = "dense"
    sparse_vector_name: str = "sparse"
    batch_size: int = 50

    # Dotted payload paths for the metadata fields we filter/index on. Nested
    # under QdrantVectorStore's own default metadata_payload_key ("metadata").
    doc_id_field: str = "metadata.doc_id"
    exams_id_field: str = "metadata.exams_id"
    tema_field: str = "metadata.tema"


@dataclass(frozen=True)
class EmbeddingsConfig:
    """Embedding models used for hybrid (dense + sparse) search."""

    dense_model: str = "BAAI/bge-m3"
    sparse_model: str = "Qdrant/bm25"


@dataclass(frozen=True)
class RetrievalConfig:
    """Retrieval and context-selection parameters."""

    top_k: int = 5
    max_context_tokens: int = 3000
    chars_per_token: float = 3.6  # Spanish heuristic; no local tokenizer available
    max_history_turns: int = 6


@dataclass(frozen=True)
class LlmConfig:
    """Gemini (Vertex AI) generation parameters.

    One model tag for both stages: query processing (relevance + condensation) and generation.
    Query processing needs real capability - it must faithfully reproduce most of the query
    unchanged, not just classify it - so both stages share the same model.
    """

    model: str = "gemini-3.5-flash-lite"
    timeout_s: int = 30
    temperature: float = 0.2
    max_output_tokens: int = 800


@dataclass(frozen=True)
class AppConfig:
    """Root configuration shared by ingestion and the query-side pipeline."""

    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    chunk_overlap_tokens: int = 100
    chunk_max_length_tokens: int = 1024  # fixed cap, independent of the embedding model's own limit
    metadata_csv: str = "./corpus/metadata.csv"
    raw_dir: str = "./corpus/raw"
    extracted_dir: str = "./corpus/extracted_text"
    chunking_review_json: str = "./documentation/chunking_review.json"
    trace_db: str = os.getenv("RAG_TRACE_DB", "./documentation/interactions.db")
