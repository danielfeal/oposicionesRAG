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
    batch_size: int = 100

    # Dotted payload paths for the metadata fields we filter/index on. Nested
    # under QdrantVectorStore's own default metadata_payload_key ("metadata").
    doc_id_field: str = "metadata.doc_id"
    exams_id_field: str = "metadata.exams_id"
    tema_field: str = "metadata.tema"


@dataclass(frozen=True)
class EmbeddingsConfig:
    """Embedding models used for hybrid (dense + sparse) search."""

    dense_model: str = "intfloat/multilingual-e5-small"
    sparse_model: str = "Qdrant/bm25"


@dataclass(frozen=True)
class RetrievalConfig:
    """Retrieval, reranking and context-selection parameters."""

    top_k: int = 10
    rerank_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    rerank_batch_size: int = 8
    rerank_max_length: int = 512
    threads: int = 2  # Google Cloud e2-highmem-2: 2 vCPUs, no GPU
    max_chunks: int = 4
    max_context_tokens: int = 3000
    chars_per_token: float = 3.6  # Spanish heuristic; no local tokenizer available
    min_relevance_prob: float = 0.15  # NO_CONTEXT gate on sigmoid
    keep_ratio: float = 0.50  # drop chunks scoring below keep_ratio * top prob
    max_history_turns: int = 6


@dataclass(frozen=True)
class LlmConfig:
    """Ollama connection and default generation parameters.

    Two model tags: `model` answers the question, `relevance_model` only
    classifies on/off-topic (a much easier task, therefore a smaller, faster model)
    """

    base_url: str = os.getenv("RAG_LLM_BASE_URL", "http://localhost:11434")
    model: str = "qwen3.5:2b"
    relevance_model: str = "qwen3.5:0.8b"
    think: bool = False  # qwen3(.5) emits <think> blocks otherwise -> huge CPU cost
    timeout_s: int = 180
    keep_alive: str = "60m"  # How long Ollama keeps the model resident after the last request.
    temperature: float = 0.2
    num_predict: int = 800
    num_ctx: int = 8192


@dataclass(frozen=True)
class AppConfig:
    """Root configuration shared by ingestion and the query-side pipeline."""

    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    chunk_overlap_tokens: int = 50
    metadata_csv: str = "./corpus/metadata.csv"
    raw_dir: str = "./corpus/raw"
    extracted_dir: str = "./corpus/extracted_text"
    chunking_review_json: str = "./documentation/chunking_review.json"
    trace_db: str = os.getenv("RAG_TRACE_DB", "./documentation/interactions.db")
