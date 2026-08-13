import logging

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import (
    FastEmbedSparse,
    QdrantVectorStore,
    RetrievalMode,
    SparseEmbeddings,
)
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)
from qdrant_client.models import FieldCondition, Filter, MatchAny
from tqdm import tqdm

from rag.config import AppConfig
from rag.ingest_documents.document_chunker import ChunkRecord

logger = logging.getLogger(__name__)


def build_vector_store(
    client: QdrantClient,
    config: AppConfig,
    dense_embeddings: Embeddings,
    sparse_embeddings: SparseEmbeddings,
) -> QdrantVectorStore:
    """Build the hybrid (dense + sparse) vector store used by BOTH ingestion and
    retrieval.

    Model names, vector names and payload keys must be identical on both sides -
    a mismatch degrades retrieval silently instead of raising. Do not add e5
    "query:"/"passage:" prefixes on one side only: the corpus was embedded
    without them and a one-sided change silently degrades retrieval.

    Args:
        client: Connected Qdrant client.
        config: Shared application config; only `config.qdrant` is used.
        dense_embeddings: Dense embedding model, matching the one used at
            ingestion time.
        sparse_embeddings: Sparse embedding model, matching the one used at
            ingestion time.

    Returns:
        A `QdrantVectorStore` configured for hybrid retrieval.
    """
    qdrant_config = config.qdrant
    return QdrantVectorStore(
        client=client,
        collection_name=qdrant_config.collection_name,
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name=qdrant_config.dense_vector_name,
        sparse_vector_name=qdrant_config.sparse_vector_name,
        content_payload_key=qdrant_config.content_payload_key,
        metadata_payload_key=qdrant_config.metadata_payload_key,
    )


class QdrantIngestor:
    """Embeds chunk records (dense + sparse) and upserts them into a Qdrant collection."""

    def __init__(self, config: AppConfig) -> None:
        """Connect to Qdrant and load the embedding models from shared config.

        Args:
            config: Shared application config.
        """
        self._config = config
        self._qdrant_config = config.qdrant

        self._client = QdrantClient(host=self._qdrant_config.host, port=self._qdrant_config.port)
        self._dense_embeddings = HuggingFaceEmbeddings(model_name=config.embeddings.dense_model)
        self._sparse_embeddings = FastEmbedSparse(model_name=config.embeddings.sparse_model)
        self._dense_dim = len(self._dense_embeddings.embed_query("dim probe"))

    def run(self, corpus_chunks: list[ChunkRecord], doc_ids: list[str] | None = None) -> None:
        """Create or update the Qdrant collection, optionally scoped to a subset of
        doc_ids.

        Args:
            corpus_chunks: Chunk records to ingest.
            doc_ids: If given, only re-ingest chunks whose `doc_id` is in this list -
                existing points for those doc_ids are deleted first and the rest of
                the collection is left untouched. If omitted, the whole collection is
                recreated from `corpus_chunks`.
        """
        self._ensure_collection(doc_ids)
        vector_store = build_vector_store(
            self._client, self._config, self._dense_embeddings, self._sparse_embeddings
        )

        chunks_to_ingest = [c for c in corpus_chunks if c["doc_id"] in doc_ids] if doc_ids else corpus_chunks
        qdrant_docs = self._to_documents(chunks_to_ingest)

        for i in tqdm(range(0, len(qdrant_docs), self._qdrant_config.batch_size), desc="Uploading to Qdrant"):
            batch = qdrant_docs[i:i + self._qdrant_config.batch_size]
            vector_store.add_documents(documents=batch)

        logger.info("Ingested %d chunks into '%s'.", len(qdrant_docs), self._qdrant_config.collection_name)

    def _ensure_collection(self, doc_ids: list[str] | None) -> None:
        """Delete stale points/collection as needed, then (re)create the collection
        and its payload indexes if it doesn't already exist.

        Args:
            doc_ids: Scope of the deletion; see `run`.
        """
        collection_name = self._qdrant_config.collection_name
        payload_indexes = {
            self._qdrant_config.doc_id_field: "keyword",
            self._qdrant_config.exams_id_field: "keyword",
            self._qdrant_config.tema_field: "keyword",
        }

        if self._client.collection_exists(collection_name):
            if doc_ids:
                result = self._client.delete(
                    collection_name=collection_name,
                    points_selector=Filter(
                        must=[
                            FieldCondition(
                                key=self._qdrant_config.doc_id_field,
                                match=MatchAny(any=doc_ids),
                            )
                        ]
                    ),
                )
                logger.info(
                    "Deleted points for %d doc_ids from '%s': %s",
                    len(doc_ids), collection_name, result.status,
                )
            else:
                self._client.delete_collection(collection_name)
                logger.info("Dropped collection '%s'.", collection_name)

        if not self._client.collection_exists(collection_name):
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    self._qdrant_config.dense_vector_name: VectorParams(
                        size=self._dense_dim, distance=Distance.COSINE
                    ),
                },
                sparse_vectors_config={
                    self._qdrant_config.sparse_vector_name: SparseVectorParams(
                        index=SparseIndexParams(on_disk=False)
                    ),
                },
            )
            for field, schema in payload_indexes.items():
                self._client.create_payload_index(
                    collection_name=collection_name, field_name=field, field_schema=schema
                )
            logger.info("Created collection '%s'.", collection_name)

    def _to_documents(self, chunks: list[ChunkRecord]) -> list[Document]:
        """Convert chunk records into langchain `Document`s for ingestion."""
        return [
            Document(
                page_content=chunk["content"],
                metadata={
                    "doc_id": chunk["doc_id"],
                    "doc_name": chunk["doc_name"],
                    "doc_date": chunk["doc_date"],
                    "source_url": chunk["source_url"],
                    "exams_id": chunk["exams_id"],
                    "tema": chunk["tema"],
                    "tipo_seccion": chunk["tipo_seccion"],
                    "sección": chunk["sección"],
                },
            )
            for chunk in chunks
        ]
