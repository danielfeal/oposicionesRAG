import logging

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)
from qdrant_client.models import FieldCondition, Filter, MatchAny
from tqdm import tqdm

from rag.ingest_documents.document_chunker import ChunkRecord

logger = logging.getLogger(__name__)


class QdrantIngestor:
    """Embeds chunk records (dense + sparse) and upserts them into a Qdrant collection."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "oposiciones_rag",
        dense_model_name: str = "intfloat/multilingual-e5-small",
        sparse_model_name: str = "Qdrant/bm25",
        batch_size: int = 100,
    ) -> None:
        """Connect to Qdrant and load the embedding models.

        Args:
            host: Qdrant host.
            port: Qdrant port.
            collection_name: Name of the collection to create or update.
            dense_model_name: HuggingFace model used for dense embeddings.
            sparse_model_name: FastEmbed model used for sparse (BM25) embeddings.
            batch_size: Number of documents uploaded to Qdrant per batch.
        """
        self._collection_name = collection_name
        self._batch_size = batch_size

        self._client = QdrantClient(host=host, port=port)
        self._dense_embeddings = HuggingFaceEmbeddings(model_name=dense_model_name)
        self._sparse_embeddings = FastEmbedSparse(model_name=sparse_model_name)
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
        vector_store = self._build_vector_store()

        chunks_to_ingest = [c for c in corpus_chunks if c["doc_id"] in doc_ids] if doc_ids else corpus_chunks
        qdrant_docs = self._to_documents(chunks_to_ingest)

        for i in tqdm(range(0, len(qdrant_docs), self._batch_size), desc="Uploading to Qdrant"):
            batch = qdrant_docs[i:i + self._batch_size]
            vector_store.add_documents(documents=batch)

        logger.info("Ingested %d chunks into '%s'.", len(qdrant_docs), self._collection_name)

    def _ensure_collection(self, doc_ids: list[str] | None) -> None:
        """Delete stale points/collection as needed, then (re)create the collection
        and its payload indexes if it doesn't already exist.

        Args:
            doc_ids: Scope of the deletion; see `run`.
        """
        payload_indexes = {"exams_id": "keyword", "tema": "keyword"}

        if self._client.collection_exists(self._collection_name):
            if doc_ids:
                result = self._client.delete(
                    collection_name=self._collection_name,
                    points_selector=Filter(must=[FieldCondition(key="doc_id", match=MatchAny(any=doc_ids))]),
                )
                logger.info(
                    "Deleted points for %d doc_ids from '%s': %s",
                    len(doc_ids), self._collection_name, result.status,
                )
            else:
                self._client.delete_collection(self._collection_name)
                logger.info("Dropped collection '%s'.", self._collection_name)

        if not self._client.collection_exists(self._collection_name):
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config={
                    "dense": VectorParams(size=self._dense_dim, distance=Distance.COSINE),
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False)),
                },
            )
            for field, schema in payload_indexes.items():
                self._client.create_payload_index(
                    collection_name=self._collection_name, field_name=field, field_schema=schema
                )
            logger.info("Created collection '%s'.", self._collection_name)

    def _build_vector_store(self) -> QdrantVectorStore:
        """Build the hybrid (dense + sparse) vector store for the collection."""
        return QdrantVectorStore(
            client=self._client,
            collection_name=self._collection_name,
            embedding=self._dense_embeddings,
            sparse_embedding=self._sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )

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
