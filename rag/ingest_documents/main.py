"""End-to-end pipeline: read documents, segment them into chunks, and ingest them
into Qdrant.
"""

import json
import logging

import pandas as pd

from rag.config import AppConfig
from rag.ingest_documents.document_chunker import DocumentChunker
from rag.ingest_documents.document_reader import DocumentReader
from rag.ingest_documents.qdrant_ingestor import QdrantIngestor

FROM_PDF = False

logger = logging.getLogger(__name__)


def main() -> None:
    """Run the full pipeline: read -> chunk -> write review -> ingest."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config = AppConfig()

    # Read docs and metadata
    source_dir = config.raw_dir if FROM_PDF else config.extracted_dir
    export_dir = config.extracted_dir if FROM_PDF else None

    docs = DocumentReader().run(source_dir, from_pdf=FROM_PDF, export_dir=export_dir)
    logger.info("Read %d documents from '%s'.", len(docs), source_dir)

    metadata_df = pd.read_csv(config.metadata_csv, dtype=str, keep_default_na=False)

    # Process the corpus: obtain chunks and review
    chunker = DocumentChunker(
        model_name=config.embeddings.dense_model, overlap_tokens=config.chunk_overlap_tokens
    )
    corpus_chunks, chunking_review = chunker.run(docs, metadata_df)
    logger.info("Segmented %d documents into %d chunks.", len(chunking_review), len(corpus_chunks))

    with open(config.chunking_review_json, "w", encoding="utf-8") as f:
        json.dump(chunking_review, f, ensure_ascii=False, indent=2)

    # Ingest in Qdrant
    QdrantIngestor(config).run(corpus_chunks)


if __name__ == "__main__":
    main()
