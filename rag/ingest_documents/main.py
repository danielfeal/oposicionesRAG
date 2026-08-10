"""End-to-end pipeline: read documents, segment them into chunks, and ingest them
into Qdrant.
"""

import json
import logging

import pandas as pd

from rag.ingest_documents.document_chunker import DocumentChunker
from rag.ingest_documents.document_reader import DocumentReader
from rag.ingest_documents.qdrant_ingestor import QdrantIngestor

RAW_DIR = "./corpus/raw"
EXTRACTED_DIR = "./corpus/extracted_text"
FROM_PDF = False
METADATA_CSV = "./corpus/metadata.csv"
REVIEW_JSON = "./documentation/chunking_review.json"

logger = logging.getLogger(__name__)


def main() -> None:
    """Run the full pipeline: read -> chunk -> write review -> ingest."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Read docs and metadata
    source_dir = RAW_DIR if FROM_PDF else EXTRACTED_DIR
    export_dir = EXTRACTED_DIR if FROM_PDF else None

    docs = DocumentReader().run(source_dir, from_pdf=FROM_PDF, export_dir=export_dir)
    logger.info("Read %d documents from '%s'.", len(docs), source_dir)

    metadata_df = pd.read_csv(METADATA_CSV, dtype=str, keep_default_na=False)

    # Process the corpus: obtain chunks and review
    corpus_chunks, chunking_review = DocumentChunker().run(docs, metadata_df)
    logger.info("Segmented %d documents into %d chunks.", len(chunking_review), len(corpus_chunks))

    with open(REVIEW_JSON, "w", encoding="utf-8") as f:
        json.dump(chunking_review, f, ensure_ascii=False, indent=2)

    # Ingest in Qdrant
    QdrantIngestor().run(corpus_chunks)


if __name__ == "__main__":
    main()
