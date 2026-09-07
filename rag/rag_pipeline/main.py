"""End-to-end query pipeline: query processing -> retrieve -> select context -> generate.

CLI usage:
    python -m rag.rag_pipeline.main --exam A1 --query "..."
    python -m rag.rag_pipeline.main --exam A1 --tema II --tema V --query "..."
"""

import argparse
import asyncio
import logging
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import AsyncIterator
from uuid import uuid4

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import FastEmbedSparse
from qdrant_client import QdrantClient

from rag.config import AppConfig
from rag.ingest_documents.qdrant_ingestor import build_vector_store
from rag.rag_pipeline.generator import AnswerGenerator
from rag.rag_pipeline.llm_utils import GeminiClient
from rag.rag_pipeline.query_processor import QueryProcessingResult, QueryProcessor
from rag.rag_pipeline.retriever import HybridRetriever
from rag.rag_pipeline.trace_store import NullTraceStore, SqliteTraceStore
from rag.rag_pipeline.types import (
    STATUS_MESSAGES,
    PipelineStatus,
    PipelineTrace,
    RelevanceVerdict,
)

logger = logging.getLogger(__name__)


@dataclass
class Components:
    """Concrete, already-constructed dependencies `answer()` orchestrates."""

    retriever: HybridRetriever
    query_processor: QueryProcessor
    generator: AnswerGenerator
    trace_store: SqliteTraceStore | NullTraceStore
    config: AppConfig


def build_retrieval_only(config: AppConfig) -> HybridRetriever:
    """Construct the retriever: Qdrant client, embeddings, vector store."""
    client = QdrantClient(host=config.qdrant.host, port=config.qdrant.port)
    dense_embeddings = HuggingFaceEmbeddings(model_name=config.embeddings.dense_model)
    sparse_embeddings = FastEmbedSparse(model_name=config.embeddings.sparse_model)
    vector_store = build_vector_store(client, config, dense_embeddings, sparse_embeddings)

    return HybridRetriever(vector_store, config)


def build_llm(config: AppConfig) -> tuple[QueryProcessor, AnswerGenerator]:
    """Construct the two Gemini-backed stages, sharing one client and one model."""
    llm = GeminiClient(config.llm)
    return QueryProcessor(llm), AnswerGenerator(llm, config.retrieval)


def build_components(config: AppConfig | None = None) -> Components:
    """Assemble the production components from configuration."""
    config = config or AppConfig()

    retriever = build_retrieval_only(config)
    query_processor, generator = build_llm(config)
    trace_store = SqliteTraceStore(config.trace_db) if config.trace_db else NullTraceStore()

    return Components(
        retriever=retriever,
        query_processor=query_processor,
        generator=generator,
        trace_store=trace_store,
        config=config,
    )


def new_trace(
    config: AppConfig, exam: str, temas: list[str], query: str, session_id: str | None = None
) -> PipelineTrace:
    """Build the trace."""

    return PipelineTrace(
        session_id=session_id or uuid4().hex,
        timestamp_utc=datetime.now(UTC).isoformat(),
        exam=exam,
        temas=temas,
        raw_query=query,
        condensed_query=query,  # overwritten by the query processor once answer() runs
        relevance=RelevanceVerdict.UNKNOWN,
        status=PipelineStatus.OK,
        models={
            "dense": config.embeddings.dense_model,
            "sparse": config.embeddings.sparse_model,
            "llm": config.llm.model,
        },
    )


async def answer(
    components: Components,
    trace: PipelineTrace,
    history: list[dict[str, str]] | None = None,
    query_result: QueryProcessingResult | None = None,
) -> AsyncIterator[str]:
    """Stream an answer, mutating `trace` in place as each stage completes.

    The caller builds `trace` via `new_trace()` and should read it once this
    generator is exhausted - it is live-mutated during streaming, not
    returned, so the trace reflects whatever state the pipeline reached even
    on early exit (off-topic, no context, or an error).

    Pass `query_result` to reuse an already-computed relevance verdict + condensation instead of
    calling the query processor again - the evaluation harness's generation phase does this to
    reuse `run_query_processing`'s output rather than paying for query processing twice.

    Never lets an exception escape: failures become a status on `trace`, and
    the trace is always logged via `components.trace_store` before this
    generator ends.
    """
    history = list(history or [])
    start = time.perf_counter()

    try:
        # Query processing
        if query_result is None:
            t0 = time.perf_counter()
            query_result = await asyncio.to_thread(
                components.query_processor.process, trace.condensed_query, history
            )
            trace.durations_ms["relevance"] = (time.perf_counter() - t0) * 1000
        trace.condensed_query = query_result.condensed_query
        trace.relevance = query_result.relevance
        if trace.relevance == RelevanceVerdict.OFF_TOPIC:
            trace.status = PipelineStatus.OFF_TOPIC
            trace.message = STATUS_MESSAGES[PipelineStatus.OFF_TOPIC]
            return

        # Retrieval
        query = trace.condensed_query
        t0 = time.perf_counter()
        try:
            trace.retrieved = await asyncio.to_thread(
                components.retriever.retrieve,
                query,
                trace.exam,
                trace.temas,
                components.config.retrieval.top_k,
            )
        except Exception as exc:
            logger.exception("Retrieval failed for session '%s'.", trace.session_id)
            trace.status = PipelineStatus.RETRIEVAL_ERROR
            trace.message = STATUS_MESSAGES[PipelineStatus.RETRIEVAL_ERROR]
            trace.error = repr(exc)
            return
        trace.durations_ms["retrieve"] = (time.perf_counter() - t0) * 1000
        if not trace.retrieved:
            trace.status = PipelineStatus.NO_CONTEXT
            trace.message = STATUS_MESSAGES[PipelineStatus.NO_CONTEXT]
            return

        # Context filtering and rendering
        selected, n_tokens = AnswerGenerator.filter_context(
            trace.retrieved, components.config.retrieval
        )
        built = components.generator.render_context(selected, n_tokens)
        trace.final_chunks = built.chunks
        trace.sources = built.sources

        # Generation
        t0 = time.perf_counter()
        parts: list[str] = []
        try:
            async for token in components.generator.stream(query, built, history):
                parts.append(token)
                yield token
        except Exception as exc:
            logger.exception("Generation failed for session '%s'.", trace.session_id)
            trace.status = PipelineStatus.GENERATION_ERROR
            trace.message = STATUS_MESSAGES[PipelineStatus.GENERATION_ERROR]
            trace.error = repr(exc)
        finally:
            trace.answer = "".join(parts)
            trace.durations_ms["generate"] = (time.perf_counter() - t0) * 1000
    finally:
        trace.durations_ms["total"] = (time.perf_counter() - start) * 1000
        await asyncio.to_thread(components.trace_store.log, trace)


async def _run_once(
    components: Components, query: str, exam: str, temas: list[str]
) -> PipelineTrace:
    """Stream one answer to stdout and return the completed trace."""
    trace = new_trace(components.config, exam, temas, query)
    async for token in answer(components, trace):
        print(token, end="", flush=True)
    print()
    return trace


def _print_trace(trace: PipelineTrace) -> None:
    """Print status, stage timings and the retrieved chunks with scores."""
    print(f"\nstatus: {trace.status.value}  relevance: {trace.relevance.value}")
    if trace.message:
        print(f"message: {trace.message}")
    print("durations_ms: " + ", ".join(f"{k}={v:.0f}" for k, v in trace.durations_ms.items()))

    if trace.retrieved:
        print("\nretrieved chunks:")
        final_ids = {c.point_id for c in trace.final_chunks}
        print(f"{'doc_id':<10} {'seccion':<30} {'rrf':>8}  kept?")
        for c in trace.retrieved:
            kept = "yes" if c.point_id in final_ids else "no"
            seccion = (c.metadata.seccion or "")[:30]
            print(f"{c.metadata.doc_id:<10} {seccion:<30} {c.retrieval_score:>8.3f}  {kept}")

    if trace.sources:
        print("\nsources:")
        for s in trace.sources:
            print(f"  [{s.index}] {s.doc_name} — {s.seccion} — {s.source_url}")


def main() -> None:
    """CLI entry point."""
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("--exam", required=True, choices=["A1", "A2", "C1", "C2"])
    parser.add_argument("--tema", action="append", default=[], help="Repeatable; A1 only.")
    args = parser.parse_args()

    components = build_components(AppConfig())
    trace = asyncio.run(_run_once(components, args.query, args.exam, args.tema))
    _print_trace(trace)

    sys.exit(0 if trace.status == PipelineStatus.OK else 1)


if __name__ == "__main__":
    main()
