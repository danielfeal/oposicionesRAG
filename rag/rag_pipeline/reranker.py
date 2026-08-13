"""CPU cross-encoder reranking of retrieved chunks."""

import math
from typing import Sequence

import torch
from sentence_transformers import CrossEncoder

from rag.config import RetrievalConfig
from rag.rag_pipeline.types import RetrievedChunk


def _sigmoid(x: float) -> float:
    """Numerically stable logistic function."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class CrossEncoderReranker:
    """CPU cross-encoder reranker over (query, passage) pairs.

    The model must be multilingual: the corpus is Spanish. The default,
    `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, is mMARCO-trained (includes
    Spanish), ~470 MB, 12 layers, Apache-licensed - a deliberately cheaper
    choice than 24-layer multilingual rerankers, since this stage runs on a
    4-core ARM CPU with no GPU.
    """

    def __init__(self, config: RetrievalConfig) -> None:
        """
        Args:
            config: Shared retrieval config (model, batch size, max length,
                thread count).
        """
        torch.set_num_threads(config.threads)
        self._model = CrossEncoder(config.rerank_model, max_length=config.rerank_max_length)
        self._batch_size = config.rerank_batch_size

    def rerank(self, query: str, chunks: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
        """Score every chunk against the query and return them sorted, best first.

        Args:
            query: User query.
            chunks: Candidate chunks from retrieval.

        Returns:
            The same chunks, sorted best-first, with `rerank_score` (raw
            logit) and `rerank_prob` (sigmoid of the logit - the calibrated
            signal used for the NO_CONTEXT gate and the relative cutoff)
            populated. Empty input returns an empty list without touching the
            model.
        """
        if not chunks:
            return []

        pairs = [(query, chunk.content) for chunk in chunks]
        logits = self._model.predict(pairs, batch_size=self._batch_size)

        scored: list[RetrievedChunk] = []
        for chunk, logit in zip(chunks, logits):
            logit_value = float(logit)
            chunk.rerank_score = logit_value
            chunk.rerank_prob = _sigmoid(logit_value)
            scored.append(chunk)

        scored.sort(key=lambda c: c.rerank_prob, reverse=True)
        return scored


class NoOpReranker:
    """Identity reranker: preserves retrieval order, leaves rerank_score None."""

    def rerank(self, query: str, chunks: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
        """Return `chunks` unchanged, as a list."""
        return list(chunks)
