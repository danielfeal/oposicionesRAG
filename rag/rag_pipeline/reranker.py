"""CPU cross-encoder reranking of retrieved chunks, plus selecting the final
subset that gets passed to generation.
"""

import logging
import math
from typing import Sequence

import torch
from sentence_transformers import CrossEncoder

from rag.config import RetrievalConfig
from rag.rag_pipeline.types import RetrievedChunk

logger = logging.getLogger(__name__)


def _sigmoid(x: float) -> float:
    """Numerically stable logistic function."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _estimate_tokens(text: str, chars_per_token: float) -> int:
    """Approximate token count as ceil(len(text) / chars_per_token)."""
    return -(-len(text) // int(chars_per_token)) if chars_per_token >= 1 else len(text)


class CrossEncoderReranker:
    """CPU cross-encoder reranker over (query, passage) pairs."""

    def __init__(self, config: RetrievalConfig) -> None:
        torch.set_num_threads(config.threads)
        self._model = CrossEncoder(config.rerank_model, max_length=config.rerank_max_length)
        self._config = config

    def rerank(
        self, query: str, chunks: Sequence[RetrievedChunk]
    ) -> tuple[list[RetrievedChunk], list[RetrievedChunk], int]:
        """Score, sort, and filter chunks.

        Scoring failures are absorbed here rather than raised: the caller
        falls back to retrieval order (no `rerank_score`/`rerank_prob`), and
        filtering below degrades accordingly - its relative-score cutoff
        self-skips when `rerank_prob` is None.
        """
        scored = self._score_and_sort(query, chunks)
        filtered, n_tokens = self._filter(scored)
        return scored, filtered, n_tokens

    def _score_and_sort(self, query: str, chunks: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
        """Score every chunk against the query and return them sorted, best first.

        Returns `chunks` unscored, in retrieval order, if the model call fails.
        """
        if not chunks:
            return []

        try:
            pairs = [(query, chunk.content) for chunk in chunks]
            logits = self._model.predict(pairs, batch_size=self._config.rerank_batch_size)
        except Exception:
            logger.exception("Reranking failed; using retrieval order.")
            return list(chunks)

        scored: list[RetrievedChunk] = []
        for chunk, logit in zip(chunks, logits):
            logit_value = float(logit)
            chunk.rerank_score = logit_value
            chunk.rerank_prob = _sigmoid(logit_value)
            scored.append(chunk)

        scored.sort(key=lambda c: c.rerank_prob, reverse=True)
        return scored

    def _filter(self, chunks: Sequence[RetrievedChunk]) -> tuple[list[RetrievedChunk], int]:
        """Select the final chunks from scored (or fallback-ordered) candidates.

        Applies three limits, whichever binds first:

        1. Relative score cutoff: keep chunks with `rerank_prob >= keep_ratio * top_prob`.
        2. `max_chunks`: hard cap on the final count.
        3. `max_context_tokens`: stop adding once the running estimate would
           exceed it; never emits a partially-truncated chunk.

        Returns no chunks when the input is empty or, with scores available,
        the top chunk's probability is below `min_relevance_prob`.
        """
        if not chunks:
            return [], 0

        top_prob = chunks[0].rerank_prob
        if top_prob is not None and top_prob < self._config.min_relevance_prob:
            return [], 0

        candidates = list(chunks)
        if top_prob is not None:
            threshold = self._config.keep_ratio * top_prob
            candidates = [
                c for c in candidates if c.rerank_prob is not None and c.rerank_prob >= threshold
            ]

        selected: list[RetrievedChunk] = []
        running_tokens = 0
        for chunk in candidates[: self._config.max_chunks]:
            chunk_tokens = _estimate_tokens(chunk.content, self._config.chars_per_token)
            if selected and running_tokens + chunk_tokens > self._config.max_context_tokens:
                break
            selected.append(chunk)
            running_tokens += chunk_tokens

        return selected, running_tokens
