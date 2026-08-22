"""Relevance-check pipeline stage: classifies a query as on- or off-topic."""

import logging

from rag.rag_pipeline.llm_utils import OllamaClient
from rag.rag_pipeline.types import RelevanceVerdict

logger = logging.getLogger(__name__)

RELEVANCE_SYSTEM_PROMPT = (
    "Eres un clasificador. El asistente al que sirves responde preguntas sobre "
    "los exámenes de oposiciones a la Administración Pública española utilizando "
    "el temario oficial, compuesto principalmente de artículos del BOE. Tu única"
    "tarea es decidir si la pregunta del usuario está relacionada con ese dominio. "
    "Responde con una sola palabra: SI o NO. No expliques tu respuesta."
)


class RelevanceChecker:
    """Classifies a query as on- or off-topic for the oposiciones corpus.

    A retrieval-score threshold cannot do this: hybrid RRF scores are
    rank-based and uncalibrated, so an off-topic query still yields ~the same
    top score as an on-topic one.
    """

    def __init__(self, llm: OllamaClient) -> None:
        self._llm = llm

    def check(self, query: str) -> RelevanceVerdict:
        """Classify `query` as RELEVANT, OFF_TOPIC, or UNKNOWN.

        Fails open: any error or unparseable reply is treated as UNKNOWN,
        which the pipeline continues on as if RELEVANT, so an Ollama hiccup
        degrades into a normal retrieval rather than a false rejection.
        """
        messages = [
            {"role": "system", "content": RELEVANCE_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
        try:
            reply = self._llm.chat(messages, temperature=0.0, num_predict=4)
        except Exception:
            logger.exception("Relevance check failed for query %r.", query)
            return RelevanceVerdict.UNKNOWN

        normalized = reply.strip().upper()
        if normalized.startswith("SI"):
            return RelevanceVerdict.RELEVANT
        if normalized.startswith("NO"):
            return RelevanceVerdict.OFF_TOPIC
        logger.warning("Unparseable relevance reply %r for query %r.", reply, query)
        return RelevanceVerdict.UNKNOWN
