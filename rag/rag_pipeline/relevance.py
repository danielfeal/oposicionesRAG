"""Relevance-check pipeline stage: classifies a query as on- or off-topic."""

import logging

from rag.rag_pipeline.llm_utils import OllamaClient
from rag.rag_pipeline.types import RelevanceVerdict

logger = logging.getLogger(__name__)

RELEVANCE_SYSTEM_PROMPT = (
    "Tu única función es recibir preguntas y clasificarlas como relevantes o no relevantes. "
    "Una pregunta es relevante si está relacionada con los exámenes de oposiciones a la "
    "Administración Civil española y con su temario oficial, compuesto principalmente de "
    "artículos del BOE.\n\n"
    "Reglas:\n"
    "1. Si la pregunta incluye opciones de respuesta con letras (a, b, c, d...), "
    "trátala siempre como una pregunta de examen de oposición y responde SI, "
    "aunque el contenido de las opciones te resulte desconocido.\n"
    "2. Ante cualquier duda, responde SI. Rechazar por error una pregunta relevante " 
    "es un fallo mucho más grave que aceptar por error una que no lo es.\n"
    "3. Responde NO solo cuando la pregunta trate con claridad un tema ajeno, sin "
    "ninguna conexión con ese dominio.\n\n"
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
