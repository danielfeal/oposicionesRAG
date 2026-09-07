"""Query-processing pipeline stage: scores domain relevance and produces a retrieval-ready
condensation of the query, in one structured call.

Replaces the earlier SI/NO relevance checker. Merging condensation into the same call avoids a
third sequential LLM call, and the two tasks are naturally the same kind of work: "produce a
clean, standalone version of this query for the rest of the pipeline to use."
"""

import logging
from dataclasses import dataclass
from typing import Literal, Sequence

from pydantic import BaseModel, Field

from rag.pipeline.llm_utils import GeminiClient
from rag.pipeline.types import RelevanceVerdict

logger = logging.getLogger(__name__)

QUERY_PROCESSOR_SYSTEM_PROMPT = """\
# Contexto
Eres el primer paso en un agente que responde preguntas sobre exámenes de oposición a la Administración Pública española. Tienes dos tareas.

# TAREA 1 - Evaluar relevancia.
Evalúa si la pregunta pertenece al dominio adecuado. Preguntas sobre el temario de los exámenes de oposición, legislación española, derecho administrativo, función pública, etc. se consideran relevantes. Responde:
"SI" - La pregunta está relacionada con el dominio.
"NO" - La pregunta no está relacionada con el dominio.

Rechazar por error una pregunta relevante es un fallo mucho más grave que aceptar por error una que no lo es. Ante cualquier duda, responde "SI".

# TAREA 2 - Refinamiento de la pregunta.
El resultado de esta tarea depende de la respuesta a la TAREA 1.

## TAREA 1 = "SI"
Devuelve la pregunta ORIGINAL palabra por palabra, haciendo ÚNICAMENTE estos dos cambios
- Únicamente en el caso de que la pregunta cite una ley o norma de manera explícita, elimina esa referencia de la pregunta, reformulando la pregunta si es necesario para que mantenga el significado original.
- Si hay historial de conversación, sustituye referencias ambiguas ("eso", "ese caso") por aquello a lo que se refieren.

Realiza únicamente estas modificaciones. Si la pregunta contiene posibles respuestas (a, b, c, d...), no las modifiques.

## TAREA 1 = "NO"
Devuelve un string vacío.

Responde siempre en el formato JSON solicitado.

# Ejemplo
PREGUNTA: "De acuerdo con lo establecido en la Ley Orgánica 2/1979, de 3 de octubre, del Tribunal Constitucional, ¿cuántos miembros deben estar presentes para que el Tribunal en Pleno puede adoptar acuerdos?"

RESPUESTA: {"relevance": "SI", "condensed_query": "¿Cuántos miembros deben estar presentes para que el Tribunal Constitucional en Pleno pueda adoptar acuerdos?"}
"""


class _QueryProcessingOutput(BaseModel):
    """Schema for the structured Gemini call. Not exposed outside this module."""

    relevance: Literal["SI", "NO"] = Field(description='"SI" o "NO", ver instrucciones.')
    condensed_query: str = Field(description="Pregunta original con solo los cambios indicados.")


@dataclass(frozen=True)
class QueryProcessingResult:
    """Output of one relevance + condensation call."""

    condensed_query: str
    relevance: RelevanceVerdict

    @classmethod
    def verdict(cls, relevance: Literal["SI", "NO"], condensed_query: str, raw_query: str) -> "QueryProcessingResult":
        """Build from a parsed SI/NO relevance verdict."""
        verdict = RelevanceVerdict.RELEVANT if relevance == "SI" else RelevanceVerdict.OFF_TOPIC
        return cls(condensed_query=condensed_query or raw_query, relevance=verdict)

    @classmethod
    def failed(cls, query: str) -> "QueryProcessingResult":
        """Fail-open result: UNKNOWN relevance, condensed_query falls back to the raw query."""
        return cls(condensed_query=query, relevance=RelevanceVerdict.UNKNOWN)


class QueryProcessor:
    """Scores domain relevance and condenses the query for retrieval, in one structured call.

    Replaces the pipeline's earlier separate relevance-check stage. A retrieval-score threshold
    cannot do relevance classification: hybrid RRF scores are rank-based and uncalibrated, so an
    off-topic query still yields ~the same top score as an on-topic one.
    """

    def __init__(self, llm: GeminiClient) -> None:
        self._llm = llm

    def process(self, query: str, history: Sequence[dict[str, str]] = ()) -> QueryProcessingResult:
        """Score relevance and produce a retrieval-ready condensation of `query`.

        Fails open: any error or unparseable reply is treated as UNKNOWN relevance (the
        pipeline continues as if RELEVANT) and `condensed_query` falls back to the raw query.
        """
        messages = [
            {"role": "system", "content": QUERY_PROCESSOR_SYSTEM_PROMPT},
            {"role": "user", "content": self._render_user_turn(query, history)},
        ]

        try:
            parsed = self._llm.chat_structured(
                messages,
                response_schema=_QueryProcessingOutput,
                temperature=0.0,
                max_output_tokens=600,
            )
        except Exception:
            logger.exception("Query processing failed for query %r.", query)
            return QueryProcessingResult.failed(query)

        return QueryProcessingResult.verdict(parsed.relevance, parsed.condensed_query, query)

    @staticmethod
    def _render_user_turn(query: str, history: Sequence[dict[str, str]]) -> str:
        """Fold conversation history into the user turn, unambiguously separate from the query."""
        if not history:
            return query
        transcript = "\n".join(
            f"{'Usuario' if turn.get('role') == 'user' else 'Asistente'}: {turn.get('content', '')}"
            for turn in history
        )
        return f"Historial:\n{transcript}\n\nPregunta actual: {query}"
