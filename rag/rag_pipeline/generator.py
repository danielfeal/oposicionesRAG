"""Context rendering and streamed answer generation."""

from dataclasses import dataclass, field
from typing import AsyncIterator, Sequence

from rag.config import RetrievalConfig
from rag.rag_pipeline.llm_utils import OllamaClient
from rag.rag_pipeline.types import RetrievedChunk, SourceRef

GENERATION_SYSTEM_PROMPT = (
    "Eres un tutor que ayuda a estudiantes a preparar exámenes de oposiciones a la "
    "Administración Pública española. Respondes SOLO con la información contenida en "
    "los bloques de contexto numerados que se te proporcionan, nunca con conocimiento "
    "previo.\n\n"
    "Cada bloque tiene un índice entre corchetes (ej. [1]). Cuando una afirmación se "
    "apoye en un bloque, cita su indice entre corchetes justo despues de la afirmacion, "
    "por ejemplo: 'El plazo es de tres meses [1].' No escribas nunca URLs, nombres de "
    "documento ni fechas: la lista de fuentes se añade aparte, tú solo citas el índice.\n\n"
    "Las respuestas tienden a estar contenidas de forma literal en una única fuente. "
    "Prioriza citar una única fuente por afirmación y responder con el contenido de la "
    "fuente de forma textual (no es necesario usar comillas). Cita varias fuentes sólo "
    "si aportan información relevante, que contradice o matiza la fuente principal. Sé "
    "conciso: ve directo a la respuesta, sin introducciones ni resúmenes finales. No "
    "sacrifiques matices legales relevantes por acortar.\n\n"
    "Ejemplo de respuestas buenas:\n"
    "- 'El plazo de alegaciones es de diez días hábiles [1], salvo en el procedimiento "
    "sancionador simplificado, donde se reduce a cinco [2].'\n"
    "- 'El Bono Alquiler Joven es de aplicación a todas las comunidades autónomas y las "
    "ciudades de Ceuta y Melilla, con excepción del País Vasco y Navarra. [1]'\n\n"
    "Ejemplo de respuesta a evitar (por redundante):\n"
    "'Es importante destacar que, según la normativa vigente, el plazo establecido para "
    "la presentación de alegaciones... En resumen, el plazo es de diez días [1].'\n\n"
    "Si el contexto no contiene la respuesta, responde exactamente: 'No dispongo de "
    "información suficiente en el temario para responder a esa pregunta.' y no cites nada.\n\n"
    "Responde siempre en español."
)

REFUSAL_ANSWER = "No dispongo de información suficiente en el temario para responder a esa pregunta."


@dataclass(frozen=True)
class BuiltContext:
    """Final context handed to the answer generator."""

    chunks: list[RetrievedChunk] = field(default_factory=list)  # survivors, final order
    blocks: str = ""  # numbered text handed to the LLM
    sources: list[SourceRef] = field(default_factory=list)  # index -> citation metadata
    n_tokens_estimate: int = 0


def _render(chunks: Sequence[RetrievedChunk]) -> tuple[str, list[SourceRef]]:
    """Render numbered context blocks and the matching source list.

    The LLM only ever sees the index and the passage text - never a URL,
    document name or date - so citations are resolved deterministically in
    code from `sources`, immune to the model mangling or inventing a URL.
    """
    lines: list[str] = []
    sources: list[SourceRef] = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.metadata
        heading = " — ".join(part for part in (meta.doc_name, meta.seccion) if part)
        lines.append(f"[{i}] {heading}\n{chunk.content}")
        sources.append(
            SourceRef(
                index=i,
                doc_name=meta.doc_name,
                seccion=meta.seccion,
                source_url=meta.source_url,
                doc_id=meta.doc_id,
                point_id=chunk.point_id,
            )
        )
    return "\n\n".join(lines), sources


class AnswerGenerator:
    """Streams a Spanish answer grounded strictly in the numbered context blocks."""

    def __init__(self, llm: OllamaClient, config: RetrievalConfig) -> None:
        self._llm = llm
        self._config = config

    @staticmethod
    def render_context(chunks: Sequence[RetrievedChunk], n_tokens_estimate: int) -> BuiltContext:
        """Render the reranker's selected chunks into numbered prompt blocks."""
        if not chunks:
            return BuiltContext()
        blocks, sources = _render(chunks)
        return BuiltContext(
            chunks=list(chunks), blocks=blocks, sources=sources, n_tokens_estimate=n_tokens_estimate
        )

    async def stream(
        self, query: str, context: BuiltContext, history: Sequence[dict[str, str]]
    ) -> AsyncIterator[str]:
        """Yield answer fragments as they arrive from Ollama."""
        if not context.chunks:
            yield REFUSAL_ANSWER
            return

        messages = [{"role": "system", "content": GENERATION_SYSTEM_PROMPT}]
        messages.extend(list(history)[-self._config.max_history_turns :])
        messages.append(
            {"role": "user", "content": f"Contexto:\n{context.blocks}\n\nPregunta: {query}"}
        )

        async for fragment in self._llm.stream_chat(messages):
            yield fragment
