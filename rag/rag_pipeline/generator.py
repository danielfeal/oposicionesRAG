"""Context selection and streamed answer generation."""

from dataclasses import dataclass, field
from typing import AsyncIterator, Sequence

from rag.config import RetrievalConfig
from rag.rag_pipeline.llm import OllamaClient
from rag.rag_pipeline.types import RetrievedChunk, SourceRef

GENERATION_SYSTEM_PROMPT = (
    "Eres un tutor que ayuda a estudiantes a preparar oposiciones a la "
    "Administración Pública española. Respondes SOLO con la informacion "
    "contenida en los bloques de contexto numerados que se te proporcionan, "
    "nunca con conocimiento previo. Cada bloque tiene un indice entre "
    "corchetes, por ejemplo [1]. Cuando una afirmacion de tu respuesta se "
    "apoye en un bloque, cita su indice entre corchetes justo despues de la "
    "afirmacion, por ejemplo: 'El plazo es de tres meses [1].' No escribas "
    "nunca URLs, nombres de documentos ni fechas como citas: solo el indice "
    "entre corchetes. Si la pregunta trata sobre la redaccion literal de un "
    "articulo, cita el texto entre comillas. Si el contexto no contiene la "
    "respuesta, responde exactamente: 'No dispongo de informacion suficiente "
    "en el temario para responder a esa pregunta.' y no cites nada. Responde "
    "siempre en español."
)

REFUSAL_ANSWER = (
    "No dispongo de información suficiente en el temario para responder a esa pregunta."
)


@dataclass(frozen=True)
class BuiltContext:
    """Final context handed to the answer generator."""

    chunks: list[RetrievedChunk] = field(default_factory=list)  # survivors, final order
    blocks: str = ""  # numbered text handed to the LLM
    sources: list[SourceRef] = field(default_factory=list)  # index -> citation metadata
    n_tokens_estimate: int = 0


def select_context(
    chunks: Sequence[RetrievedChunk], config: RetrievalConfig
) -> BuiltContext:
    """Select the final chunks and render them as numbered context blocks.

    Applies three limits, whichever binds first:

    1. Relative score cutoff: keep chunks with
       `rerank_prob >= keep_ratio * top_prob`. Legal answers usually live in a
       single artículo, so this often cuts to 2 chunks and saves generation
       time. Skipped when chunks have no `rerank_prob` (e.g. `NoOpReranker`).
    2. `max_chunks`: hard cap on the final count.
    3. `max_context_tokens`: stop adding once the running estimate would
       exceed it; never emits a partially-truncated chunk.

    Returns an empty `BuiltContext` when the input is empty or, with a
    reranker in use, the top chunk's probability is below
    `min_relevance_prob` - the caller treats that as the NO_CONTEXT gate,
    since rerank probabilities are calibrated whereas raw retrieval (RRF)
    scores are not.

    Args:
        chunks: Reranked (or retrieval-ordered) chunks, best first.
        config: Shared retrieval config.

    Returns:
        The selected chunks, rendered context blocks and per-index sources.
    """
    if not chunks:
        return BuiltContext()

    top_prob = chunks[0].rerank_prob
    if top_prob is not None and top_prob < config.min_relevance_prob:
        return BuiltContext()

    candidates = list(chunks)
    if top_prob is not None:
        threshold = config.keep_ratio * top_prob
        candidates = [c for c in candidates if c.rerank_prob is not None and c.rerank_prob >= threshold]

    selected: list[RetrievedChunk] = []
    running_tokens = 0
    for chunk in candidates[: config.max_chunks]:
        chunk_tokens = _estimate_tokens(chunk.content, config.chars_per_token)
        if selected and running_tokens + chunk_tokens > config.max_context_tokens:
            break
        selected.append(chunk)
        running_tokens += chunk_tokens

    blocks, sources = _render(selected)
    return BuiltContext(
        chunks=selected, blocks=blocks, sources=sources, n_tokens_estimate=running_tokens
    )


def _estimate_tokens(text: str, chars_per_token: float) -> int:
    """Approximate token count as ceil(len(text) / chars_per_token)."""
    return -(-len(text) // int(chars_per_token)) if chars_per_token >= 1 else len(text)


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
        """
        Args:
            llm: Shared Ollama client.
            config: Shared retrieval config, used for `max_history_turns`.
        """
        self._llm = llm
        self._config = config

    async def stream(
        self, query: str, context: BuiltContext, history: Sequence[dict[str, str]]
    ) -> AsyncIterator[str]:
        """Yield answer fragments as they arrive from Ollama.

        Args:
            query: User query.
            context: Selected context blocks from `select_context`. If it has
                no chunks, yields the fixed refusal sentence without calling
                the model.
            history: Prior conversation turns as Ollama-style message dicts
                (`{"role": ..., "content": ...}`), trimmed to the configured
                `max_history_turns` and passed as message history rather than
                stuffed into the prompt text.

        Yields:
            Successive fragments of the generated answer.
        """
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
