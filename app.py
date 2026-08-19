"""Chainlit UI over the query-side RAG pipeline.

A thin consumer of `rag.rag_pipeline.main`: exam and tema are both chosen
through the chat settings panel (gear icon next to the composer), and
`answer()` is streamed into a chat message. No pipeline logic lives here.

The pipeline components are built in the background at startup rather than
awaited, so the welcome message is reachable immediately and the ~30s of
model loading overlaps with the user reading it.

Run from the repository root, so the relative `AppConfig.trace_db` path
resolves against it:

    chainlit run app.py -w
"""

import asyncio
import logging
from typing import Any

import chainlit as cl
from chainlit.input_widget import InputWidget, Select

from rag.config import AppConfig
from rag.rag_pipeline.main import Components, answer, build_components, new_trace
from rag.rag_pipeline.types import PipelineStatus, SourceRef

logger = logging.getLogger(__name__)

# Label -> value, both shown in the sidebar Select.
EXAM_ITEMS: dict[str, str] = {
    "A1 — Cuerpo Superior de Administradores Civiles del Estado": "A1",
    "A2 — Gestión de la Administración Civil del Estado": "A2",
    "C1 — Cuerpo General Administrativo": "C1",
    "C2 — Cuerpo General Auxiliar": "C2",
}
DEFAULT_EXAM = "A1"

# Only A1 documents carry tema metadata, so the filter is offered only there.
# `HybridRetriever` would silently drop it for the other exams anyway.
TEMA_EXAM = "A1"

# Label -> value. Mirrors documentation/guia_temario_oposiciones.md; there is
# no tema enum in the codebase (metadata.csv stores them as a pipe-separated
# string), so the valid set is hardcoded here. Single-select: at most one
# tema narrows the search, or the sentinel searches the whole temario.
# Radix's Select.Item rejects an empty-string value (it's reserved internally
# for "no selection"), so the sentinel must be non-empty.
TEMA_ALL = "TODOS"  # sentinel: no tema filter
TEMA_ITEMS: dict[str, str] = {
    "Todos los temas": TEMA_ALL,
    "I — Materias comunes": "I",
    "II — Materias jurídicas": "II",
    "III — Materias sociales": "III",
    "IV — Materias económicas": "IV",
    "V — Materias técnicas": "V",
}

WELCOME_MESSAGE = (
    "**Bienvenido, soy un asistente basado en IA para estudiantes de oposiciones**\n\n"
    "Haz preguntas y obtendrás una respuesta junto a su fuente oficial en el BOE.\n\n"
    "Antes de empezar, abre el icono de ajustes ⚙️ junto al cuadro de mensaje "
    "y selecciona tu examen de oposición. En caso de ser el examen A1, podrás seleccionar "
    "también el tema. Las respuestas se basarán unicamente en el temario oficial para el "
    "examen y tema seleccionados."
)
START_LABEL = "OK"
LOADING_MESSAGE = "⏳ Cargando los modelos, esto tarda unos segundos..."
READY_MESSAGE = "Ya puedes hacer preguntas!"
BUILD_ERROR_MESSAGE = "Ha habido un error, no se han podido cargar los modelos."
# Shown only if the pipeline both streams nothing and sets no message. No
# current code path does that; this exists so the UI can never render an
# empty bubble.
FALLBACK_MESSAGE = "No he podido generar una respuesta. Inténtalo de nuevo."

_build_task: "asyncio.Task[Components] | None" = None


@cl.on_app_startup
async def startup() -> None:
    """Start building the pipeline components without blocking the server.

    Deliberately not awaited: awaiting here would hold the port closed for the
    whole model load. The task is created once per process, so the global
    `torch.set_num_threads()` call inside `CrossEncoderReranker` still happens
    exactly once, and no lock is needed - awaiting the same task from several
    sessions returns the one cached result.
    """
    global _build_task
    _build_task = asyncio.create_task(asyncio.to_thread(build_components, AppConfig()))


@cl.on_app_shutdown
async def shutdown() -> None:
    """Close the trace store's SQLite handle, if one was ever opened.

    Cancels the build instead if it is still running, and stays silent when it
    failed - shutdown must not raise on top of an earlier error. The `hasattr`
    guard is needed because `NullTraceStore` has no `close`.
    """
    if _build_task is None:
        return
    if not _build_task.done():
        _build_task.cancel()
        return
    if _build_task.cancelled() or _build_task.exception() is not None:
        return
    trace_store = _build_task.result().trace_store
    if hasattr(trace_store, "close"):
        await asyncio.to_thread(trace_store.close)


def _settings_widgets(exam: str) -> list[InputWidget]:
    """Build the settings widget list for the given exam.

    The tema `Select` is included only for A1, so switching to any other exam
    removes it instead of leaving a filter that would be silently ignored by
    the retriever.

    Args:
        exam: The exam currently selected.

    Returns:
        Widgets to pass to `cl.ChatSettings`.
    """
    widgets: list[InputWidget] = [
        Select(id="exam", label="Oposición", items=EXAM_ITEMS, initial_value=exam)
    ]
    if exam == TEMA_EXAM:
        widgets.append(
            Select(id="tema", label="Tema", items=TEMA_ITEMS, initial_value=TEMA_ALL)
        )
    return widgets


@cl.on_chat_start
async def start() -> None:
    """Send the settings panel, the welcome message and a Start button."""
    cl.user_session.set("exam", DEFAULT_EXAM)
    cl.user_session.set("temas", [])
    cl.user_session.set("history", [])

    await cl.ChatSettings(_settings_widgets(DEFAULT_EXAM)).send()

    welcome = cl.Message(
        content=WELCOME_MESSAGE,
        actions=[cl.Action(name="start", payload={}, label=START_LABEL)],
    )
    await welcome.send()


@cl.action_callback("start")
async def on_start(action: cl.Action) -> None:
    """Report readiness when clicked: instant if already built, else wait.

    The button does not gate the chat input - Chainlit has no API to disable
    it - so `on_message` has its own fallback for a question typed before this
    is clicked. This only makes the already-ready/loading state visible.
    """
    del action  # payload is empty, nothing to read
    assert _build_task is not None  # set by the startup hook

    if _build_task.done():
        await cl.Message(content=READY_MESSAGE).send()
        return

    notice = cl.Message(content=LOADING_MESSAGE)
    await notice.send()
    try:
        await _build_task
    except Exception:
        logger.exception("Component build failed.")
        notice.content = BUILD_ERROR_MESSAGE
    else:
        notice.content = READY_MESSAGE
    await notice.update()


@cl.on_settings_update
async def settings_update(settings: dict[str, Any]) -> None:
    """Persist the exam/tema selection; re-render the panel on exam change.

    Switching exam clears `history`, since a different exam targets a
    different corpus and carrying old context across it would be wrong, not
    helpful. The tema widget is added or removed by resending
    `ChatSettings`, which only pushes a widget-definition event to the client
    - the event that re-enters this handler is the separate, user-only
    "settings submitted" event, so this cannot loop.
    """
    new_exam = settings.get("exam") or DEFAULT_EXAM
    old_exam = cl.user_session.get("exam", DEFAULT_EXAM)
    tema = settings.get("tema") or TEMA_ALL
    new_temas = [tema] if new_exam == TEMA_EXAM and tema != TEMA_ALL else []

    cl.user_session.set("exam", new_exam)
    cl.user_session.set("temas", new_temas)

    if new_exam != old_exam:
        cl.user_session.set("history", [])
        await cl.ChatSettings(_settings_widgets(new_exam)).send()


async def _await_components() -> Components:
    """Return the shared components, showing a notice if still building.

    Fallback for a question that arrives before the Start button was clicked
    (or before its wait resolved).

    Raises:
        Exception: Whatever `build_components` raised. The same exception is
            re-raised on every call, since the task caches it.
    """
    assert _build_task is not None  # set by the startup hook
    if _build_task.done():
        return _build_task.result()

    notice = cl.Message(content=LOADING_MESSAGE)
    await notice.send()
    try:
        return await _build_task
    finally:
        await notice.remove()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Stream one pipeline answer, then attach status text and sources.

    `answer()` never raises: it reports failures as a status on the trace. It
    also yields nothing at all for OFF_TOPIC, NO_CONTEXT and RETRIEVAL_ERROR,
    so whether any token arrived decides how the reply is assembled.

    Args:
        message: The incoming user message.
    """
    try:
        components = await _await_components()
    except Exception:
        logger.exception("Component build failed; cannot answer.")
        await cl.Message(content=BUILD_ERROR_MESSAGE).send()
        return

    history: list[dict[str, str]] = cl.user_session.get("history", [])
    trace = new_trace(
        components.config,
        cl.user_session.get("exam", DEFAULT_EXAM),
        cl.user_session.get("temas", []),
        message.content,
        session_id=cl.user_session.get("id"),
    )

    reply = cl.Message(content="")
    await reply.send()

    streamed = False
    async for token in answer(components, trace, history):
        streamed = True
        await reply.stream_token(token)

    if not streamed:
        reply.content = trace.message or FALLBACK_MESSAGE
    else:
        if trace.status is not PipelineStatus.OK and trace.message:
            reply.content += f"\n\n{trace.message}"
        if trace.sources:
            reply.content += _render_sources(trace.sources)
    await reply.update()

    # Only real answers become history - persisting a status message would
    # feed the Spanish error text back into the next generation as context.
    if trace.answer:
        history.append({"role": "user", "content": message.content})
        history.append({"role": "assistant", "content": trace.answer})
        cl.user_session.set("history", history)


def _render_sources(sources: list[SourceRef]) -> str:
    """Render the `[n]` citations the model emits as a markdown source list.

    The model only ever sees bracketed indices, never document names or URLs,
    so resolving them is the UI's job. Headings use the same
    `doc_name — seccion` convention as the context blocks the model was shown.

    Args:
        sources: Citation refs from the completed trace, in index order.

    Returns:
        A markdown block to append to the answer.
    """
    lines = ["\n\n---\n**Fuentes**\n"]
    for source in sources:
        heading = " — ".join(p for p in (source.doc_name, source.seccion) if p)
        label = heading or source.doc_id
        # Index stays outside the link: nesting it as [[1] label](url) relies
        # on balanced-bracket parsing that not every markdown renderer gets right.
        linked = f"[{label}]({source.source_url})" if source.source_url else label
        lines.append(f"- **[{source.index}]** {linked}")
    return "\n".join(lines)
