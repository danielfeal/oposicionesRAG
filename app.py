"""Chainlit UI over the RAG pipeline.

CLI usage (run from repo root):
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

EXAM_ITEMS: dict[str, str] = {
    "A1 — Cuerpo Superior de Administradores Civiles del Estado": "A1",
    "A2 — Gestión de la Administración Civil del Estado": "A2",
    "C1 — Cuerpo General Administrativo": "C1",
    "C2 — Cuerpo General Auxiliar": "C2",
}
DEFAULT_EXAM = "A1"

TEMA_ITEMS: dict[str, str] = {
    "Todos los temas": "TODOS",
    "I — Materias comunes": "I",
    "II — Materias jurídicas": "II",
    "III — Materias sociales": "III",
    "IV — Materias económicas": "IV",
    "V — Materias técnicas": "V",
}

WELCOME_MESSAGE = (
    "**Bienvenido! Soy un asistente basado en IA para estudiantes de oposiciones**\n\n"
    "Haz preguntas y obtendrás una respuesta junto a su fuente oficial en el BOE.\n\n"
    "Antes de empezar, abre el icono de ajustes ⚙️ junto al cuadro de mensaje "
    "y selecciona tu examen de oposición. En caso de ser el examen A1, podrás seleccionar "
    "también el tema. Las respuestas se basarán unicamente en el temario oficial para el "
    "examen y tema seleccionados.\n\n"
    "Lee más información pulsando en 'Readme'."
)
LOADING_MESSAGE = "⏳ Cargando los modelos, esto tarda unos segundos..."
READY_MESSAGE = "✅ Ya puedes hacer preguntas!"
BUILD_ERROR_MESSAGE = "Ha habido un error, no se han podido cargar los modelos."
FALLBACK_MESSAGE = "No he podido generar una respuesta. Inténtalo de nuevo."

_build_task: "asyncio.Task[Components] | None" = None


@cl.on_app_startup
async def startup() -> None:
    """Start building the pipeline components without blocking the server."""
    global _build_task
    _build_task = asyncio.create_task(asyncio.to_thread(build_components, AppConfig()))


@cl.on_app_shutdown
async def shutdown() -> None:
    """Close the trace store's SQLite handle, if one was ever opened.

    Cancels the build instead if it is still running, and stays silent when it
    failed - shutdown must not raise on top of an earlier error.
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
    """
    widgets: list[InputWidget] = [
        Select(id="exam", label="Oposición", items=EXAM_ITEMS, initial_value=exam)
    ]
    if exam == "A1":
        widgets.append(
            Select(id="tema", label="Tema", items=TEMA_ITEMS, initial_value="TODOS")
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
        actions=[cl.Action(name="start", payload={}, label="OK")],
    )
    await welcome.send()


@cl.action_callback("start")
async def on_start(action: cl.Action) -> None:
    """Report readiness when clicked: instant if already built, else wait."""
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
    different corpus and carrying old context across it would be wrong.
    
    The tema widget is added or removed by resending `ChatSettings`.
    """
    new_exam = settings.get("exam") or DEFAULT_EXAM
    old_exam = cl.user_session.get("exam", DEFAULT_EXAM)
    tema = settings.get("tema") or "TODOS"
    new_temas = [tema] if new_exam == "A1" and tema != "TODOS" else []

    cl.user_session.set("exam", new_exam)
    cl.user_session.set("temas", new_temas)

    if new_exam != old_exam:
        cl.user_session.set("history", [])
        await cl.ChatSettings(_settings_widgets(new_exam)).send()


async def _await_components() -> Components:
    """Return the shared components, showing a notice if still building.

    Fallback for a question that arrives before the Start button was clicked
    (or before its wait resolved).
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

    # Only real answers become history.
    if trace.answer:
        history.append({"role": "user", "content": message.content})
        history.append({"role": "assistant", "content": trace.answer})
        cl.user_session.set("history", history)


def _render_sources(sources: list[SourceRef]) -> str:
    """Render the `[n]` citations the model emits as a markdown source list.

    The model only ever sees bracketed indices, never document names or URLs,
    so resolving them is the UI's job. Headings use the same
    `doc_name — seccion` convention as the context blocks the model was shown.
    """
    lines = ["\n\n---\n**Fuentes**\n"]
    for source in sources:
        heading = " — ".join(p for p in (source.doc_name, source.seccion) if p)
        label = heading or source.doc_id
        linked = f"[{label}]({source.source_url})" if source.source_url else label
        lines.append(f"- **[{source.index}]** {linked}")
    return "\n".join(lines)
