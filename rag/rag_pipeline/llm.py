"""Shared Ollama client and the relevance checker built directly on top of it."""

import logging
from typing import Any, AsyncIterator

import ollama

from rag.config import LlmConfig
from rag.rag_pipeline.types import RelevanceVerdict

logger = logging.getLogger(__name__)

RELEVANCE_SYSTEM_PROMPT = (
    "Eres un clasificador. El asistente al que sirves responde preguntas sobre "
    "la normativa de oposiciones a la Administración Pública española (textos "
    "consolidados del BOE, temarios de los cuerpos A1, A2, C1 y C2). Tu única "
    "tarea es decidir si la pregunta del usuario pertenece a ese dominio. "
    "Responde con una sola palabra: SI o NO. No expliques tu respuesta."
)


class OllamaClient:
    """Thin shared wrapper over one Ollama model. Injected, never per-component.

    Holds one sync and one async client so the condenser, relevance checker
    and answer generator all talk to the same running model instance.
    """

    def __init__(self, config: LlmConfig) -> None:
        """
        Args:
            config: Shared LLM config (base URL, model, thinking mode, timeout).
        """
        self._config = config
        self._client = ollama.Client(host=config.base_url, timeout=config.timeout_s)
        self._async_client = ollama.AsyncClient(host=config.base_url, timeout=config.timeout_s)

    @property
    def model(self) -> str:
        """The configured model tag."""
        return self._config.model

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        num_predict: int | None = None,
    ) -> str:
        """Blocking single-shot chat call.

        Args:
            messages: Ollama-style message list.
            temperature: Overrides `config.temperature` when given.
            num_predict: Overrides `config.num_predict` when given.

        Returns:
            The assistant message content.
        """
        response = self._client.chat(
            model=self._config.model,
            messages=messages,
            think=self._config.think,
            options=self._options(temperature, num_predict),
            keep_alive=self._config.keep_alive,
        )
        return response["message"]["content"]

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        num_predict: int | None = None,
    ) -> AsyncIterator[str]:
        """Async token stream.

        Args:
            messages: Ollama-style message list.
            temperature: Overrides `config.temperature` when given.
            num_predict: Overrides `config.num_predict` when given.

        Yields:
            Successive fragments of the assistant message content.
        """
        stream = await self._async_client.chat(
            model=self._config.model,
            messages=messages,
            think=self._config.think,
            options=self._options(temperature, num_predict),
            keep_alive=self._config.keep_alive,
            stream=True,
        )
        async for chunk in stream:
            content = chunk["message"]["content"]
            if content:
                yield content

    def warm_up(self) -> None:
        """Force the model into memory now, at the configured `keep_alive`.

        Called once at startup so the first real query doesn't pay for a cold
        load. Failures are logged, not raised - a slow first query is better
        than a startup crash if Ollama happens to be briefly unavailable.
        """
        try:
            self.chat(
                [{"role": "user", "content": "Hola"}],
                temperature=0.0,
                num_predict=1,
            )
        except Exception:
            logger.exception("Warm-up call failed for model '%s'.", self._config.model)

    def _options(self, temperature: float | None, num_predict: int | None) -> dict[str, Any]:
        """Build the Ollama `options` dict, falling back to config defaults."""
        return {
            "temperature": self._config.temperature if temperature is None else temperature,
            "num_predict": self._config.num_predict if num_predict is None else num_predict,
            "num_ctx": self._config.num_ctx,
        }


class RelevanceChecker:
    """Classifies a query as on- or off-topic for the oposiciones corpus.

    A retrieval-score threshold cannot do this: hybrid RRF scores are
    rank-based and uncalibrated, so an off-topic query still yields ~the same
    top score as an on-topic one.
    """

    def __init__(self, llm: OllamaClient) -> None:
        """
        Args:
            llm: Shared Ollama client.
        """
        self._llm = llm

    def check(self, query: str) -> RelevanceVerdict:
        """Classify `query` as RELEVANT, OFF_TOPIC, or UNKNOWN.

        Fails open: any error or unparseable reply is treated as UNKNOWN,
        which the pipeline continues on as if RELEVANT, so an Ollama hiccup
        degrades into a normal retrieval rather than a false rejection.

        Args:
            query: The (already condensed) user query.

        Returns:
            The verdict.
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
