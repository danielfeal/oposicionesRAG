"""Shared Ollama client used by every LLM-consuming pipeline stage."""

import logging
from typing import Any, AsyncIterator

import ollama

from rag.config import LlmConfig

logger = logging.getLogger(__name__)


class OllamaClient:
    """Thin shared wrapper over one Ollama model. Injected, never per-component.

    Holds one sync and one async client so the condenser, relevance checker
    and answer generator all talk to the same running model instance.
    """

    def __init__(self, config: LlmConfig) -> None:
        self._config = config
        self._client = ollama.Client(host=config.base_url, timeout=config.timeout_s)
        self._async_client = ollama.AsyncClient(host=config.base_url, timeout=config.timeout_s)

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        num_predict: int | None = None,
    ) -> str:
        """Blocking single-shot chat call."""
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
        """Async token stream."""
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
        load. Failures are logged, not raised.
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
