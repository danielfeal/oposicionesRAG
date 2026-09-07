"""Shared Gemini (Vertex AI) client used by every LLM-consuming pipeline stage."""

import os
from typing import AsyncIterator, TypeVar

from google import genai
from google.genai import types
from pydantic import BaseModel

from rag.config import LlmConfig

_T = TypeVar("_T", bound=BaseModel)


class GeminiClient:
    """Thin shared wrapper over one Gemini model. Injected, never per-component.

    `chat_structured` uses the sync surface (`.models`) because `QueryProcessor.process` must
    stay callable synchronously from `rag/evaluation/query_processing.py`. `stream_chat` uses the
    async surface (`.aio.models`), mirroring the old Ollama client's own sync/async split.
    """

    def __init__(self, config: LlmConfig) -> None:
        self._config = config
        self._client = genai.Client(
            vertexai=True,
            api_key=os.environ["GOOGLE_API_KEY"],
            project=os.environ["GOOGLE_CLOUD_PROJECT"],
            location=os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
            http_options=types.HttpOptions(timeout=config.timeout_s * 1000),  # ms
        )

    def chat_structured(
        self,
        messages: list[dict[str, str]],
        *,
        response_schema: type[_T],
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> _T:
        """One non-streaming structured call. Returns the parsed model, like the eval judge."""
        system_instruction, contents = self._to_contents(messages)
        response = self._client.models.generate_content(
            model=self._config.model,
            contents=contents,
            config=self._gen_config(
                system_instruction,
                temperature,
                max_output_tokens,
                response_mime_type="application/json",
                response_schema=response_schema,
            ),
        )
        return response.parsed

    async def stream_chat(
        self, messages: list[dict[str, str]], *, temperature: float | None = None
    ) -> AsyncIterator[str]:
        system_instruction, contents = self._to_contents(messages)
        stream = await self._client.aio.models.generate_content_stream(
            model=self._config.model,
            contents=contents,
            config=self._gen_config(system_instruction, temperature, None),
        )
        async for chunk in stream:
            if chunk.text:
                yield chunk.text

    def _gen_config(
        self,
        system_instruction: str | None,
        temperature: float | None,
        max_output_tokens: int | None,
        **extra,
    ) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=self._config.temperature if temperature is None else temperature,
            max_output_tokens=(
                self._config.max_output_tokens if max_output_tokens is None else max_output_tokens
            ),
            **extra,
        )

    @staticmethod
    def _to_contents(messages: list[dict[str, str]]) -> tuple[str | None, list[dict]]:
        """Split out the system message and remap "assistant" role to Gemini's "model" role.

        Returns (system_instruction, contents) as a plain tuple - no shared mutable state on
        `self`, since one `GeminiClient` instance is shared across concurrent requests.
        """
        system_instruction = None
        contents = []
        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
                continue
            role = "model" if msg["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})
        return system_instruction, contents
