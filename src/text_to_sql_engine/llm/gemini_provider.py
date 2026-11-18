"""Google Gemini adapter behind the LLMProvider protocol (official google-genai SDK)."""

from __future__ import annotations

from typing import Any

from google import genai
from google.genai import types

RETRY_ATTEMPTS = 5


def build_client(api_key: str) -> genai.Client:
    """Client that retries 429/5xx with exponential backoff (free-tier rate limits)."""
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(attempts=RETRY_ATTEMPTS)
        ),
    )


class GeminiProvider:
    """Thin adapter over ``models.generate_content``.

    Temperature defaults to 0 — SQL generation should be as deterministic as
    the provider allows. ``client`` is injectable for tests.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.5-flash-lite",
        temperature: float = 0.0,
        client: Any = None,
    ) -> None:
        self._client: Any = client or build_client(api_key)
        self._model = model
        self._temperature = temperature

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        config = types.GenerateContentConfig(
            system_instruction=system, temperature=self._temperature
        )
        response = self._client.models.generate_content(
            model=self._model, contents=prompt, config=config
        )
        return str(response.text or "")
