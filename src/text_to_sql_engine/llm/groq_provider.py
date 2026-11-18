"""Groq adapter behind the LLMProvider protocol.

Groq serves an OpenAI-compatible chat-completions API, so this reuses the
OpenAI SDK pointed at Groq's endpoint. Groq has no embeddings endpoint; the
container pairs this provider with the local lexical embedder.
"""

from __future__ import annotations

from typing import Any

from openai import OpenAI

from text_to_sql_engine.llm.openai_provider import OpenAIProvider

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class GroqProvider(OpenAIProvider):
    def __init__(
        self,
        api_key: str,
        model: str = "qwen/qwen3.8-27b",
        temperature: float = 0.0,
        client: Any = None,
        reasoning_effort: str | None = None,
    ) -> None:
        super().__init__(
            api_key, model=model, temperature=temperature, reasoning_effort=reasoning_effort
        )
        self._client = client or OpenAI(api_key=api_key, base_url=GROQ_BASE_URL, max_retries=5)
