"""OpenAI embeddings adapter behind the Embedder protocol."""

from __future__ import annotations

from collections.abc import Sequence

from openai import OpenAI


class OpenAIEmbedder:
    """Batches all texts into a single embeddings request."""

    def __init__(
        self, api_key: str, model: str = "text-embedding-3-small", base_url: str | None = None
    ) -> None:
        self._client = OpenAI(api_key=api_key, base_url=base_url, max_retries=5)
        self._model = model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(model=self._model, input=list(texts))
        return [item.embedding for item in response.data]
