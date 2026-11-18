"""Memoizing wrapper: identical texts are embedded once per process.

Schema cards are re-ranked on every question and the question itself is embedded
by both card retrieval and few-shot retrieval; without this each query cost
three embedding requests, which exhausts free-tier daily quotas quickly.
"""

from __future__ import annotations

from collections.abc import Sequence

from text_to_sql_engine.embeddings.base import Embedder


class CachedEmbedder:
    def __init__(self, inner: Embedder, max_entries: int = 4096) -> None:
        self._inner = inner
        self._max = max_entries
        self._cache: dict[str, list[float]] = {}

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        missing = list(dict.fromkeys(t for t in texts if t not in self._cache))
        if missing:
            if len(self._cache) + len(missing) > self._max:
                self._cache.clear()
            for text, vector in zip(missing, self._inner.embed(missing), strict=True):
                self._cache[text] = vector
        return [self._cache[t] for t in texts]
