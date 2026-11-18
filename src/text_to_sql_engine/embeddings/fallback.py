"""Embedder failover. Vectors from different models are not comparable, so each
call is served entirely by one backend, and ``space`` tells consumers (the
few-shot store) when the vector space changed and stored vectors must be rebuilt."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence

from text_to_sql_engine.embeddings.base import Embedder

log = logging.getLogger(__name__)


class FallbackEmbedder:
    def __init__(
        self, primary: Embedder, fallback: Embedder, primary_name: str, fallback_name: str,
        cooldown_s: float = 300.0,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._names = (primary_name, fallback_name)
        self._cooldown = cooldown_s
        self._skip_until = 0.0
        self.space = primary_name
        self.fallback_calls = 0

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if time.monotonic() >= self._skip_until:
            try:
                vectors = self._primary.embed(texts)
                self.space = self._names[0]
                return vectors
            except Exception as error:
                log.warning("%s embeddings failed (%s); using %s", self._names[0],
                            type(error).__name__, self._names[1])
                self._skip_until = time.monotonic() + self._cooldown
        self.space = self._names[1]
        self.fallback_calls += 1
        return self._fallback.embed(texts)
