"""Automatic failover between two LLM providers (e.g. Gemini quota -> Groq).

After a primary failure the primary is skipped for ``cooldown_s`` seconds so a
quota outage does not cost a full retry cycle on every call. Every fallback use
is counted, because results produced partly by the fallback model must not be
reported as primary-model results.
"""

from __future__ import annotations

import logging
import time

from text_to_sql_engine.llm.base import LLMProvider

log = logging.getLogger(__name__)


class FallbackLLM:
    def __init__(
        self,
        primary: LLMProvider,
        fallback: LLMProvider,
        primary_name: str,
        fallback_name: str,
        cooldown_s: float = 300.0,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self.primary_name = primary_name
        self.fallback_name = fallback_name
        self._cooldown = cooldown_s
        self._skip_until = 0.0
        self.primary_calls = 0
        self.fallback_calls = 0

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        if time.monotonic() >= self._skip_until:
            try:
                text = self._primary.complete(prompt, system=system)
                self.primary_calls += 1
                return text
            except Exception as error:
                log.warning("%s failed (%s); using %s", self.primary_name,
                            type(error).__name__, self.fallback_name)
                self._skip_until = time.monotonic() + self._cooldown
        self.fallback_calls += 1
        return self._fallback.complete(prompt, system=system)
