"""Provider failover behaviour (pure fakes)."""

from __future__ import annotations

from pathlib import Path

import pytest

from text_to_sql_engine.config import Settings
from text_to_sql_engine.container import build_components
from text_to_sql_engine.embeddings.fake import FakeEmbedder
from text_to_sql_engine.embeddings.fallback import FallbackEmbedder
from text_to_sql_engine.embeddings.gemini_embedder import GeminiEmbedder
from text_to_sql_engine.fewshot.exemplars import DEMO_EXEMPLARS
from text_to_sql_engine.fewshot.store import FewShotStore
from text_to_sql_engine.llm.fake import FakeLLM
from text_to_sql_engine.llm.fallback import FallbackLLM


class _Broken:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        self.calls += 1
        raise RuntimeError("429 RESOURCE_EXHAUSTED")

    def embed(self, texts):  # type: ignore[no-untyped-def]
        self.calls += 1
        raise RuntimeError("429 RESOURCE_EXHAUSTED")


def test_llm_falls_back_and_skips_primary_during_cooldown() -> None:
    primary, backup = _Broken(), FakeLLM(script=["SELECT 1", "SELECT 2"])
    llm = FallbackLLM(primary, backup, "gemini", "groq")
    assert llm.complete("a") == "SELECT 1"
    assert llm.complete("b") == "SELECT 2"
    assert primary.calls == 1 and llm.fallback_calls == 2 and llm.primary_calls == 0


def test_llm_uses_primary_when_healthy() -> None:
    llm = FallbackLLM(FakeLLM(script=["P"]), FakeLLM(script=["F"]), "g", "q")
    assert llm.complete("a") == "P" and llm.fallback_calls == 0


def test_fewshot_rebuilds_vectors_after_embedder_switch() -> None:
    embedder = FallbackEmbedder(_Broken(), FakeEmbedder(), "gemini", "lexical")  # type: ignore[arg-type]
    store = FewShotStore(DEMO_EXEMPLARS, embedder)
    assert len(store.top_k("How many customers do we have?", k=3)) == 3


def test_container_wires_fallback_when_both_keys_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(GeminiEmbedder, "embed", _Broken().embed)
    settings = Settings(
        database_url=None, llm_provider="gemini", gemini_api_key="g", groq_api_key="q",
        history_db_path=str(tmp_path / "h.sqlite"),
    )
    assert isinstance(build_components(settings).pipeline.llm, FallbackLLM)
