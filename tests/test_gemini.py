"""Gemini provider, embedder, and provider-selection tests (no network)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from text_to_sql_engine.config import Settings
from text_to_sql_engine.container import build_components
from text_to_sql_engine.embeddings.fake import FakeEmbedder
from text_to_sql_engine.embeddings.gemini_embedder import MAX_BATCH, GeminiEmbedder
from text_to_sql_engine.execution.executor import ExecutionResult, FakeExecutor
from text_to_sql_engine.llm.gemini_provider import GeminiProvider
from text_to_sql_engine.models import QueryStatus


class _StubModels:
    """Records calls and mimics the shapes of google-genai responses."""

    def __init__(self, text: str | None = "SELECT 1") -> None:
        self.text = text
        self.generate_calls: list[dict[str, Any]] = []
        self.embed_calls: list[dict[str, Any]] = []

    def generate_content(self, **kwargs: Any) -> SimpleNamespace:
        self.generate_calls.append(kwargs)
        return SimpleNamespace(text=self.text)

    def embed_content(self, **kwargs: Any) -> SimpleNamespace:
        self.embed_calls.append(kwargs)
        embeddings = [SimpleNamespace(values=[float(len(t)), 1.0]) for t in kwargs["contents"]]
        return SimpleNamespace(embeddings=embeddings)


def _client(models: _StubModels) -> SimpleNamespace:
    return SimpleNamespace(models=models)


def test_provider_sends_prompt_system_instruction_and_zero_temperature() -> None:
    models = _StubModels("SELECT 42")
    provider = GeminiProvider("unused", model="m-1", client=_client(models))
    assert provider.complete("hello", system="be careful") == "SELECT 42"
    call = models.generate_calls[0]
    assert call["model"] == "m-1"
    assert call["contents"] == "hello"
    assert call["config"].system_instruction == "be careful"
    assert call["config"].temperature == 0.0


def test_provider_returns_empty_string_when_response_has_no_text() -> None:
    provider = GeminiProvider("unused", client=_client(_StubModels(text=None)))
    assert provider.complete("hello") == ""


def test_embedder_batches_and_preserves_order() -> None:
    models = _StubModels()
    texts = [f"t{'x' * i}" for i in range(MAX_BATCH + 5)]
    vectors = GeminiEmbedder("unused", model="e-1", client=_client(models)).embed(texts)
    assert [call["model"] for call in models.embed_calls] == ["e-1", "e-1"]
    assert [len(call["contents"]) for call in models.embed_calls] == [MAX_BATCH, 5]
    assert [v[0] for v in vectors] == [float(len(t)) for t in texts]


def test_embedder_rejects_mismatched_response_size() -> None:
    models = _StubModels()
    models.embed_content = lambda **_: SimpleNamespace(embeddings=[])  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="0 embeddings for 2 texts"):
        embedder = GeminiEmbedder("unused", model="gemini-embedding-001", client=_client(models))
        embedder.embed(["a", "b"])


def test_embedder_empty_input_makes_no_call() -> None:
    models = _StubModels()
    assert GeminiEmbedder("unused", client=_client(models)).embed([]) == []
    assert models.embed_calls == []


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, "fake"),
        ({"gemini_api_key": "g"}, "gemini"),
        ({"openai_api_key": "o"}, "openai"),
        ({"gemini_api_key": "g", "openai_api_key": "o"}, "gemini"),
        ({"gemini_api_key": "g", "openai_api_key": "o", "llm_provider": "openai"}, "openai"),
        ({"groq_api_key": "q"}, "groq"),
        ({"gemini_api_key": "g", "groq_api_key": "q", "llm_provider": "groq"}, "groq"),
        ({"groq_api_key": "q", "openai_api_key": "o"}, "groq"),
        ({"openai_api_key": "o", "llm_provider": "groq"}, "fake"),
        ({"openai_api_key": "o", "llm_provider": "gemini"}, "fake"),
        ({"gemini_api_key": "g", "llm_provider": "openai"}, "fake"),
    ],
)
def test_active_provider_selection(kwargs: dict[str, str], expected: str) -> None:
    assert Settings(**kwargs).active_provider == expected  # type: ignore[arg-type]


@pytest.fixture
def embedder_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Replace the network embedder in the container; record how it was built."""
    calls: list[tuple[str, str]] = []

    def fake_factory(api_key: str, model: str) -> FakeEmbedder:
        calls.append((api_key, model))
        return FakeEmbedder()

    monkeypatch.setattr("text_to_sql_engine.container.GeminiEmbedder", fake_factory)
    return calls


def test_gemini_needs_no_openai_key_and_reports_provider(
    tmp_path: Path, embedder_calls: list[tuple[str, str]]
) -> None:
    settings = Settings(
        database_url=None,
        gemini_api_key="test-key-not-real",
        llm_provider="gemini",
        history_db_path=str(tmp_path / "h.sqlite"),
    )
    models = _StubModels("SELECT status, COUNT(*) FROM orders GROUP BY status")
    llm = GeminiProvider("unused", client=_client(models))
    result = ExecutionResult(columns=["status", "count"], rows=[{"status": "paid", "count": 1}])
    executor = FakeExecutor(default_result=result)
    components = build_components(settings, llm=llm, executor=executor)
    assert embedder_calls == [("test-key-not-real", "gemini-embedding-2")]
    assert components.pipeline.answer("How many orders per status?").status is QueryStatus.OK
    assert components.llm_mode == "gemini"
    assert components.llm_model == "gemini-3.5-flash-lite"
    assert settings.openai_api_key is None


def test_gemini_sql_still_goes_through_guardrails(
    tmp_path: Path, embedder_calls: list[tuple[str, str]]
) -> None:
    settings = Settings(
        database_url=None,
        gemini_api_key="test-key-not-real",
        history_db_path=str(tmp_path / "h.sqlite"),
    )
    models = _StubModels("DELETE FROM orders")
    executor = FakeExecutor()
    components = build_components(
        settings, llm=GeminiProvider("unused", client=_client(models)), executor=executor
    )
    result = components.pipeline.answer("Show all orders")
    assert result.status is QueryStatus.REJECTED
    assert result.error is not None and "select-only" in result.error
    assert executor.executed == []


def test_openai_compat_mode_targets_gemini_openai_endpoint(tmp_path: Path) -> None:
    from text_to_sql_engine.llm.openai_provider import OpenAIProvider

    settings = Settings(
        database_url=None, llm_provider="gemini", gemini_api_key="k",
        gemini_api="openai_compat", llm_fallback="none",
        history_db_path=str(tmp_path / "h.sqlite"),
    )
    components = build_components(settings)
    llm = components.pipeline.llm
    assert isinstance(llm, OpenAIProvider)
    assert "generativelanguage.googleapis.com/v1beta/openai" in str(llm._client.base_url)
    assert components.llm_mode == "gemini"
