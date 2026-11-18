"""Groq provider wiring (stub client; no network)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from text_to_sql_engine.config import Settings
from text_to_sql_engine.container import build_components
from text_to_sql_engine.llm.groq_provider import GROQ_BASE_URL, GroqProvider


class _Completions:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] = {}

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.kwargs = kwargs
        message = SimpleNamespace(content="SELECT 1")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_groq_provider_sends_system_prompt_and_zero_temperature() -> None:
    completions = _Completions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    provider = GroqProvider("k", client=client)
    assert provider.complete("hi", system="sys") == "SELECT 1"
    assert completions.kwargs["model"] == "qwen/qwen3.8-27b"
    assert completions.kwargs["temperature"] == 0.0
    assert completions.kwargs["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]


def test_groq_default_client_targets_groq_endpoint() -> None:
    provider = GroqProvider("k")
    assert str(provider._client.base_url).startswith(GROQ_BASE_URL)


def test_container_reports_groq_mode(tmp_path: Path) -> None:
    settings = Settings(
        database_url=None, llm_provider="groq", groq_api_key="k",
        history_db_path=str(tmp_path / "h.sqlite"),
    )
    components = build_components(settings)
    assert components.llm_mode == "groq"
    assert components.llm_model == "qwen/qwen3.8-27b"
