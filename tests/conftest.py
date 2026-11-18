"""Shared fixtures. Everything here is offline: fixture schema, fakes only."""

from __future__ import annotations

import os

import pytest

from text_to_sql_engine.embeddings.fake import FakeEmbedder
from text_to_sql_engine.guardrails.base import GuardrailContext, GuardrailPipeline
from text_to_sql_engine.guardrails.rules import default_rules
from text_to_sql_engine.models import DatabaseSchema
from text_to_sql_engine.prompts import PromptBuilder
from text_to_sql_engine.schema.fixtures import DEMO_SCHEMA


def pytest_configure(config: pytest.Config) -> None:
    """Runs before test modules are imported (``api`` builds an app at import time).

    Clears every variable Settings reads (keys, LLM_PROVIDER, GEMINI_API, LLM_FALLBACK,
    DATABASE_URL, ...), so a developer shell or cloud environment cannot change what the
    offline tests see.
    """
    from text_to_sql_engine.config import Settings

    for name in Settings.model_fields:
        os.environ.pop(name.upper(), None)


@pytest.fixture
def schema() -> DatabaseSchema:
    return DEMO_SCHEMA


@pytest.fixture
def ctx(schema: DatabaseSchema) -> GuardrailContext:
    return GuardrailContext(schema=schema, default_limit=200, dialect="postgres")


@pytest.fixture
def guardrails(ctx: GuardrailContext) -> GuardrailPipeline:
    return GuardrailPipeline(default_rules(), ctx)


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def prompts() -> PromptBuilder:
    return PromptBuilder(dialect="postgres")
