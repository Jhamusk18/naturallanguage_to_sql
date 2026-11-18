"""Unsafe-request handling: intent gate, constant-query and all-columns rules."""

from __future__ import annotations

from pathlib import Path

import pytest

from text_to_sql_engine.config import Settings
from text_to_sql_engine.container import build_components
from text_to_sql_engine.guardrails.base import GuardrailPipeline, GuardrailViolation
from text_to_sql_engine.guardrails.intent import detect_write_intent
from text_to_sql_engine.llm.fake import FakeLLM
from text_to_sql_engine.models import QueryStatus


@pytest.mark.parametrize(
    "question",
    [
        "Delete all cancelled orders from the database.",
        "Update every product price to zero.",
        "Drop the customers table.",
        "Insert a new customer named Eve with email eve@example.com.",
        "Show all orders and then remove all refunds.",
        "Please truncate the refunds table",
    ],
)
def test_write_requests_are_detected(question: str) -> None:
    assert detect_write_intent(question) is not None


@pytest.mark.parametrize(
    "question",
    [
        "What is the most recent order update time?",
        "How many orders were deleted?",
        "Which products were removed from the catalog last week?",
        "How many customers signed up after 2025-10-31?",
    ],
)
def test_read_questions_are_not_flagged(question: str) -> None:
    assert detect_write_intent(question) is None


def test_write_request_is_rejected_without_calling_the_llm(tmp_path: Path) -> None:
    llm = FakeLLM(script=["SELECT 1"])
    settings = Settings(database_url=None, history_db_path=str(tmp_path / "h.sqlite"))
    result = build_components(settings, llm=llm).pipeline.answer("Drop the customers table.")
    assert result.status is QueryStatus.REJECTED and result.sql is None
    assert llm.calls == []


def test_constant_query_is_rejected(guardrails: GuardrailPipeline) -> None:
    with pytest.raises(GuardrailViolation, match="reads no table"):
        guardrails.validate("SELECT 1 WHERE 1 = 0")


def test_listing_every_sensitive_column_is_rejected(guardrails: GuardrailPipeline) -> None:
    sql = "SELECT id, email, full_name, country, created_at FROM customers"
    with pytest.raises(GuardrailViolation, match="every column"):
        guardrails.validate(sql)


def test_a_subset_of_sensitive_columns_is_allowed(guardrails: GuardrailPipeline) -> None:
    sql = "SELECT country, COUNT(*) FROM customers GROUP BY country"
    assert "customers" in guardrails.validate(sql)
