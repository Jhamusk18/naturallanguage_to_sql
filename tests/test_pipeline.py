"""End-to-end pipeline tests, fully offline via the container's fakes."""

from __future__ import annotations

from pathlib import Path

from text_to_sql_engine.config import Settings
from text_to_sql_engine.container import build_components
from text_to_sql_engine.execution.executor import ExecutionResult, FakeExecutor
from text_to_sql_engine.history import InMemoryHistoryStore
from text_to_sql_engine.models import QueryStatus


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        database_url=None,
        openai_api_key=None,
        default_row_limit=200,
        max_correction_rounds=2,
        history_db_path=str(tmp_path / "history.sqlite"),
    )


def test_offline_question_produces_validated_sql_and_summary(tmp_path: Path) -> None:
    executor = FakeExecutor(
        default_result=ExecutionResult(
            columns=["status", "order_count"], rows=[{"status": "completed", "order_count": 11}]
        )
    )
    history = InMemoryHistoryStore()
    components = build_components(_settings(tmp_path), executor=executor, history=history)
    result = components.pipeline.answer("How many orders are in each status?")

    assert result.status is QueryStatus.OK
    assert result.sql is not None
    assert "LIMIT 200" in result.sql  # template had no LIMIT; guardrails injected one
    assert "orders.status" in result.sql  # columns were fully qualified
    assert result.summary == "The query returned 1 row."
    assert result.row_count == 1
    assert set(result.retrieved_tables) == {
        "customers", "products", "orders", "order_items", "refunds"
    }
    assert result.corrections == []
    # EXPLAIN preflight ran before execution, on the same validated SQL.
    assert executor.explained == executor.executed


def test_sensitive_star_question_is_rejected_end_to_end(tmp_path: Path) -> None:
    history = InMemoryHistoryStore()
    components = build_components(
        _settings(tmp_path), executor=FakeExecutor(), history=history
    )
    result = components.pipeline.answer("Show me everything about our customers.")

    assert result.status is QueryStatus.REJECTED
    assert result.error is not None and "sensitive" in result.error
    assert result.rows == []
    assert result.summary is None
    # The FakeLLM keeps returning the same SELECT *, so the budget is spent.
    assert result.attempts == 3
    assert all(round_.stage == "guardrail" for round_ in result.corrections)


def test_fake_llm_does_not_fake_success_for_unsupported_questions(tmp_path: Path) -> None:
    executor = FakeExecutor()
    components = build_components(
        _settings(tmp_path), executor=executor, history=InMemoryHistoryStore()
    )
    result = components.pipeline.answer("What is the total revenue by month?")

    assert result.status is QueryStatus.REJECTED
    assert result.error is not None and "OPENAI_API_KEY" in result.error
    assert executor.executed == []


def test_history_records_both_outcomes(tmp_path: Path) -> None:
    history = InMemoryHistoryStore()
    components = build_components(
        _settings(tmp_path), executor=FakeExecutor(), history=history
    )
    components.pipeline.answer("How many orders are in each status?")
    components.pipeline.answer("Show me everything about our customers.")
    entries = history.recent()
    assert len(entries) == 2
    assert entries[0].status is QueryStatus.REJECTED
    assert entries[1].status is QueryStatus.OK


def test_schema_refresh_reloads_through_cache(tmp_path: Path) -> None:
    components = build_components(_settings(tmp_path), executor=FakeExecutor())
    before = components.schema_cache.get()
    components.schema_cache.invalidate()
    after = components.schema_cache.get()
    assert before.table_names() == after.table_names()
