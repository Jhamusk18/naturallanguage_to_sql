"""Offline-first demo: answers the bundled questions through the full pipeline.

If DATABASE_URL points at a reachable Postgres (e.g. `make db-up`), queries
run against it. Otherwise everything runs offline: fixture schema, FakeLLM
templates, and fixture rows that were computed from the same seed data.
The LLM side is always the FakeLLM here — the demo never needs an API key.
"""

from __future__ import annotations

from typing import Any

from text_to_sql_engine.config import Settings
from text_to_sql_engine.container import build_components
from text_to_sql_engine.demo_data import CORRECTION_SHOWCASE, DEMO_CASES
from text_to_sql_engine.execution.executor import ExecutionError, FakeExecutor, QueryExecutor
from text_to_sql_engine.history import HistoryStore
from text_to_sql_engine.llm.fake import FakeLLM
from text_to_sql_engine.models import QueryResult

_RULE = "-" * 72


def _database_reachable(dsn: str) -> bool:
    try:
        import psycopg

        with psycopg.connect(dsn, connect_timeout=2):
            return True
    except Exception:
        return False


def _format_value(value: Any) -> str:
    return str(value)


def _print_rows(result: QueryResult, max_rows: int = 5) -> None:
    if not result.columns:
        return
    widths = {
        column: max(
            len(column),
            *(len(_format_value(row.get(column))) for row in result.rows[:max_rows] or [{}]),
        )
        for column in result.columns
    }
    header = " | ".join(column.ljust(widths[column]) for column in result.columns)
    print(f"  {header}")
    print(f"  {'-' * len(header)}")
    for row in result.rows[:max_rows]:
        line = " | ".join(
            _format_value(row.get(column)).ljust(widths[column]) for column in result.columns
        )
        print(f"  {line}")
    if result.row_count > max_rows:
        print(f"  ... {result.row_count - max_rows} more row(s)")


def _print_result(result: QueryResult) -> None:
    print(f"Q: {result.question}")
    print(f"   status={result.status.value} attempts={result.attempts} "
          f"latency={result.latency_ms:.1f}ms")
    if result.sql:
        print(f"   SQL: {result.sql}")
    for round_ in result.corrections:
        print(f"   [{round_.stage}] attempt {round_.attempt}: {round_.error}")
    if result.summary:
        print(f"   summary: {result.summary}")
    _print_rows(result)
    print(_RULE)


def _correction_showcase(
    settings: Settings, history: HistoryStore | None, *, live: bool
) -> HistoryStore:
    """Run one scripted fail-then-succeed query so the loop is visible.

    The scripted first attempt passes the AST guardrails (every table and
    column exists) but violates GROUP BY semantics — the class of error only
    the database can see. Live, PostgreSQL's EXPLAIN raises it for real;
    offline, the FakeExecutor raises the same SQLSTATE-42803 error. Either
    way the loop feeds it back and the corrected second attempt succeeds.
    """
    case = CORRECTION_SHOWCASE
    llm = FakeLLM(script=[case.bad_sql, case.good_sql])
    executor: QueryExecutor | None = None
    if not live:
        executor = FakeExecutor(
            explain_script=[
                ExecutionError(case.explain_error_message, sqlstate=case.explain_error_sqlstate)
            ],
            default_result=case.result,
        )
    components = build_components(settings, llm=llm, executor=executor, history=history)
    print("Self-correction showcase — attempt 1 fails the EXPLAIN preflight:")
    _print_result(components.pipeline.answer(case.question))
    return components.history


def main() -> int:
    settings = Settings()
    live = bool(settings.database_url) and _database_reachable(settings.database_url or "")
    print(_RULE)
    if live:
        print("text-to-sql-engine demo — live PostgreSQL, offline FakeLLM templates")
    else:
        print("text-to-sql-engine demo — fully offline (fixture schema + fixture rows)")
    print(_RULE)

    if live:
        live_settings = settings.model_copy(
            update={"openai_api_key": None, "gemini_api_key": None}
        )
        components = build_components(live_settings)
        for case in DEMO_CASES:
            _print_result(components.pipeline.answer(case.question))
        history = _correction_showcase(live_settings, components.history, live=True)
    else:
        offline = settings.model_copy(
            update={"database_url": None, "openai_api_key": None, "gemini_api_key": None}
        )
        history = None
        for case in DEMO_CASES:
            executor = FakeExecutor(default_result=case.result)
            components = build_components(offline, executor=executor, history=history)
            history = components.history
            _print_result(components.pipeline.answer(case.question))
        history = _correction_showcase(offline, history, live=False)

    if history is not None:
        entries = history.recent(limit=len(DEMO_CASES) + 1)
        print(f"History (last {len(entries)} entries, newest first):")
        for entry in entries:
            print(f"  #{entry.id} [{entry.status.value}] {entry.question} "
                  f"({entry.latency_ms:.1f}ms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
