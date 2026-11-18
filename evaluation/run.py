"""Evaluation harness for the text-to-sql pipeline.

    python -m evaluation.run                 # real LLM + PostgreSQL (OPENAI_API_KEY, DATABASE_URL)
    python -m evaluation.run --oracle        # dataset/pipeline sanity check, NOT a model evaluation
    python -m evaluation.run --limit 10      # first N questions only

Dataset: evaluation/questions.jsonl (answerable questions carry a gold SQL;
other categories are unsafe requests, nonexistent columns/tables, and
ambiguous input). Nothing here is hardcoded to make numbers look good: with a
real model the metrics are whatever the model achieves.

Metric definitions (answerable questions unless noted):
  valid SQL rate         first attempt passed guardrails and the EXPLAIN preflight
  execution success      final status == ok
  execution accuracy     final rows match the gold SQL rows (order-insensitive,
                         column names ignored)
  correction success     of questions whose first attempt failed, share that ended ok
  unsafe rejection       unsafe requests whose final status != ok (nothing executed)
  data unchanged         row counts of every table identical before and after the run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import psycopg

from text_to_sql_engine.config import Settings
from text_to_sql_engine.container import build_components
from text_to_sql_engine.execution.executor import ExecutionError, PostgresExecutor
from text_to_sql_engine.models import QueryResult, QueryStatus

DATASET = Path(__file__).parent / "questions.jsonl"
ARTIFACTS = Path("artifacts")


class OracleLLM:
    """Returns the dataset's reference SQL for each question.

    Exists only to validate the dataset and the pipeline plumbing (gold SQL
    must pass guardrails and execute; unsafe SQL must be rejected). It says
    nothing about how well a language model translates questions.
    """

    def __init__(self, sql_by_question: dict[str, str]) -> None:
        self._sql = sql_by_question

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        if "row_count=" in prompt:
            return ""
        question = prompt.rsplit("### Question", 1)[-1].split("\n\nSQL:")[0].strip()
        if prompt.startswith("Your previous"):
            question = prompt.split("### Question", 1)[1].split("### Failed SQL")[0].strip()
        return self._sql.get(question, "SELECT * FROM oracle_has_no_sql_for_this_question")


@dataclass
class Row:
    id: int
    category: str
    question: str
    status: str
    attempts: int
    first_attempt_valid: bool
    corrected_ok: bool | None
    accurate: bool | None
    sql: str | None
    error: str | None
    note: str = ""
    corrections: list[str] = field(default_factory=list)


def _norm(value: Any) -> Any:
    if isinstance(value, Decimal | float):
        return round(float(value), 4)
    if isinstance(value, datetime | date):
        return str(value)
    return value


def _multiset(rows: list[list[Any]]) -> list[str]:
    return sorted(repr([_norm(v) for v in row]) for row in rows)


def _rows_as_lists(result: QueryResult) -> list[list[Any]]:
    return [[row[c] for c in result.columns] for row in result.rows]


def _table_counts(dsn: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    with psycopg.connect(dsn) as conn:
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY 1"
            )
        ]
        for table in tables:
            row = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()
            counts[table] = int(row[0]) if row else -1
    return counts


def _gold_rows(executor: PostgresExecutor, sql: str) -> list[list[Any]]:
    result = executor.execute(sql)
    return [[row[c] for c in result.columns] for row in result.rows]


def _first_attempt_valid(result: QueryResult) -> bool:
    return not any(
        c.attempt == 1 and c.stage in {"guardrail", "explain"} for c in result.corrections
    )


def _pct(num: int, den: int) -> str:
    return f"{num}/{den} = {100 * num / den:.1f}%" if den else "n/a (0 cases)"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--oracle", action="store_true", help="use reference SQL instead of an LLM")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    settings = Settings()
    if not settings.database_url:
        print("DATABASE_URL is required (the dataset is defined against the PostgreSQL seed).")
        return 2
    if not args.oracle and settings.offline_llm:
        print("OPENAI_API_KEY is not set. Refusing to score the FakeLLM as if it were a model.\n"
              "Set OPENAI_API_KEY, or run with --oracle for a dataset/pipeline sanity check.")
        return 2

    questions = [json.loads(line) for line in DATASET.read_text().splitlines() if line.strip()]
    if args.limit:
        questions = questions[: args.limit]

    settings = settings.model_copy(
        update={"history_db_path": str(Path(tempfile.mkdtemp()) / "eval-history.sqlite")}
    )
    llm = None
    if args.oracle:
        sql_by_q = {q["question"]: q.get("gold_sql") or q.get("oracle_sql") for q in questions}
        llm = OracleLLM({k: v for k, v in sql_by_q.items() if v})
        settings = settings.model_copy(update={"openai_api_key": None})
    components = build_components(settings, llm=llm)
    gold_executor = PostgresExecutor(settings.database_url, statement_timeout_ms=10000)

    base_url = os.environ.get("OPENAI_BASE_URL")
    endpoint = f" endpoint={base_url}" if base_url else ""
    mode = "ORACLE SANITY CHECK (reference SQL, not a model)" if args.oracle else (
        f"LIVE model={settings.openai_model}{endpoint}")
    print(f"Mode: {mode} | database: postgres | questions: {len(questions)}")
    before = _table_counts(settings.database_url)

    rows: list[Row] = []
    for q in questions:
        if args.oracle and q["category"] == "ambiguous":
            rows.append(Row(q["id"], q["category"], q["question"], "skipped", 0, False, None,
                            None, None, None, note="no reference SQL in oracle mode"))
            continue
        result = components.pipeline.answer(q["question"])
        ok = result.status is QueryStatus.OK
        accurate: bool | None = None
        if q["category"] == "answerable":
            accurate = False
            if ok:
                try:
                    expected = _gold_rows(gold_executor, q["gold_sql"])
                    accurate = _multiset(_rows_as_lists(result)) == _multiset(expected)
                except ExecutionError as error:
                    print(f"  !! gold SQL failed for #{q['id']}: {error.message}")
        first_valid = _first_attempt_valid(result)
        needed_correction = bool(result.corrections)
        rows.append(Row(
            q["id"], q["category"], q["question"], result.status.value, result.attempts,
            first_valid, (ok if needed_correction else None), accurate, result.sql, result.error,
            corrections=[f"{c.stage}: {c.error}" for c in result.corrections],
        ))
        mark = "ok " if ok else result.status.value[:3]
        print(f"  #{q['id']:>2} [{q['category'][:5]}] {mark} a={result.attempts} "
              f"{q['question'][:62]}")

    after = _table_counts(settings.database_url)

    ans = [r for r in rows if r.category == "answerable"]
    unsafe = [r for r in rows if r.category == "unsafe"]
    corrected = [r for r in rows if r.corrected_ok is not None and r.category == "answerable"]
    print("\n" + "=" * 70)
    print(f"RESULTS  —  {mode}")
    print("=" * 70)
    rejected = sum(r.status in ("rejected", "error") for r in unsafe)
    lines = [
        ("answerable questions", str(len(ans))),
        ("valid SQL rate (1st try)", _pct(sum(r.first_attempt_valid for r in ans), len(ans))),
        ("execution success rate", _pct(sum(r.status == "ok" for r in ans), len(ans))),
        ("execution accuracy vs gold", _pct(sum(bool(r.accurate) for r in ans), len(ans))),
        ("correction success rate",
         _pct(sum(bool(r.corrected_ok) for r in corrected), len(corrected))
         + "  (questions whose 1st attempt failed)"),
        ("unsafe rejection rate", _pct(rejected, len(unsafe))),
        ("data unchanged", "YES" if before == after else "NO  <-- DATA CHANGED"),
    ]
    for label, value in lines:
        print(f"{label:<27}: {value}")
    for category in ("nonexistent", "ambiguous"):
        counts = Counter(r.status for r in rows if r.category == category)
        print(f"{category:<27}: {dict(counts)}")
    if args.oracle:
        print("\nORACLE MODE validates the dataset and pipeline plumbing only. "
              "These numbers are NOT a measure of LLM quality.")

    ARTIFACTS.mkdir(exist_ok=True)
    out = ARTIFACTS / ("eval_oracle.json" if args.oracle else "eval_live.json")
    payload = {"mode": mode, "rows": [asdict(r) for r in rows]}
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nPer-question results: {out}")
    return 0 if before == after else 1


if __name__ == "__main__":
    sys.exit(main())
