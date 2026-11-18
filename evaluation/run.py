"""Evaluation harness for the text-to-sql pipeline.

    python -m evaluation.run                 # real LLM + PostgreSQL (API key + DATABASE_URL)
    python -m evaluation.run --oracle        # dataset/pipeline sanity check, NOT a model evaluation
    python -m evaluation.run --limit 10      # first N questions only

Ablation flags (same dataset, same model, one variable changed per arm):
    --prompt-mode {schema_aware,columns_only,baseline}
                                            baseline = bare table names, no exemplars
                                            columns_only = table + column names, nothing else
    --sample-rows N                         sample rows per card (0 isolates their effect)
    --max-corrections N                     0 disables the self-correction loop
    --out PATH                              where to write the per-question JSON

Dataset: evaluation/questions.jsonl (answerable questions carry a gold SQL;
other categories are unsafe requests, nonexistent columns/tables, and
ambiguous input). Nothing here is hardcoded to make numbers look good: with a
real model the metrics are whatever the model achieves.

Metric definitions (answerable questions unless noted):
  valid SQL rate         first attempt passed guardrails and the EXPLAIN preflight
  execution success      ("query success") final status == ok: the SQL passed every
                         guardrail and executed on PostgreSQL without error. It says
                         nothing about whether the rows are right; see accuracy.
  ... (1st try)          same, using only the first attempt. In a run with corrections
                         on, the first attempt is exactly what a corrections-off run
                         produces (same prompt, same model, temperature 0), so one run
                         gives a paired correction OFF vs ON comparison per question.
  execution accuracy     final rows match the gold SQL rows (order-insensitive,
                         column names ignored)
  correction success     of questions whose first attempt failed, share that ended ok
  unsafe rejection       unsafe requests whose final status != ok (nothing executed:
                         rejected, error, or declined as unanswerable)
  hallucination rate     first-attempt SQL referenced a table/column absent from the
                         schema (static sqlglot check, evaluation/hallucination.py)
  declined               status == unanswerable (model replied CANNOT_ANSWER); for
                         answerable questions this is a false refusal, for nonexistent
                         and ambiguous ones it is the desired outcome
  data unchanged         row counts of every table identical before and after the run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import psycopg

from evaluation.hallucination import find_hallucinations
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
    first_attempt_sql: str | None = None
    hallucinations: list[str] = field(default_factory=list)
    note: str = ""
    corrections: list[str] = field(default_factory=list)
    first_attempt_ok: bool = False
    first_attempt_accurate: bool | None = None


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


def _first_attempt_sql(result: QueryResult) -> str | None:
    """The SQL the model produced on attempt 1 (the final SQL if it needed no correction)."""
    first = [c for c in result.corrections if c.attempt == 1]
    return first[0].sql if first else result.sql


def _pct(num: int, den: int) -> str:
    return f"{num}/{den} = {100 * num / den:.1f}%" if den else "n/a (0 cases)"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--oracle", action="store_true", help="use reference SQL instead of an LLM")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--prompt-mode", choices=["schema_aware", "columns_only", "baseline"], default=None
    )
    parser.add_argument("--sample-rows", type=int, default=None)
    parser.add_argument("--generation-mode", choices=["text", "tools"], default=None)
    parser.add_argument("--max-corrections", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--resume", action="store_true",
        help="continue an incomplete --out file (same provider/model/config), skipping "
             "questions already scored; for runs cut short by a free-tier quota",
    )
    args = parser.parse_args(argv)

    settings = Settings()
    if not settings.database_url:
        print("DATABASE_URL is required (the dataset is defined against the PostgreSQL seed).")
        return 2
    if not args.oracle and settings.offline_llm:
        print("No LLM API key is set. Refusing to score the FakeLLM as if it were a model.\n"
              "Set GEMINI_API_KEY, GROQ_API_KEY or OPENAI_API_KEY, or run with --oracle for a "
              "dataset/pipeline sanity check.")
        return 2

    questions = [json.loads(line) for line in DATASET.read_text().splitlines() if line.strip()]
    if args.limit:
        questions = questions[: args.limit]

    overrides: dict[str, Any] = {
        "history_db_path": str(Path(tempfile.mkdtemp()) / "eval-history.sqlite"),
        "summarize_results": False,
    }
    if args.prompt_mode:
        overrides["prompt_mode"] = args.prompt_mode
    if args.max_corrections is not None:
        overrides["max_correction_rounds"] = args.max_corrections
    if args.generation_mode:
        overrides["generation_mode"] = args.generation_mode
    if args.sample_rows is not None:
        overrides["sample_rows_per_table"] = args.sample_rows
    settings = settings.model_copy(update=overrides)
    llm = None
    if args.oracle:
        sql_by_q = {q["question"]: q.get("gold_sql") or q.get("oracle_sql") for q in questions}
        llm = OracleLLM({k: v for k, v in sql_by_q.items() if v})
        settings = settings.model_copy(
            update={"openai_api_key": None, "gemini_api_key": None, "groq_api_key": None}
        )
    components = build_components(settings, llm=llm)
    gold_executor = PostgresExecutor(settings.database_url, statement_timeout_ms=10000)

    base_url = os.environ.get("OPENAI_BASE_URL")
    endpoint = f" endpoint={base_url}" if base_url else ""
    mode = "ORACLE SANITY CHECK (reference SQL, not a model)" if args.oracle else (
        f"LIVE provider={settings.active_provider} model={settings.llm_model_name}{endpoint}")
    print(f"Mode: {mode} | database: postgres | questions: {len(questions)}")
    print(f"Config: prompt_mode={settings.prompt_mode} "
          f"max_corrections={settings.max_correction_rounds} "
          f"sample_rows={settings.sample_rows_per_table}")
    schema = components.schema_cache.get()
    before = _table_counts(settings.database_url)

    out = args.out or ARTIFACTS / ("eval_oracle.json" if args.oracle else "eval_live.json")
    rows: list[Row] = []
    prior_usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    if args.resume and out.exists():
        previous = json.loads(out.read_text())
        pc = previous["config"]
        same = {
            "provider": settings.active_provider, "model": settings.llm_model_name,
            "prompt_mode": settings.prompt_mode,
            "max_corrections": settings.max_correction_rounds,
            "sample_rows": settings.sample_rows_per_table,
            "generation_mode": settings.generation_mode,
        }
        mismatched = {k: (pc.get(k), v) for k, v in same.items() if pc.get(k) != v}
        if mismatched:
            print(f"Cannot resume {out}: config differs {mismatched}")
            return 2
        rows = [Row(**r) for r in previous["rows"]]
        prior_usage = previous.get("usage", prior_usage)
        print(f"Resuming {out}: {len(rows)} questions already scored")
    done = {r.id for r in rows}
    incomplete = False
    for q in questions:
        if q["id"] in done:
            continue
        if args.oracle and q["category"] == "ambiguous":
            rows.append(Row(q["id"], q["category"], q["question"], "skipped", 0, False, None,
                            None, None, None, note="no reference SQL in oracle mode"))
            continue
        result = None
        for wait in (0, 20, 40, 60, 60, 60):
            time.sleep(wait)
            try:
                result = components.pipeline.answer(q["question"])
                break
            except Exception as error:  # provider quota/outage: keep what was measured so far
                text = str(error)
                if "per minute" in text and wait < 60:
                    print(f"  .. per-minute limit at #{q['id']}, waiting")
                    continue
                print(f"  !! aborting at #{q['id']}: {type(error).__name__}: {text[:160]}")
                break
        if result is None:
            incomplete = True
            break
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
        first_sql = _first_attempt_sql(result)
        found = find_hallucinations(first_sql, schema)
        needed_correction = bool(result.corrections)
        first_ok = ok and not needed_correction
        rows.append(Row(
            q["id"], q["category"], q["question"], result.status.value, result.attempts,
            first_valid, (ok if needed_correction else None), accurate, result.sql, result.error,
            corrections=[f"{c.stage}: {c.error}" for c in result.corrections],
            first_attempt_sql=first_sql,
            hallucinations=[f"table:{t}" for t in found.tables]
            + [f"column:{c}" for c in found.columns],
            first_attempt_ok=first_ok,
            first_attempt_accurate=(
                bool(accurate) and first_ok if q["category"] == "answerable" else None
            ),
        ))
        mark = "ok " if ok else result.status.value[:3]
        print(f"  #{q['id']:>2} [{q['category'][:5]}] {mark} a={result.attempts} "
              f"{q['question'][:62]}")

    after = _table_counts(settings.database_url)
    fallback_llm = components.pipeline.llm
    llm_fallback_calls = getattr(fallback_llm, "fallback_calls", 0)
    if llm_fallback_calls:
        print(f"\n!! {llm_fallback_calls} LLM calls were served by the FALLBACK model "
              f"({fallback_llm.fallback_name}); these are mixed-provider results.")

    rows.sort(key=lambda r: r.id)
    ans = [r for r in rows if r.category == "answerable"]
    unsafe = [r for r in rows if r.category == "unsafe"]
    nonexistent = [r for r in rows if r.category == "nonexistent"]
    ambiguous = [r for r in rows if r.category == "ambiguous"]
    corrected = [r for r in rows if r.corrected_ok is not None and r.category == "answerable"]
    print("\n" + "=" * 70)
    print(f"RESULTS  —  {mode}")
    print("=" * 70)
    rejected = sum(r.status != "ok" for r in unsafe)  # nothing was executed
    lines = [
        ("answerable questions", str(len(ans))),
        ("valid SQL rate (1st try)", _pct(sum(r.first_attempt_valid for r in ans), len(ans))),
        ("exec success (1st try)", _pct(sum(r.first_attempt_ok for r in ans), len(ans))
         + "  = correction OFF"),
        ("execution success rate", _pct(sum(r.status == "ok" for r in ans), len(ans))
         + "  = final"),
        ("hallucination rate (1st try)",
         _pct(sum(bool(r.hallucinations) for r in ans), len(ans))),
        ("accuracy vs gold (1st try)",
         _pct(sum(bool(r.first_attempt_accurate) for r in ans), len(ans))),
        ("execution accuracy vs gold", _pct(sum(bool(r.accurate) for r in ans), len(ans))),
        ("answerable declined", _pct(sum(r.status == "unanswerable" for r in ans), len(ans))
         + "  (false refusals)"),
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

    out.parent.mkdir(parents=True, exist_ok=True)
    n = len(ans)
    metrics = {
        "answerable": n,
        "valid_sql_first_try": sum(r.first_attempt_valid for r in ans),
        "execution_success": sum(r.status == "ok" for r in ans),
        "execution_success_first_try": sum(r.first_attempt_ok for r in ans),
        "execution_accuracy": sum(bool(r.accurate) for r in ans),
        "execution_accuracy_first_try": sum(bool(r.first_attempt_accurate) for r in ans),
        "answerable_declined": sum(r.status == "unanswerable" for r in ans),
        "hallucinated_first_try": sum(bool(r.hallucinations) for r in ans),
        "needed_correction": len(corrected),
        "corrected_ok": sum(bool(r.corrected_ok) for r in corrected),
        "unsafe_rejected": rejected,
        "unsafe_total": len(unsafe),
        "nonexistent_not_executed": sum(r.status != "ok" for r in nonexistent),
        "nonexistent_total": len(nonexistent),
        "ambiguous_not_executed": sum(r.status != "ok" for r in ambiguous),
        "ambiguous_total": len(ambiguous),
    }
    provider_llm = getattr(fallback_llm, "primary", fallback_llm)
    usage = {
        key: prior_usage.get(key, 0) + int(getattr(provider_llm, key, 0) or 0)
        for key in ("calls", "prompt_tokens", "completion_tokens")
    }
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip() or None
    except OSError:
        commit = None
    config = {
        "provider": settings.active_provider,
        "model": settings.llm_model_name,
        "prompt_mode": settings.prompt_mode,
        "max_corrections": settings.max_correction_rounds,
        "sample_rows": settings.sample_rows_per_table,
        "generation_mode": settings.generation_mode,
        "questions": len(questions),
        "incomplete": incomplete,
        "llm_fallback_calls": llm_fallback_calls,
        "reasoning_effort": (
            settings.groq_reasoning_effort if settings.active_provider == "groq" else None
        ),
        "retrieval_embedder": {"gemini": "gemini", "openai": "openai"}.get(
            settings.active_provider, "lexical"
        ),
        "git_commit": commit,
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    payload = {
        "mode": mode,
        "config": config,
        "metrics": metrics,
        "usage": usage,
        "rows": [asdict(r) for r in rows],
    }
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nPer-question results: {out}")
    return 0 if before == after else 1


if __name__ == "__main__":
    sys.exit(main())
