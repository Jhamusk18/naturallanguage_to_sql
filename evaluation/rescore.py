"""Add a column-tolerant accuracy to existing result files (no LLM calls).

    python -m evaluation.rescore evaluation/results/v2-groq-gpt-oss-20b/*.json

Strict accuracy (``execution_accuracy``) requires the predicted rows to equal
the gold rows exactly, so a correct answer with an extra column (e.g. email
next to the name) counts as wrong. The column-tolerant variant re-executes the
stored final SQL and gold SQL and counts a match when the row counts are equal
and every gold column's values (as a multiset) equal some predicted column's
values. It is reported next to the strict number, never instead of it.
Formatting differences (a timestamp vs a 'YYYY-MM' string) still count as wrong.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from evaluation.run import _norm
from text_to_sql_engine.execution.executor import ExecutionError, PostgresExecutor


def column_tolerant_match(pred: list[list[Any]], gold: list[list[Any]]) -> bool:
    if len(pred) != len(gold):
        return False
    if not gold:
        return True
    pred_cols = [sorted(repr(_norm(r[i])) for r in pred) for i in range(len(pred[0]))]
    used: set[int] = set()
    for j in range(len(gold[0])):
        gold_col = sorted(repr(_norm(r[j])) for r in gold)
        hit = next((i for i, c in enumerate(pred_cols) if i not in used and c == gold_col), None)
        if hit is None:
            return False
        used.add(hit)
    return True


def _rows(executor: PostgresExecutor, sql: str) -> list[list[Any]]:
    result = executor.execute(sql)
    return [[row[c] for c in result.columns] for row in result.rows]


def rescore(path: Path, executor: PostgresExecutor, gold: dict[int, str]) -> tuple[int, int, int]:
    data = json.loads(path.read_text())
    final = first = 0
    for row in data["rows"]:
        if row["category"] != "answerable":
            continue
        ok = False
        if row["status"] == "ok" and row["sql"]:
            try:
                predicted = _rows(executor, row["sql"])
                ok = column_tolerant_match(predicted, _rows(executor, gold[row["id"]]))
            except ExecutionError:
                ok = False
        row["accurate_column_tolerant"] = ok
        final += ok
        first += ok and row.get("first_attempt_ok", False)
    data["metrics"]["execution_accuracy_column_tolerant"] = final
    if all("first_attempt_ok" in r for r in data["rows"]):  # v1 files predate this field
        data["metrics"]["execution_accuracy_column_tolerant_first_try"] = first
    path.write_text(json.dumps(data, indent=2, default=str))
    return final, first, data["metrics"]["answerable"]


def main(argv: list[str] | None = None) -> int:
    paths = [Path(a) for a in (argv if argv is not None else sys.argv[1:])]
    dsn = os.environ.get("DATABASE_URL")
    if not dsn or not paths:
        print(__doc__)
        return 2
    questions = Path(__file__).parent / "questions.jsonl"
    gold = {
        q["id"]: q["gold_sql"]
        for q in map(json.loads, questions.read_text().splitlines())
        if q.get("gold_sql")
    }
    executor = PostgresExecutor(dsn, statement_timeout_ms=10000)
    for path in paths:
        final, first, n = rescore(path, executor, gold)
        print(f"{path.name:<28} column-tolerant accuracy {first}/{n} first try (if recorded), "
              f"{final}/{n} final")
    return 0


if __name__ == "__main__":
    sys.exit(main())
