"""Unit tests for the evaluation harness helpers (no database needed)."""

from __future__ import annotations

import json
from datetime import date
from decimal import Decimal

from evaluation.run import DATASET, OracleLLM, _first_attempt_valid, _multiset

from text_to_sql_engine.models import CorrectionRound, QueryResult, QueryStatus


def test_dataset_is_well_formed() -> None:
    rows = [json.loads(line) for line in DATASET.read_text().splitlines()]
    assert len(rows) >= 50
    assert len({row["id"] for row in rows}) == len(rows)
    assert len({row["question"] for row in rows}) == len(rows)
    for row in rows:
        assert row["category"] in {"answerable", "unsafe", "nonexistent", "ambiguous"}
        if row["category"] == "answerable":
            assert row["gold_sql"].lstrip().upper().startswith("SELECT")


def test_multiset_comparison_ignores_order_and_numeric_type() -> None:
    left = [[1, Decimal("2.50")], [date(2025, 1, 1), None]]
    right = [[date(2025, 1, 1), None], [1, 2.5]]
    assert _multiset(left) == _multiset(right)
    assert _multiset([[1]]) != _multiset([[2]])


def test_first_attempt_valid_only_counts_guardrail_and_explain_failures() -> None:
    def result(stage: str) -> QueryResult:
        return QueryResult(
            question="q", sql="SELECT 1", status=QueryStatus.OK,
            corrections=[CorrectionRound(attempt=1, stage=stage, sql="x", error="e")],
        )

    assert _first_attempt_valid(QueryResult(question="q", sql="s", status=QueryStatus.OK))
    assert not _first_attempt_valid(result("guardrail"))
    assert not _first_attempt_valid(result("explain"))
    assert _first_attempt_valid(result("execute"))


def test_oracle_llm_returns_reference_sql_for_generation_and_correction() -> None:
    llm = OracleLLM({"How many?": "SELECT 1"})
    assert llm.complete("### Question\nHow many?\n\nSQL:") == "SELECT 1"
    correction = "Your previous postgres SQL failed.\n### Question\nHow many?\n\n### Failed SQL\nx"
    assert llm.complete(correction) == "SELECT 1"
