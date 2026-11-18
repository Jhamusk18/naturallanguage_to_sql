"""Schema hallucination detector used by the ablation experiments.

A *hallucination* is a generated query that references a table or column that
does not exist in the introspected schema. Detection is purely static (sqlglot
AST + schema), so it is independent of the database, the guardrails, and the
model: the same function scores every experiment arm.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp
from sqlglot.errors import OptimizeError, SqlglotError
from sqlglot.optimizer.qualify import qualify
from sqlglot.schema import ensure_schema

from text_to_sql_engine.models import DatabaseSchema

_UNKNOWN_COLUMN = re.compile(r"Unknown column: (\S+)")


@dataclass(frozen=True)
class Hallucinations:
    tables: list[str] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)

    @property
    def any(self) -> bool:
        return bool(self.tables or self.columns)


def find_hallucinations(
    sql: str | None, schema: DatabaseSchema, dialect: str = "postgres"
) -> Hallucinations:
    """Unknown tables first; unknown columns only when every table exists.

    Unparseable SQL is a syntax problem, not a hallucination, and scores clean.
    """
    if not sql:
        return Hallucinations()
    try:
        expression = sqlglot.parse_one(sql, dialect=dialect)
    except SqlglotError:
        return Hallucinations()
    known = schema.table_names()
    ctes = {cte.alias_or_name for cte in expression.find_all(exp.CTE)}
    missing = sorted(
        {t.name for t in expression.find_all(exp.Table) if t.name not in known | ctes}
    )
    if missing:
        return Hallucinations(tables=missing)
    try:
        qualify(
            expression,
            schema=ensure_schema(schema.to_sqlglot_schema(), dialect=dialect),
            dialect=dialect,
            quote_identifiers=False,
        )
    except OptimizeError as error:
        match = _UNKNOWN_COLUMN.search(str(error))
        if match:
            return Hallucinations(columns=[match.group(1)])
    except SqlglotError:
        pass
    return Hallucinations()
