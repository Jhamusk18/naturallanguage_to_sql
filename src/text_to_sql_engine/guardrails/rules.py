"""The individual guardrail rules. Each is independently testable.

Default order matters and is encoded in :func:`default_rules`:

1. select-only        — reject anything that is not a read query
2. no-comments        — strip comments so nothing smuggles past later rules
3. table-allowlist    — every referenced table must exist in the schema
4. no-star-sensitive  — SELECT * (or every column) over PII-flagged tables is denied
   reads-a-table      — constant queries that read no table are rejected
5. qualify-columns    — rewrite to fully-qualified columns (schema-checked)
6. limit-inject       — add LIMIT when the model forgot one
"""

from __future__ import annotations

from sqlglot import exp
from sqlglot.errors import OptimizeError, SqlglotError
from sqlglot.optimizer.qualify import qualify
from sqlglot.schema import ensure_schema

from text_to_sql_engine.guardrails.base import GuardrailContext, GuardrailRule, GuardrailViolation

# Node types that must never appear anywhere in an accepted statement — not
# just at the root. Postgres allows data-modifying CTEs (WITH x AS (DELETE
# ...) SELECT ...), SELECT ... INTO creates a table, and FOR UPDATE takes row
# locks, so a root-only check would leave real write paths open.
_FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.TruncateTable,
    exp.Grant,
    exp.Command,
    exp.Transaction,
    exp.Commit,
    exp.Rollback,
    exp.Set,
    exp.Use,
    exp.Copy,
    exp.Into,
    exp.Lock,
)


class SelectOnlyRule:
    """Rejects any statement that is not a pure SELECT (or a set operation of
    SELECTs, e.g. UNION) — and walks the whole tree, so DML smuggled inside a
    CTE, ``SELECT ... INTO``, and ``FOR UPDATE`` locks are rejected too."""

    name = "select-only"

    def apply(self, expression: exp.Expression, ctx: GuardrailContext) -> exp.Expression:
        if not isinstance(expression, (exp.Select, exp.SetOperation)):
            kind = expression.key.upper()
            raise GuardrailViolation(self.name, f"only SELECT statements are allowed, got {kind}")
        for node in expression.walk():
            if isinstance(node, _FORBIDDEN_NODES):
                kind = node.key.upper()
                raise GuardrailViolation(
                    self.name, f"forbidden {kind} expression inside a SELECT statement"
                )
        return expression


class NoCommentsRule:
    """Strips every comment from the AST. Comments are a classic smuggling
    channel (hints, hidden instructions, terminator tricks); none survive."""

    name = "no-comments"

    def apply(self, expression: exp.Expression, ctx: GuardrailContext) -> exp.Expression:
        for node in expression.walk():
            node.comments = None
        return expression


class TableAllowlistRule:
    """Every table reference must be a table the introspected schema knows
    about. CTE names defined inside the query are allowed; anything else
    (system catalogs, other schemas, typos) is rejected."""

    name = "table-allowlist"

    def apply(self, expression: exp.Expression, ctx: GuardrailContext) -> exp.Expression:
        allowed = ctx.schema.table_names()
        cte_names = {cte.alias_or_name for cte in expression.find_all(exp.CTE)}
        for table in expression.find_all(exp.Table):
            name = table.name
            if name in cte_names:
                continue
            if name not in allowed:
                raise GuardrailViolation(self.name, f"table '{name}' is not in the allowlist")
        return expression


class SensitiveStarRule:
    """Denies SELECT * (bare or alias-qualified) over tables flagged
    sensitive, so a casual question can never bulk-export a PII table."""

    name = "no-star-sensitive"

    def apply(self, expression: exp.Expression, ctx: GuardrailContext) -> exp.Expression:
        sensitive = ctx.schema.sensitive_tables()
        if not sensitive:
            return expression
        for select in expression.find_all(exp.Select):
            sources = _select_sources(select)
            for projection in select.expressions:
                if isinstance(projection, exp.Star):
                    exposed = sorted(name for name in sources.values() if name in sensitive)
                    if exposed:
                        listed = ", ".join(exposed)
                        raise GuardrailViolation(
                            self.name,
                            f"SELECT * is not allowed over sensitive table(s): {listed}",
                        )
                elif isinstance(projection, exp.Column) and isinstance(projection.this, exp.Star):
                    source = sources.get(projection.table)
                    if source in sensitive:
                        raise GuardrailViolation(
                            self.name,
                            f"SELECT {projection.table}.* is not allowed: "
                            f"'{source}' is flagged sensitive",
                        )
        return expression


class SensitiveAllColumnsRule:
    """Spelling out every column of a sensitive table is ``SELECT *`` in disguise."""

    name = "no-all-columns-sensitive"

    def apply(self, expression: exp.Expression, ctx: GuardrailContext) -> exp.Expression:
        sensitive = ctx.schema.sensitive_tables()
        if not sensitive:
            return expression
        tables = {table.name: table for table in ctx.schema.tables}
        for select in expression.find_all(exp.Select):
            sources = _select_sources(select)
            projected: dict[str, set[str]] = {}
            for projection in select.expressions:
                column = projection.unalias()
                if not isinstance(column, exp.Column):
                    continue
                if column.table:
                    source = sources.get(column.table)
                    owners = [source] if source else []
                else:
                    owners = list(set(sources.values()))
                for owner in owners:
                    projected.setdefault(owner, set()).add(column.name)
            for owner, names in projected.items():
                if owner in sensitive and set(tables[owner].column_names()) <= names:
                    raise GuardrailViolation(
                        self.name,
                        f"selecting every column of sensitive table '{owner}' is not allowed",
                    )
        return expression


class NoConstantQueryRule:
    """A query that reads no table cannot answer a data question; models emit
    ``SELECT 1 WHERE FALSE`` or an error string as a disguised refusal."""

    name = "reads-a-table"

    def apply(self, expression: exp.Expression, ctx: GuardrailContext) -> exp.Expression:
        if not any(True for _ in expression.find_all(exp.Table)):
            raise GuardrailViolation(
                self.name, "the query reads no table, so it cannot answer a data question"
            )
        return expression


class QualifyColumnsRule:
    """Rewrites every column to a fully-qualified reference using the
    introspected schema. As a side effect this validates column existence
    and ambiguity *before* the database ever sees the query."""

    name = "qualify-columns"

    def apply(self, expression: exp.Expression, ctx: GuardrailContext) -> exp.Expression:
        try:
            return qualify(
                expression,
                schema=ensure_schema(ctx.schema.to_sqlglot_schema(), dialect=ctx.dialect),
                dialect=ctx.dialect,
                quote_identifiers=False,
            )
        except (OptimizeError, SqlglotError) as error:
            raise GuardrailViolation(self.name, str(error)) from error


class LimitInjectRule:
    """Adds ``LIMIT <default>`` to the outermost query when absent, so an
    unbounded ``SELECT`` can never stream a whole table to the client."""

    name = "limit-inject"

    def apply(self, expression: exp.Expression, ctx: GuardrailContext) -> exp.Expression:
        if isinstance(expression, (exp.Select, exp.SetOperation)) and (
            expression.args.get("limit") is None
        ):
            return expression.limit(ctx.default_limit)
        return expression


def _select_sources(select: exp.Select) -> dict[str, str]:
    """Map alias-or-name -> base table name for one SELECT scope."""
    sources: dict[str, str] = {}
    candidates: list[exp.Expression] = []
    from_clause = select.args.get("from_")
    if isinstance(from_clause, exp.From):
        candidates.append(from_clause.this)
    for join in select.args.get("joins") or []:
        candidates.append(join.this)
    for node in candidates:
        if isinstance(node, exp.Table):
            sources[node.alias_or_name] = node.name
    return sources


def default_rules() -> list[GuardrailRule]:
    """The production rule chain, in enforcement order."""
    return [
        SelectOnlyRule(),
        NoCommentsRule(),
        TableAllowlistRule(),
        SensitiveStarRule(),
        SensitiveAllColumnsRule(),
        NoConstantQueryRule(),
        QualifyColumnsRule(),
        LimitInjectRule(),
    ]
