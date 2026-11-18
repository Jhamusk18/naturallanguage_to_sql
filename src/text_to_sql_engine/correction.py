"""The self-correction loop: generate -> validate -> EXPLAIN -> execute.

Failures at any stage are classified and — when a rewrite could plausibly
help — fed back to the model with the failing SQL and the exact error, up to
``max_corrections`` rounds. Non-retryable failures (permissions, read-only
violations, timeouts) stop the loop immediately: retrying cannot fix policy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import sqlglot
from sqlglot import exp

from text_to_sql_engine.execution.errors import ErrorCategory, is_retryable
from text_to_sql_engine.execution.executor import ExecutionError, ExecutionResult, QueryExecutor
from text_to_sql_engine.guardrails.base import GuardrailPipeline, GuardrailViolation
from text_to_sql_engine.llm.base import LLMProvider
from text_to_sql_engine.models import CorrectionRound, DatabaseSchema, QueryStatus
from text_to_sql_engine.prompts import CANNOT_ANSWER, SYSTEM_PROMPT, PromptBuilder

_FENCE_RE = re.compile(r"```(?:sql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_DECLINE_RE = re.compile(
    rf"^\s*(?:```\w*\s*)?(?:--\s*)?`?{CANNOT_ANSWER}`?\s*:?\s*(.*)", re.DOTALL
)


def declined_reason(text: str) -> str | None:
    """The model's reason when it answered CANNOT_ANSWER instead of SQL, else None."""
    match = _DECLINE_RE.match(text)
    if match is None:
        return None
    reason = match.group(1).strip().strip("`").strip()
    return reason or "the schema does not contain what the question needs"


def catalog_hint(sql: str, schema: DatabaseSchema | None) -> str:
    """Ground a correction in the real catalog.

    Lists every existing table, plus the real column names of each existing
    table the failed SQL referenced. Unknown table names are called out. This
    is what turns "column orders.amount does not exist" into an error the
    model can actually fix.
    """
    if schema is None or not schema.tables:
        return ""
    by_name = {table.name.lower(): table for table in schema.tables}
    try:
        tree = sqlglot.parse_one(sql, read="postgres")
        referenced = {t.name.lower() for t in tree.find_all(exp.Table) if t.name}
    except Exception:  # unparseable SQL: fall back to "all tables, no columns"
        referenced = set()
    ctes = set()
    if referenced:
        ctes = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    lines = ["Existing tables: " + ", ".join(t.name for t in schema.tables)]
    unknown = sorted(referenced - set(by_name) - ctes)
    if unknown:
        lines.append("These tables do NOT exist: " + ", ".join(unknown))
    for name in sorted(referenced & set(by_name)):
        table = by_name[name]
        lines.append(f"Columns of {table.name}: " + ", ".join(c.name for c in table.columns))
    return "\n".join(lines)


def extract_sql(text: str) -> str:
    """Pull SQL out of a model response: unwrap fences, trim, drop trailing ';'."""
    match = _FENCE_RE.search(text)
    candidate = (match.group(1) if match else text).strip()
    return candidate.rstrip(";").strip()


@dataclass(frozen=True)
class _Failure:
    stage: str  # "guardrail" | "explain" | "execute"
    message: str
    category: ErrorCategory | None
    retryable: bool


@dataclass(frozen=True)
class _Attempt:
    """Outcome of one validate/EXPLAIN/execute pass."""

    validated_sql: str | None
    result: ExecutionResult | None
    failure: _Failure | None


@dataclass(frozen=True)
class LoopOutcome:
    """Everything the pipeline needs to build the final QueryResult."""

    status: QueryStatus
    sql: str | None
    result: ExecutionResult | None
    rounds: list[CorrectionRound]
    attempts: int
    error: str | None


class CorrectionLoop:
    def __init__(
        self,
        llm: LLMProvider,
        guardrails: GuardrailPipeline,
        executor: QueryExecutor,
        prompts: PromptBuilder,
        max_corrections: int = 2,
        *,
        schema: DatabaseSchema | None = None,
        schema_text: str = "",
    ) -> None:
        self._llm = llm
        self._guardrails = guardrails
        self._executor = executor
        self._prompts = prompts
        self._max_corrections = max_corrections
        self._schema = schema
        self._schema_text = schema_text

    def run(self, question: str, initial_prompt: str) -> LoopOutcome:
        rounds: list[CorrectionRound] = []
        raw = self._llm.complete(initial_prompt, system=SYSTEM_PROMPT)
        for attempt in range(1, self._max_corrections + 2):
            reason = declined_reason(raw)
            if reason is not None:
                return LoopOutcome(
                    QueryStatus.UNANSWERABLE, None, None, rounds, attempt,
                    f"unanswerable: {reason}",
                )
            candidate = extract_sql(raw)
            outcome = self._try_attempt(candidate)
            if outcome.failure is None and outcome.result is not None:
                return LoopOutcome(
                    QueryStatus.OK, outcome.validated_sql, outcome.result, rounds, attempt, None
                )
            failure = outcome.failure
            if failure is None:  # pragma: no cover — _try_attempt always sets one or the other
                raise RuntimeError("attempt produced neither result nor failure")
            rounds.append(
                CorrectionRound(
                    attempt=attempt,
                    stage=failure.stage,
                    sql=candidate,
                    error=failure.message,
                    category=failure.category.value if failure.category else None,
                )
            )
            if not failure.retryable or attempt > self._max_corrections:
                status = (
                    QueryStatus.REJECTED if failure.stage == "guardrail" else QueryStatus.ERROR
                )
                final_sql = outcome.validated_sql or candidate
                return LoopOutcome(status, final_sql, None, rounds, attempt, failure.message)
            raw = self._llm.complete(
                self._prompts.correction(
                    question,
                    candidate,
                    failure.message,
                    schema_text=self._schema_text,
                    catalog_hint=catalog_hint(candidate, self._schema),
                ),
                system=SYSTEM_PROMPT,
            )
        raise RuntimeError("unreachable: loop always returns within the attempt budget")

    def check(self, candidate: str) -> _Attempt:
        """Validate, EXPLAIN and execute one candidate (used by the tool-calling loop)."""
        return self._try_attempt(candidate)

    def _try_attempt(self, candidate: str) -> _Attempt:
        """One pass through guardrails, EXPLAIN preflight, and execution."""
        try:
            validated = self._guardrails.validate(candidate)
        except GuardrailViolation as violation:
            return _Attempt(None, None, _Failure("guardrail", str(violation), None, True))
        try:
            self._executor.explain(validated)
        except ExecutionError as error:
            failure = _Failure(
                "explain", error.message, error.category, is_retryable(error.category)
            )
            return _Attempt(validated, None, failure)
        try:
            result = self._executor.execute(validated)
        except ExecutionError as error:
            failure = _Failure(
                "execute", error.message, error.category, is_retryable(error.category)
            )
            return _Attempt(validated, None, failure)
        return _Attempt(validated, result, None)
