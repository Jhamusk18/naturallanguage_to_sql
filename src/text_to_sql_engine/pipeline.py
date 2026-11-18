"""End-to-end orchestration: question in, validated SQL + rows + summary out."""

from __future__ import annotations

import time

from text_to_sql_engine.correction import CorrectionLoop
from text_to_sql_engine.embeddings.base import Embedder
from text_to_sql_engine.execution.executor import ExecutionResult, QueryExecutor
from text_to_sql_engine.fewshot.store import FewShotStore
from text_to_sql_engine.guardrails.base import GuardrailContext, GuardrailPipeline
from text_to_sql_engine.guardrails.rules import default_rules
from text_to_sql_engine.history import HistoryStore
from text_to_sql_engine.llm.base import LLMProvider
from text_to_sql_engine.models import QueryResult, QueryStatus
from text_to_sql_engine.prompts import PromptBuilder
from text_to_sql_engine.schema.cache import SchemaCache
from text_to_sql_engine.schema.cards import select_cards


class TextToSqlPipeline:
    """Wires retrieval, prompting, guardrails, execution, and history together.

    The guardrail pipeline is rebuilt per query from the *current* cached
    schema, so a POST /schema/refresh immediately tightens (or widens) the
    table allowlist without restarting anything.
    """

    def __init__(
        self,
        schema_cache: SchemaCache,
        embedder: Embedder,
        fewshot: FewShotStore,
        llm: LLMProvider,
        executor: QueryExecutor,
        history: HistoryStore,
        *,
        token_budget: int = 1200,
        default_limit: int = 200,
        max_corrections: int = 2,
        dialect: str = "postgres",
    ) -> None:
        self._schema_cache = schema_cache
        self._embedder = embedder
        self._fewshot = fewshot
        self._llm = llm
        self._executor = executor
        self._history = history
        self._token_budget = token_budget
        self._default_limit = default_limit
        self._max_corrections = max_corrections
        self._dialect = dialect
        self._prompts = PromptBuilder(dialect=dialect)

    def answer(self, question: str) -> QueryResult:
        started = time.perf_counter()
        schema = self._schema_cache.get()
        cards = select_cards(question, schema, self._embedder, self._token_budget)
        exemplars = self._fewshot.top_k(question, k=3)
        prompt = self._prompts.initial(question, cards, exemplars)
        guardrails = GuardrailPipeline(
            default_rules(),
            GuardrailContext(
                schema=schema, default_limit=self._default_limit, dialect=self._dialect
            ),
        )
        loop = CorrectionLoop(
            self._llm, guardrails, self._executor, self._prompts, self._max_corrections
        )
        outcome = loop.run(question, prompt)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)

        summary: str | None = None
        columns: list[str] = []
        rows: list[dict[str, object]] = []
        if outcome.status is QueryStatus.OK and outcome.result is not None:
            summary = self._summarize(question, outcome.result)
            columns = outcome.result.columns
            rows = outcome.result.rows
        result = QueryResult(
            question=question,
            sql=outcome.sql,
            status=outcome.status,
            columns=columns,
            rows=rows,
            row_count=len(rows),
            summary=summary,
            retrieved_tables=[card.table for card in cards],
            attempts=outcome.attempts,
            corrections=outcome.rounds,
            latency_ms=latency_ms,
            error=outcome.error,
        )
        self._history.add(question, outcome.sql, outcome.status, latency_ms)
        return result

    def _summarize(self, question: str, result: ExecutionResult) -> str:
        """One-line natural-language summary; deterministic fallback if empty."""
        text = self._llm.complete(self._prompts.summary(question, result)).strip()
        return text or f"{result.row_count} row(s) returned."
