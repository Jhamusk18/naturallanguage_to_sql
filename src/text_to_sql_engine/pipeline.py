"""End-to-end orchestration: question in, validated SQL + rows + summary out."""

from __future__ import annotations

import time

from text_to_sql_engine.correction import CorrectionLoop
from text_to_sql_engine.embeddings.base import Embedder
from text_to_sql_engine.execution.executor import ExecutionResult, QueryExecutor
from text_to_sql_engine.fewshot.store import FewShotStore
from text_to_sql_engine.guardrails.base import GuardrailContext, GuardrailPipeline
from text_to_sql_engine.guardrails.intent import detect_write_intent
from text_to_sql_engine.guardrails.rules import default_rules
from text_to_sql_engine.history import HistoryStore
from text_to_sql_engine.llm.base import LLMProvider
from text_to_sql_engine.models import Exemplar, QueryResult, QueryStatus
from text_to_sql_engine.prompts import PromptBuilder
from text_to_sql_engine.schema.cache import SchemaCache
from text_to_sql_engine.schema.cards import (
    build_columns_only_card,
    build_name_only_card,
    select_cards,
)
from text_to_sql_engine.tool_calling import ToolCallingLoop


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
        prompt_mode: str = "schema_aware",
        summarize: bool = True,
        generation_mode: str = "text",
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
        self._prompt_mode = prompt_mode
        self._summarize_enabled = summarize
        if generation_mode == "tools" and not callable(getattr(llm, "chat", None)):
            raise ValueError("GENERATION_MODE=tools needs a tool-calling provider (Groq/OpenAI)")
        self._generation_mode = generation_mode
        self._prompts = PromptBuilder(dialect=dialect)

    @property
    def llm(self) -> LLMProvider:
        return self._llm

    def answer(self, question: str) -> QueryResult:
        started = time.perf_counter()
        phrase = detect_write_intent(question)
        if phrase is not None:
            message = (
                f"intent-gate: the request asks to modify data ('{phrase}'); "
                "this service is read-only"
            )
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            self._history.add(question, None, QueryStatus.REJECTED, latency_ms)
            return QueryResult(
                question=question,
                sql=None,
                status=QueryStatus.REJECTED,
                latency_ms=latency_ms,
                error=message,
            )
        schema = self._schema_cache.get()
        if self._prompt_mode == "baseline":
            # Ablation baseline: bare table names, no retrieval, no exemplars.
            cards = [build_name_only_card(table) for table in schema.tables]
            exemplars: list[Exemplar] = []
        elif self._prompt_mode == "columns_only":
            # Ablation middle arm: every table with its column names, nothing else.
            cards = [build_columns_only_card(table) for table in schema.tables]
            exemplars = []
        else:
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
            self._llm,
            guardrails,
            self._executor,
            self._prompts,
            self._max_corrections,
            schema=schema,
            schema_text="\n".join(card.text for card in cards),
        )
        if self._generation_mode == "tools":
            tool_loop = ToolCallingLoop(
                self._llm,  # type: ignore[arg-type]
                loop,
                schema,
                self._max_corrections,
            )
            outcome = tool_loop.run(question, prompt)
        else:
            outcome = loop.run(question, prompt)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)

        summary: str | None = None
        columns: list[str] = []
        rows: list[dict[str, object]] = []
        if outcome.status is QueryStatus.OK and outcome.result is not None:
            if self._summarize_enabled:
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
