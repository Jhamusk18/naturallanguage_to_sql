"""The self-correction loop state machine, driven by a scriptable FakeLLM."""

from __future__ import annotations

from text_to_sql_engine.correction import CorrectionLoop, extract_sql
from text_to_sql_engine.execution.errors import ErrorCategory
from text_to_sql_engine.execution.executor import ExecutionError, ExecutionResult, FakeExecutor
from text_to_sql_engine.guardrails.base import GuardrailPipeline
from text_to_sql_engine.llm.fake import FakeLLM
from text_to_sql_engine.models import QueryStatus
from text_to_sql_engine.prompts import PromptBuilder

GOOD_SQL = "SELECT orders.id FROM orders"
BAD_GUARDRAIL_SQL = "UPDATE orders SET status = 'x'"


def _loop(
    llm: FakeLLM,
    guardrails: GuardrailPipeline,
    executor: FakeExecutor,
    prompts: PromptBuilder,
    max_corrections: int = 2,
) -> CorrectionLoop:
    return CorrectionLoop(llm, guardrails, executor, prompts, max_corrections=max_corrections)


class TestExtractSql:
    def test_unwraps_sql_fence(self) -> None:
        assert extract_sql("```sql\nSELECT 1\n```") == "SELECT 1"

    def test_unwraps_anonymous_fence(self) -> None:
        assert extract_sql("```\nSELECT 1\n```") == "SELECT 1"

    def test_strips_trailing_semicolon(self) -> None:
        assert extract_sql("SELECT 1;") == "SELECT 1"

    def test_passes_through_bare_sql(self) -> None:
        assert extract_sql("  SELECT 1  ") == "SELECT 1"


class TestCorrectionLoop:
    def test_clean_first_attempt_succeeds(
        self, guardrails: GuardrailPipeline, prompts: PromptBuilder
    ) -> None:
        llm = FakeLLM(script=[GOOD_SQL])
        executor = FakeExecutor(default_result=ExecutionResult(columns=["id"], rows=[{"id": 1}]))
        outcome = _loop(llm, guardrails, executor, prompts).run("q", "initial prompt")
        assert outcome.status is QueryStatus.OK
        assert outcome.attempts == 1
        assert outcome.rounds == []
        assert outcome.sql is not None and "LIMIT 200" in outcome.sql
        assert len(llm.calls) == 1

    def test_guardrail_failure_then_success(
        self, guardrails: GuardrailPipeline, prompts: PromptBuilder
    ) -> None:
        llm = FakeLLM(script=[BAD_GUARDRAIL_SQL, GOOD_SQL])
        executor = FakeExecutor()
        outcome = _loop(llm, guardrails, executor, prompts).run("q", "initial prompt")
        assert outcome.status is QueryStatus.OK
        assert outcome.attempts == 2
        assert [round_.stage for round_ in outcome.rounds] == ["guardrail"]
        # The correction prompt carried the failing SQL and the violation text.
        assert BAD_GUARDRAIL_SQL in llm.calls[1]
        assert "select-only" in llm.calls[1]

    def test_explain_failure_then_success(
        self, guardrails: GuardrailPipeline, prompts: PromptBuilder
    ) -> None:
        llm = FakeLLM(script=[GOOD_SQL, GOOD_SQL])
        executor = FakeExecutor(
            explain_script=[
                ExecutionError('column "bogus" does not exist', sqlstate="42703"),
                None,
            ]
        )
        outcome = _loop(llm, guardrails, executor, prompts).run("q", "initial prompt")
        assert outcome.status is QueryStatus.OK
        assert outcome.attempts == 2
        assert outcome.rounds[0].stage == "explain"
        assert outcome.rounds[0].category == ErrorCategory.UNDEFINED_COLUMN.value
        assert len(executor.explained) == 2
        assert len(executor.executed) == 1

    def test_execute_failure_then_success(
        self, guardrails: GuardrailPipeline, prompts: PromptBuilder
    ) -> None:
        llm = FakeLLM(script=[GOOD_SQL, GOOD_SQL])
        executor = FakeExecutor(
            execute_script=[
                ExecutionError("division by zero"),
                ExecutionResult(columns=["id"], rows=[]),
            ]
        )
        outcome = _loop(llm, guardrails, executor, prompts).run("q", "initial prompt")
        assert outcome.status is QueryStatus.OK
        assert outcome.attempts == 2
        assert outcome.rounds[0].stage == "execute"

    def test_budget_exhaustion_yields_rejected(
        self, guardrails: GuardrailPipeline, prompts: PromptBuilder
    ) -> None:
        llm = FakeLLM(script=[BAD_GUARDRAIL_SQL, BAD_GUARDRAIL_SQL, BAD_GUARDRAIL_SQL])
        executor = FakeExecutor()
        outcome = _loop(llm, guardrails, executor, prompts, max_corrections=2).run("q", "p")
        assert outcome.status is QueryStatus.REJECTED
        assert outcome.attempts == 3  # 1 initial + 2 corrections
        assert len(outcome.rounds) == 3
        assert executor.explained == []  # nothing ever reached the database

    def test_non_retryable_error_stops_immediately(
        self, guardrails: GuardrailPipeline, prompts: PromptBuilder
    ) -> None:
        llm = FakeLLM(script=[GOOD_SQL, GOOD_SQL, GOOD_SQL])
        executor = FakeExecutor(
            explain_script=[ExecutionError("permission denied for table orders")]
        )
        outcome = _loop(llm, guardrails, executor, prompts).run("q", "p")
        assert outcome.status is QueryStatus.ERROR
        assert outcome.attempts == 1
        assert outcome.rounds[0].category == ErrorCategory.PERMISSION.value
        assert len(llm.calls) == 1  # no correction prompt was sent

    def test_timeout_is_not_retried(
        self, guardrails: GuardrailPipeline, prompts: PromptBuilder
    ) -> None:
        llm = FakeLLM(script=[GOOD_SQL, GOOD_SQL])
        executor = FakeExecutor(
            execute_script=[
                ExecutionError("canceling statement due to statement timeout", sqlstate="57014")
            ]
        )
        outcome = _loop(llm, guardrails, executor, prompts).run("q", "p")
        assert outcome.status is QueryStatus.ERROR
        assert outcome.attempts == 1
        assert outcome.rounds[0].category == ErrorCategory.TIMEOUT.value

    def test_zero_corrections_budget(
        self, guardrails: GuardrailPipeline, prompts: PromptBuilder
    ) -> None:
        llm = FakeLLM(script=[BAD_GUARDRAIL_SQL, GOOD_SQL])
        executor = FakeExecutor()
        outcome = _loop(llm, guardrails, executor, prompts, max_corrections=0).run("q", "p")
        assert outcome.status is QueryStatus.REJECTED
        assert outcome.attempts == 1
        assert len(llm.calls) == 1


class TestDeclineAndGrounding:
    """CANNOT_ANSWER handling and catalog-grounded correction prompts."""

    def test_declined_reason_parses_variants(self) -> None:
        from text_to_sql_engine.correction import declined_reason

        assert declined_reason("CANNOT_ANSWER: no supplier column") == "no supplier column"
        assert declined_reason("```\nCANNOT_ANSWER: vague\n```") == "vague"
        assert declined_reason("SELECT 1") is None

    def test_decline_on_first_attempt_is_unanswerable_and_never_executes(
        self, guardrails: GuardrailPipeline, prompts: PromptBuilder
    ) -> None:
        llm = FakeLLM(script=["CANNOT_ANSWER: there is no supplier table"])
        executor = FakeExecutor()
        outcome = _loop(llm, guardrails, executor, prompts).run("q", "p")
        assert outcome.status is QueryStatus.UNANSWERABLE
        assert outcome.error is not None and "supplier" in outcome.error
        assert outcome.attempts == 1 and executor.explained == []

    def test_catalog_hint_lists_real_columns_and_unknown_tables(self) -> None:
        from text_to_sql_engine.correction import catalog_hint
        from text_to_sql_engine.schema.fixtures import DEMO_SCHEMA

        hint = catalog_hint("SELECT o.amount FROM orders o JOIN shipments s ON 1=1", DEMO_SCHEMA)
        assert "These tables do NOT exist: shipments" in hint
        assert "Columns of orders: id, customer_id" in hint
        assert "Existing tables:" in hint

    def test_correction_prompt_carries_schema_and_catalog_facts(
        self, guardrails: GuardrailPipeline, prompts: PromptBuilder
    ) -> None:
        from text_to_sql_engine.schema.fixtures import DEMO_SCHEMA

        llm = FakeLLM(script=[GOOD_SQL, GOOD_SQL])
        executor = FakeExecutor(
            explain_script=[
                ExecutionError('column "amount" does not exist', sqlstate="42703"),
                None,
            ]
        )
        loop = CorrectionLoop(
            llm, guardrails, executor, prompts, 2, schema=DEMO_SCHEMA, schema_text="table orders"
        )
        outcome = loop.run("q", "p")
        assert outcome.status is QueryStatus.OK
        retry = llm.calls[1]
        assert "### Catalog facts" in retry and "Columns of orders:" in retry
        assert "### Schema\ntable orders" in retry
