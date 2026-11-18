"""Function-calling generation: describe_table, submit_sql retries, cannot_answer."""

from __future__ import annotations

import json
from typing import Any

from text_to_sql_engine.correction import CorrectionLoop
from text_to_sql_engine.execution.executor import ExecutionError, ExecutionResult, FakeExecutor
from text_to_sql_engine.guardrails.base import GuardrailPipeline
from text_to_sql_engine.llm.fake import FakeLLM
from text_to_sql_engine.models import QueryStatus
from text_to_sql_engine.prompts import PromptBuilder
from text_to_sql_engine.schema.fixtures import DEMO_SCHEMA
from text_to_sql_engine.tool_calling import ToolCallingLoop


class ScriptedToolLLM:
    """Returns one scripted tool call per turn and records the conversations."""

    def __init__(self, calls: list[tuple[str, dict[str, str]]]) -> None:
        self._calls = list(calls)
        self.seen: list[list[dict[str, Any]]] = []

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        self.seen.append([dict(m) for m in messages])
        name, args = self._calls.pop(0)
        call_id = f"c{len(self.seen)}"
        return {"content": None,
                "tool_calls": [{"id": call_id, "name": name, "arguments": json.dumps(args)}]}


def _loop(
    llm: ScriptedToolLLM, guardrails: GuardrailPipeline, executor: FakeExecutor
) -> ToolCallingLoop:
    checker = CorrectionLoop(FakeLLM(), guardrails, executor, PromptBuilder(), 2,
                             schema=DEMO_SCHEMA)
    return ToolCallingLoop(llm, checker, DEMO_SCHEMA, max_corrections=2)


def test_describe_then_submit_succeeds(guardrails: GuardrailPipeline) -> None:
    llm = ScriptedToolLLM([
        ("describe_table", {"table": "orders"}),
        ("submit_sql", {"sql": "SELECT orders.id FROM orders"}),
    ])
    executor = FakeExecutor(default_result=ExecutionResult(columns=["id"], rows=[{"id": 1}]))
    outcome = _loop(llm, guardrails, executor).run("q", "prompt")
    assert outcome.status is QueryStatus.OK and outcome.attempts == 1
    tool_reply = llm.seen[1][-1]
    assert tool_reply["role"] == "tool" and "table orders" in tool_reply["content"]


def test_failed_submission_error_and_catalog_are_returned_in_conversation(
    guardrails: GuardrailPipeline,
) -> None:
    llm = ScriptedToolLLM([
        ("submit_sql", {"sql": "SELECT orders.id FROM orders"}),
        ("submit_sql", {"sql": "SELECT orders.id FROM orders"}),
    ])
    executor = FakeExecutor(
        explain_script=[ExecutionError('column "x" does not exist', sqlstate="42703"), None]
    )
    outcome = _loop(llm, guardrails, executor).run("q", "prompt")
    assert outcome.status is QueryStatus.OK and outcome.attempts == 2
    feedback = llm.seen[1][-1]["content"]
    assert "FAILED at explain" in feedback and "Columns of orders:" in feedback


def test_cannot_answer_is_unanswerable_and_never_executes(guardrails: GuardrailPipeline) -> None:
    llm = ScriptedToolLLM([("cannot_answer", {"reason": "no supplier table"})])
    executor = FakeExecutor()
    outcome = _loop(llm, guardrails, executor).run("q", "prompt")
    assert outcome.status is QueryStatus.UNANSWERABLE
    assert executor.explained == []


def test_retry_budget_matches_text_mode(guardrails: GuardrailPipeline) -> None:
    bad = ("submit_sql", {"sql": "DELETE FROM orders"})
    llm = ScriptedToolLLM([bad, bad, bad])
    outcome = _loop(llm, guardrails, FakeExecutor()).run("q", "prompt")
    assert outcome.status is QueryStatus.REJECTED and outcome.attempts == 3
