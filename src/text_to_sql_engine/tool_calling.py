"""SQL generation and self-correction through LLM function (tool) calling.

The text mode (``correction.py``) asks for SQL as free text and parses it out
of the reply. This mode instead gives the model three tools:

* ``describe_table(table)``  look up one table's columns, types, keys, value
  lists and sample rows on demand (schema retrieval driven by the model);
* ``submit_sql(sql)``        hand over the final query as a structured argument;
* ``cannot_answer(reason)``  decline when the schema cannot answer the question.

Every ``submit_sql`` goes through the same guardrails, EXPLAIN preflight and
read-only execution as text mode. A failure is returned to the model as the
tool result (error + catalog facts) inside the same conversation, so the
model sees its own previous attempts when it retries. Attempt counting and
stopping rules match text mode: 1 + ``max_corrections`` submissions, and
non-retryable errors stop immediately.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from text_to_sql_engine.correction import (
    CorrectionLoop,
    LoopOutcome,
    catalog_hint,
    declined_reason,
)
from text_to_sql_engine.models import CorrectionRound, DatabaseSchema, QueryStatus
from text_to_sql_engine.prompts import DECLINE_RULE, SYSTEM_PROMPT
from text_to_sql_engine.schema.cards import build_card

# describe_table calls allowed per question, on top of submissions; bounds cost.
MAX_LOOKUPS = 4

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "describe_table",
            "description": "Columns, types, keys, value lists and sample rows of one table.",
            "parameters": {
                "type": "object",
                "properties": {"table": {"type": "string"}},
                "required": ["table"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_sql",
            "description": "Submit one read-only PostgreSQL SELECT that answers the question. "
            "It is validated and executed; errors come back as the tool result.",
            "parameters": {
                "type": "object",
                "properties": {"sql": {"type": "string"}},
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cannot_answer",
            "description": "Use when the schema lacks a needed table/column or the question "
            "is too vague to map to one query. Never substitute a different column.",
            "parameters": {
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "required": ["reason"],
            },
        },
    },
]

TOOL_SYSTEM_PROMPT = (
    SYSTEM_PROMPT.replace("you respond with SQL only — no prose, no markdown fences.", "")
    + " Answer by calling submit_sql exactly once per attempt, or cannot_answer. Call "
    "describe_table first if you need columns the prompt does not show."
)


class ToolCallingLLM(Protocol):
    """A provider that supports OpenAI-style tool calls (Groq, OpenAI)."""

    def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Return ``{"content": str | None, "tool_calls": [{id, name, arguments}]}``."""
        ...


class ToolCallingLoop:
    def __init__(
        self,
        llm: ToolCallingLLM,
        checker: CorrectionLoop,
        schema: DatabaseSchema,
        max_corrections: int = 2,
    ) -> None:
        self._llm = llm
        self._checker = checker
        self._schema = schema
        self._max_corrections = max_corrections

    def _describe(self, table: str) -> str:
        for info in self._schema.tables:
            if info.name.lower() == table.strip().lower():
                return build_card(info).text
        names = ", ".join(t.name for t in self._schema.tables)
        return f"Table {table!r} does not exist. Existing tables: {names}"

    def run(self, question: str, initial_prompt: str) -> LoopOutcome:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": TOOL_SYSTEM_PROMPT},
            {"role": "user", "content": initial_prompt.replace(
                DECLINE_RULE,
                "If the question needs a table or column that does not exist, or is too vague, "
                "call cannot_answer (never substitute a different column).",
            )},
        ]
        rounds: list[CorrectionRound] = []
        attempt = 0
        lookups = 0
        last_sql: str | None = None
        last_error = "the model did not call a tool"
        while True:
            reply = self._llm.chat(messages, TOOLS)
            calls = reply.get("tool_calls") or []
            if not calls:
                text = reply.get("content") or ""
                reason = declined_reason(text)
                if reason is not None:
                    return LoopOutcome(QueryStatus.UNANSWERABLE, None, None, rounds,
                                       max(attempt, 1), f"unanswerable: {reason}")
                # No tool call: treat the text as a submission so it is still checked.
                calls = [{"id": "text", "name": "submit_sql",
                          "arguments": json.dumps({"sql": reply.get("content") or ""})}]
            messages.append(_assistant_message(reply, calls))
            for call in calls:
                try:
                    args = json.loads(call.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                name = call.get("name")
                if name == "cannot_answer":
                    reason = str(args.get("reason") or "the schema cannot answer this")
                    return LoopOutcome(QueryStatus.UNANSWERABLE, None, None, rounds,
                                       max(attempt, 1), f"unanswerable: {reason}")
                if name == "describe_table":
                    lookups += 1
                    result = (self._describe(str(args.get("table", "")))
                              if lookups <= MAX_LOOKUPS else "Lookup limit reached; submit SQL.")
                    messages.append(_tool_result(call, result))
                    continue
                attempt += 1
                candidate = str(args.get("sql") or "").strip().rstrip(";")
                last_sql = candidate
                outcome = self._checker.check(candidate)
                if outcome.failure is None and outcome.result is not None:
                    return LoopOutcome(QueryStatus.OK, outcome.validated_sql, outcome.result,
                                       rounds, attempt, None)
                failure = outcome.failure
                assert failure is not None
                last_error = failure.message
                rounds.append(CorrectionRound(
                    attempt=attempt, stage=failure.stage, sql=candidate, error=failure.message,
                    category=failure.category.value if failure.category else None,
                ))
                if not failure.retryable or attempt > self._max_corrections:
                    status = (QueryStatus.REJECTED if failure.stage == "guardrail"
                              else QueryStatus.ERROR)
                    return LoopOutcome(status, outcome.validated_sql or candidate, None, rounds,
                                       attempt, failure.message)
                hint = catalog_hint(candidate, self._schema)
                messages.append(_tool_result(
                    call, f"FAILED at {failure.stage}: {failure.message}\n{hint}\n"
                    "Fix the query and call submit_sql again.",
                ))
            if lookups > MAX_LOOKUPS + 2:  # model keeps looking things up without submitting
                return LoopOutcome(QueryStatus.ERROR, last_sql, None, rounds, max(attempt, 1),
                                   last_error)


def _assistant_message(reply: dict[str, Any], calls: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": reply.get("content") or "",
        "tool_calls": [
            {"id": c["id"], "type": "function",
             "function": {"name": c["name"], "arguments": c.get("arguments") or "{}"}}
            for c in calls
        ],
    }


def _tool_result(call: dict[str, Any], content: str) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": call["id"], "content": content}
