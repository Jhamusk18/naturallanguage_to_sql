# Architecture

```
question
  │
  ├─ intent gate (guardrails/intent.py) ── write request? → REJECTED, no LLM call
  │
  ├─ schema context (schema/introspect.py → schema/cards.py, fewshot/store.py)
  │     PROMPT_MODE=schema_aware : ranked cards (types, PK/FK, comments, categorical
  │                                value lists, diverse sample rows) + 3 exemplars
  │     PROMPT_MODE=columns_only : table + column names           (ablation)
  │     PROMPT_MODE=baseline     : table names only               (ablation)
  │
  ├─ generation (prompts.py + llm/*)
  │     GENERATION_MODE=text  : SQL parsed from a text reply (correction.py)
  │     GENERATION_MODE=tools : function calling (tool_calling.py)
  │                             describe_table / submit_sql / cannot_answer
  │
  ├─ check one candidate (CorrectionLoop.check)
  │     guardrails/rules.py   sqlglot AST: select-only, no comments, table allowlist,
  │                           sensitive-table rules, reads-a-table, column
  │                           qualification against the schema, LIMIT injection
  │     executor.explain      EXPLAIN preflight
  │     executor.execute      read-only transaction + statement_timeout
  │
  ├─ on failure: execution/errors.py classifies the error
  │     non-retryable (permission, timeout, read-only) → stop
  │     retryable → feed back failed SQL + exact error + catalog facts
  │                 (existing tables, unknown tables, real columns of the tables
  │                 the query touched) + the schema section; max 2 retries
  │     text mode: a new correction prompt; tools mode: a tool result in the
  │     same conversation
  │
  ├─ CANNOT_ANSWER / cannot_answer → UNANSWERABLE, nothing executed
  │
  └─ QueryResult: status, SQL, rows, summary (LLM, optional), retrieved tables,
     attempts, correction trace → FastAPI (api.py) → Streamlit (ui/app.py)
```

## Providers

`LLMProvider.complete()` is the only interface the text pipeline needs. Gemini
(`google-genai`), Groq and OpenAI (OpenAI SDK) implement it; Groq/OpenAI also
implement `chat(messages, tools)` for function calling. `FallbackLLM` switches
from Gemini to Groq when Gemini fails (quota). `FakeLLM` drives tests only; the
evaluation refuses to score it.

## Why function calling is optional, not the default

Function calling gives structured output (no fence or prose parsing), lets the
model fetch schema on demand (useful for wide schemas), and keeps failed
attempts in one conversation. It costs more tokens per question (the
conversation is re-sent on every step) and needs a provider with tool support
(Gemini adapter here does not implement it). Measured effect: see
`docs/RESUME_EVIDENCE.md`.

## Safety layers

1. Intent gate before the LLM. 2. AST guardrails (writes anywhere in the tree,
including data-modifying CTEs, are rejected). 3. Sensitive tables are never
sampled into prompts and cannot be dumped. 4. Read-only transaction and
server-side timeout, so a missed rule still cannot write or hang.
