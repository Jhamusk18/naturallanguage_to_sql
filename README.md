# Text-to-SQL Engine

**Natural language to safe, validated SQL: schema-aware prompting, AST guardrails, and a self-correction loop.**

Large language models are good at writing SQL and terrible at being trusted with a database. Asking the model nicely to "only generate SELECT statements" is not a security boundary — it is a suggestion that survives exactly until the first adversarial or confused prompt. This engine treats generated SQL the way a database driver treats user input: as untrusted text. Every candidate statement is parsed into a real AST with sqlglot, pushed through a chain of composable validation and rewrite rules (single-SELECT-only, table allowlist, comment stripping, LIMIT injection, sensitive-table star denial, full column qualification), preflighted with `EXPLAIN`, and executed inside a read-only transaction with a server-side statement timeout. When something still fails, the exact database error is fed back to the model for a bounded number of correction rounds — and the loop knows which errors are worth retrying and which are policy. Everything external (LLM, embeddings, database) sits behind a small protocol with a deterministic fake, so the entire pipeline — including the demo — runs offline.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136-009688?logo=fastapi&logoColor=white)
![sqlglot](https://img.shields.io/badge/sqlglot-30-4B8BBE)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1?logo=postgresql&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-v2-E92063?logo=pydantic&logoColor=white)
![CI](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?logo=githubactions&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Features

- **AST guardrails, not regex.** Generated SQL is parsed once with sqlglot and folded through six independent rules that validate *and rewrite* on the same tree — there is no window where a "checked" string can drift from the SQL that actually runs. The select-only rule walks the whole AST, so DML smuggled inside a Postgres data-modifying CTE, `SELECT ... INTO`, and `FOR UPDATE` row locks are rejected, not just top-level `UPDATE`.
- **Schema introspection with token-budgeted schema cards.** Tables, columns, types, primary/foreign keys, and comments are introspected from `pg_catalog` into compact per-table "cards", ranked by embedding similarity to the question, and greedily packed into a configurable token budget — wide databases stop blowing up the prompt. Cached, with on-demand invalidation via `POST /schema/refresh`.
- **Few-shot retrieval.** Fifteen bundled (question, SQL) exemplars for the demo schema; the top 3 by embedding similarity are included in every generation prompt alongside the schema cards and dialect rules.
- **EXPLAIN-first execution with defense in depth.** Every validated statement is preflighted with `EXPLAIN` (bad references fail before a single row is read), then executed in a read-only transaction with a server-enforced `statement_timeout` — two database-level walls that hold even if a guardrail were bypassed.
- **A self-correction loop that knows when to stop.** Failures at the guardrail, EXPLAIN, or execution stage are classified by SQLSTATE; fixable errors (unknown column, grouping mistakes, syntax) go back to the model with the failing SQL and exact error for up to 2 rounds, while policy errors (timeout, read-only violation, permission) stop immediately.
- **Results with provenance.** The API returns rows, column metadata, the final validated SQL, every failed correction round, and a one-line natural-language summary. Each query (question, SQL, status, latency) is persisted to SQLite and exposed at `GET /history`.
- **Deterministic fakes for everything external.** `FakeLLM` (scriptable + template-driven), `FakeEmbedder` (hashed bag-of-words), `FakeExecutor` (scriptable outcomes), and a fixture schema mean the full test suite and `make demo` run with no network, no API key, and no database.

## Architecture

```mermaid
flowchart LR
    Q[Question] --> CARDS
    subgraph CONTEXT[Context assembly]
        SC[(Schema cache)] --> CARDS[Schema cards<br>token-budgeted ranking]
        CARDS --> PB[Prompt builder]
        FS[Few-shot store<br>top-3 exemplars] --> PB
    end
    PB --> LLM[LLM provider<br>Gemini, OpenAI, or FakeLLM]
    LLM --> GR{Guardrail pipeline<br>sqlglot AST rules}
    GR -- "violation + SQL" --> LLM
    GR -- validated SQL --> EXP{EXPLAIN preflight}
    EXP -- "classified error" --> LLM
    EXP --> EXEC[Execute<br>read-only tx + timeout]
    EXEC -- "classified error" --> LLM
    EXEC --> RES[Rows + summary]
    RES --> HIST[(Query history)]
```

The back edges into the LLM are the correction loop — each carries the failing SQL plus the exact error, and each is bounded by the retry budget and the retryability policy.

## Tech stack

| Technology | Role | Why |
|---|---|---|
| Python 3.12 | Runtime | Modern typing and stdlib (`StrEnum`, `datetime.UTC`) with broad library support |
| sqlglot | SQL parsing, validation, rewriting | A real Postgres-dialect AST — guardrails operate on structure, not strings; also powers column qualification and star expansion |
| FastAPI | HTTP API | Pydantic-native request/response models, OpenAPI for free |
| pydantic v2 | Models & settings | Strict (`extra="forbid"`) validation at every boundary; env-driven config |
| psycopg 3 | PostgreSQL driver | First-class read-only transactions, SQLSTATE surfaces, connection options for `statement_timeout` |
| PostgreSQL 16 | Demo database | Realistic e-commerce schema with comments and FKs to introspect |
| SQLite (stdlib) | Query history | Durable persistence with zero extra dependencies |
| uv + ruff + mypy + pytest | Toolchain | Fast installs, strict lint/type gates, 141 offline tests |

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) (it fetches Python 3.12 if needed).

```bash
git clone https://github.com/Rhed-Dev/text-to-sql-engine.git
cd text-to-sql-engine

make install   # uv venv + editable install with dev extras
make demo      # fully offline: fixture schema, FakeLLM, real guardrails
make test      # 141 tests — no network, no DB, no keys
```

To run against a live PostgreSQL with the bundled e-commerce schema and seed:

```bash
make db-up     # docker compose: postgres:16 + schema + deterministic seed
DATABASE_URL=postgresql://t2s:t2s@localhost:5432/shop make demo
```

The demo asks the same bundled questions either way — the offline fixture rows were computed from the same seed data by `scripts/generate_seed.py`, so both modes print the same numbers.

To serve the API (offline by default; set `DATABASE_URL` and `GEMINI_API_KEY` (or `OPENAI_API_KEY`) to go live):

```bash
make run       # uvicorn on 127.0.0.1:8000

curl -s localhost:8000/query \
  -H 'content-type: application/json' \
  -d '{"question": "Who are our top 3 customers by total spend?"}'
```

## Interview demo guide

### Setup and run

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/) (`pip install --user uv`). Commands below are for Linux/macOS shells; use WSL on Windows.

```bash
make install            # venv + dependencies, including the Streamlit UI extra
cp .env.example .env    # optional: add GEMINI_API_KEY (free tier) and DATABASE_URL

# PostgreSQL, pick one:
make db-up              # Docker (docker compose)
make db-local           # Debian/Ubuntu without Docker (apt + sudo)

# Terminal 1 - backend (http://127.0.0.1:8000, Swagger UI at /docs)
make run
# Terminal 2 - frontend (http://localhost:8501)
make ui
```

With no `.env`, the backend still starts: fake LLM, fixture schema, fixture rows. `GET /healthz` and the banner at the top of the UI always show which backends are real (`llm`: `fake` or `openai`, `database`: `fixture` or `postgres`).

### Fake mode vs live mode

| | Fake mode (no API key) | Live mode |
|---|---|---|
| SQL source | A fixed table of templates in `demo_data.py`, matched by question substring | Gemini (default `gemini-3.5-flash-lite`) or OpenAI, prompted with ranked schema cards and few-shot examples |
| Questions answered | Only the 5 bundled questions (plus one guardrail-rejection example). Anything else is rejected with an explicit "FakeLLM has no template" error | Free-form questions about the connected schema |
| Validation, EXPLAIN, read-only execution, correction loop | Real | Real |
| Database | Real PostgreSQL if `DATABASE_URL` is set, otherwise fixture rows | Real PostgreSQL |

Fake mode demonstrates the guardrails and execution pipeline. It is not general-purpose NL-to-SQL.

### Demo questions

Fake mode (all run against PostgreSQL when `DATABASE_URL` is set):

1. Which 5 products have generated the most revenue?
2. Who are our top 3 customers by total spend?
3. How many orders are in each status?
4. What is the average order value by country?
5. Show me everything about our customers. (rejected: `SELECT *` over the sensitive `customers` table)

Live mode (needs an API key):

1. Who are our top 3 customers by total spend?
2. Which 5 products generated the highest revenue?
3. What is the total revenue by month?
4. Which customers have placed more than 2 orders?
5. Delete all cancelled orders from the database. (rejected: select-only guardrail)

Note on this seed data: no customer has more than 3 orders and every product has been ordered, so "Which customers have placed more than 3 orders?" and "Which products have never been ordered?" correctly return 0 rows. Use "more than 2 orders" for a non-empty result.

### Evaluation

`evaluation/questions.jsonl` has 67 questions written against this schema: 54 answerable (each with a gold SQL), 6 unsafe requests, 4 that reference nonexistent columns or tables, and 3 ambiguous or invalid inputs.

```bash
make eval-oracle    # sanity check of dataset + pipeline using the gold SQL, no LLM
make eval           # real model: needs GEMINI_API_KEY (or OPENAI_API_KEY) and DATABASE_URL
```

`make eval` refuses to run on the FakeLLM. Metrics: first-attempt valid SQL rate (passes guardrails and EXPLAIN), execution success rate, execution accuracy against the gold SQL's rows, correction success rate (of questions whose first attempt failed), unsafe rejection rate, and a before/after row-count check that no table changed. Per-question results are written to `artifacts/`.

What has been measured (full evidence and per-bullet mapping: [`docs/RESUME_EVIDENCE.md`](docs/RESUME_EVIDENCE.md)):

- `make eval-oracle`: 54/54 gold queries pass validation, execute and match; 6/6 unsafe requests rejected; no table changed. This checks the dataset and guardrails, not model quality.
- **v2 ablation** (current code), Groq `openai/gpt-oss-20b` (reasoning effort low, temperature 0, lexical retrieval), 54 answerable questions, all four arms on 2026-11-18, files in `evaluation/results/v2-groq-gpt-oss-20b/`. Each arm runs with up to 2 corrections; the first attempt of that run is the corrections-off result for the same question.

| Prompt | First-try hallucination | Query success OFF → ON | Accuracy OFF → ON | Failed first attempts fixed |
|---|---|---|---|---|
| table names only (`baseline`) | 17/54 (31.5%) | 18/54 (33.3%) → 39/54 (72.2%) | 17/54 → 28/54 | 21/22 |
| table + column names (`columns_only`) | 0/54 | 53/54 (98.1%) → 54/54 | 37/54 (68.5%) → 37/54 | 1/1 |
| schema-aware, no sample rows | 0/54 | 53/54 → 54/54 | 45/54 (83.3%) → 45/54 | 1/1 |
| schema-aware + sample rows (default) | 0/54 | 53/54 (98.1%) → 54/54 (100%) | 45/54 (83.3%) → 45/54 | 1/1 |

  Column-tolerant accuracy (`python -m evaluation.rescore`; the gold columns must appear with identical values, extra columns allowed, row count equal): schema-aware 48/54 (88.9%), column names only 41/54 (75.9%), names only 17/54 → 33/54 with correction. Strict accuracy stays the headline number.

  In every arm 6/6 unsafe requests were refused and no data changed. With any column information the model also declined 3/4 nonexistent-column and 3/3 ambiguous questions instead of inventing SQL; the names-only prompt falsely declined 15/54 answerable questions.
- **v3 ablation, second model** (same code, questions and arms as v2), Gemini `gemini-3.5-flash-lite` (OpenAI-compatible endpoint, temperature 0, no fallback model), all four arms on 2026-11-18, files in `evaluation/results/v3-gemini-3.5-flash-lite/`. This is a separate experiment on a different model and is not pooled with the Groq numbers.

| Prompt | First-try hallucination | Query success OFF → ON | Strict accuracy OFF → ON | Column-tolerant accuracy OFF → ON | Failed first attempts fixed | Falsely declined |
|---|---|---|---|---|---|---|
| table names only (`baseline`) | 10/54 (18.5%) | 7/54 (13.0%) → 17/54 (31.5%) | 5/54 → 12/54 | 5/54 → 12/54 | 10/10 | 37/54 |
| table + column names (`columns_only`) | 0/54 | 49/54 (90.7%) → 53/54 (98.1%) | 36/54 (66.7%) → 37/54 (68.5%) | 39/54 → 40/54 | 4/4 | 1/54 |
| schema-aware, no sample rows | 0/54 | 50/54 (92.6%) → 53/54 (98.1%) | 43/54 (79.6%) → 44/54 (81.5%) | 45/54 → 47/54 | 3/3 | 1/54 |
| schema-aware + sample rows (default) | 0/54 | 51/54 (94.4%) → 54/54 (100%) | 42/54 (77.8%) → 43/54 (79.6%) | 44/54 → 46/54 | 3/3 | 0/54 |

  In every Gemini arm 6/6 unsafe requests were refused, 4/4 nonexistent-column and 3/3 ambiguous questions were not executed, and no data changed. With names only, Gemini declined 37/54 answerable questions instead of guessing, so its low query success there is mostly refusals rather than failed SQL.
- **Function calling** (`GENERATION_MODE=tools`, `tool_calling.py`): runs on the same model were stopped by the Groq free-tier daily token cap (200,000 tokens/day for `gpt-oss-20b`). `evaluation/results/v2-groq-gpt-oss-20b/incomplete/aware_tools_c2.json` holds 13 of 67 questions (13/13 first-try success, 11/13 strict accuracy) and `names_only_tools_c2.json` none; both are marked `incomplete`, `compare.py` refuses them, and no tool-mode result is claimed. Finish with `--resume` once the quota frees.
- **v1 ablation** (code before the catalog-grounded correction prompt), Groq `qwen/qwen3.8-27b`, `evaluation/results/v1-groq-qwen3.8-27b/`: first-try hallucination 34/54 (names only) → 0/54 (schema-aware); correction lifted names-only query success 14/54 → 23/54 (9 of 40 failures fixed).
- Earlier Gemini `gemini-3.5-flash-lite` baseline runs (names-only prompt only; the schema-aware arms were stopped by the free-tier quota) are kept in `evaluation/results/gemini/`; the complete v3 run above supersedes them. Neither is compared with the Groq files.

What these show: schema-aware prompting removes hallucinated tables/columns (100% relative reduction vs names-only, on both models) and raises accuracy over a column-list prompt (68.5% → 83.3%); sample rows on top of metadata had no measurable effect on this small schema. Feeding the error plus the real catalog columns back to the model fixes 95% of failed first attempts (v1 fixed 23%), but with schema-aware prompting first attempts already succeed 98% of the time, so the end-to-end lift is largest when the prompt context is weak. The Gemini run shows the same pattern on a second model: hallucinations 10/54 → 0/54 with schema-aware prompting, strict accuracy 68.5% (column names) → 79.6% (schema-aware), and correction fixed every failed first attempt (3/3 to 10/10 per arm).

### Architecture in one paragraph

Diagram and design notes: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).


A question is embedded and used to rank per-table schema cards (introspected from `pg_catalog`) and pick 3 few-shot examples; the prompt builder assembles dialect rules, cards, examples, and the question. The LLM's SQL is parsed once with sqlglot and folded through AST rules (single statement, SELECT only, table allowlist, comment stripping, LIMIT injection, sensitive-table star denial, column qualification). Validated SQL is preflighted with `EXPLAIN` and executed in a read-only transaction with a server-side timeout. Fixable errors are fed back to the model with the failing SQL for up to 2 rounds; policy errors stop immediately. The FastAPI service returns the SQL, rows, summary, retrieved tables, and the correction trace; the Streamlit app in `ui/app.py` renders them.

### Limitations and future work

- The evaluation set is small (54 answerable questions) and uses one tiny schema (12 customers, 30 orders); accuracy misses are mostly definition disagreements with the gold SQL (extra columns, status filters on revenue).
- Schema retrieval ranks all 5 tables but the whole schema fits in the token budget, so retrieval does not filter anything on this database. It matters for wide schemas.
- The few-shot exemplars are written for the demo schema and need to be replaced for another database.
- The result summary is a one-sentence LLM paraphrase of metadata and the first row, not an analysis.
- No authentication, rate limiting, or conversation memory; each question is independent.
- The Docker path (`make db-up`) and the real Gemini and OpenAI paths were not run in the development environment (no Docker, no API key). The same PostgreSQL steps were run via apt on Ubuntu 24.04 and `scripts/setup_local_postgres.sh` was re-run against an existing install.
- Ideas: execution-based self-consistency, per-column value sampling in the cards, a larger evaluation set, and streaming responses.

## Configuration

Copy `.env.example` to `.env` and adjust. Every variable is optional — with none set, the engine runs fully offline on fakes and fixtures.

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | *(unset → fixtures)* | PostgreSQL DSN for introspection and execution |
| `LLM_PROVIDER` | `auto` | `auto` (Gemini, then Groq, then OpenAI, by which key is set; else fake), `gemini`, `groq`, or `openai` |
| `GEMINI_API_KEY` | *(unset)* | Enables the Gemini provider and embedder (free tier available) |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | Chat model for generation, correction, summaries |
| `GEMINI_EMBEDDING_MODEL` | `gemini-embedding-2` | Embeddings for schema-card and few-shot retrieval |
| `GEMINI_API` | `native` | `openai_compat` = call Gemini through its OpenAI-compatible endpoint with the key as `Authorization: Bearer` (use when the key is injected that way, e.g. by a proxy); also enables `GENERATION_MODE=tools` on Gemini |
| `GROQ_API_KEY` | *(unset)* | Enables the Groq provider (`LLM_PROVIDER=groq`); retrieval uses the local lexical embedder because Groq has no embeddings API |
| `GROQ_MODEL` | `qwen/qwen3.8-27b` | Chat model for generation, correction, summaries |
| `GROQ_REASONING_EFFORT` | *(unset)* | `low`/`medium`/`high` for reasoning models such as `openai/gpt-oss-20b` |
| `LLM_FALLBACK` | `auto` | With Gemini selected and `GROQ_API_KEY` set, a Gemini failure (e.g. free-tier quota) automatically fails over to Groq (and lexical retrieval); `none` disables. Evaluation reports how many calls used the fallback, since those results are mixed-provider |
| `OPENAI_API_KEY` | *(unset)* | Enables the OpenAI provider and embedder |
| `OPENAI_MODEL` | `gpt-4o-mini` | Chat model for generation, correction, summaries |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embeddings for schema-card and few-shot retrieval |
| `DEFAULT_ROW_LIMIT` | `200` | `LIMIT` injected when the model forgot one |
| `MAX_CORRECTION_ROUNDS` | `2` | Retry budget of the self-correction loop |
| `STATEMENT_TIMEOUT_MS` | `5000` | Server-enforced per-statement timeout |
| `SCHEMA_TOKEN_BUDGET` | `2000` | Approximate token cap for schema cards in the prompt |
| `SAMPLE_ROWS_PER_TABLE` | `3` | Real rows per non-sensitive table shown in schema cards (`0` disables; sensitive tables are never sampled) |
| `PROMPT_MODE` | `schema_aware` | `baseline` = bare table names; `columns_only` = table + column names; both without exemplars (ablation only) |
| `GENERATION_MODE` | `text` | `tools` = LLM function calling (`describe_table`, `submit_sql`, `cannot_answer`); Groq/OpenAI only |
| `SENSITIVE_TABLES` | `customers` | Comma-separated tables where `SELECT *` is denied |
| `HISTORY_DB_PATH` | `artifacts/history.sqlite` | SQLite file for query history |

## API & CLI reference

| Endpoint | Description |
|---|---|
| `POST /query` | `{"question": "..."}` → validated SQL, rows, columns, summary, correction rounds, latency |
| `GET /history?limit=20` | Most recent queries (question, SQL, status, latency), newest first |
| `POST /schema/refresh` | Invalidate the schema cache; next query re-introspects |
| `GET /healthz` | Liveness + which backends are wired (`fake`/`openai`, `fixture`/`postgres`) |

| Make target | Description |
|---|---|
| `make install` | Create `.venv` and install with dev extras |
| `make lint` / `make type` / `make test` | ruff / mypy (strict) / pytest — the CI gates |
| `make check` | All three gates |
| `make demo` | Offline-first demo (uses live Postgres when reachable) |
| `make run` | Serve the API on port 8000 |
| `make db-up` / `make db-down` | Start/stop the seeded PostgreSQL container |

## What the demo shows

Real output from `make demo` (offline mode). A `SELECT *` over the PII-flagged table is refused — the model "insists" three times and the guardrail wins every time:

```
Q: Show me everything about our customers.
   status=rejected attempts=3 latency=0.8ms
   SQL: SELECT * FROM customers
   [guardrail] attempt 1: [no-star-sensitive] SELECT * is not allowed over sensitive table(s): customers
   [guardrail] attempt 2: [no-star-sensitive] SELECT * is not allowed over sensitive table(s): customers
   [guardrail] attempt 3: [no-star-sensitive] SELECT * is not allowed over sensitive table(s): customers
```

And the correction loop recovering from an error only the database can see — the first attempt references every table and column correctly, so static checks pass, but it violates GROUP BY semantics:

```
Self-correction showcase — attempt 1 fails the EXPLAIN preflight:
Q: What is the total spend per country?
   status=ok attempts=2 latency=2.0ms
   SQL: SELECT customers.country AS country, SUM(orders.total_cents) AS spent_cents FROM orders AS orders JOIN customers AS customers ON customers.id = orders.customer_id GROUP BY customers.country ORDER BY spent_cents DESC LIMIT 200
   [explain] attempt 1: column "customers.full_name" must appear in the GROUP BY clause or be used in an aggregate function
   summary: The query returned 5 rows.
  country | spent_cents
  ---------------------
  PH      | 768000
  CA      | 377300
  US      | 347400
  GB      | 255000
  DE      | 127100
```

## Implementation highlights

### SQL safety is an AST problem, not a string problem

Regex-based SQL filters share a fatal flaw: they reason about *text* while the database reasons about *structure*. A blocklist that catches `DELETE` misses `WITH gone AS (DELETE FROM refunds RETURNING id) SELECT id FROM gone` — a legal Postgres statement whose root is a `SELECT`. Comments, string literals containing semicolons, and dialect quirks generate endless false positives and, worse, false negatives. This engine parses each candidate exactly once and applies every rule to the tree:

```python
class GuardrailPipeline:
    """Parses once, then folds the AST through every rule in order."""

    def validate(self, sql: str) -> str:
        """Return safe, rewritten SQL — or raise GuardrailViolation."""
        expression = parse_single_statement(sql, self._ctx.dialect)
        for rule in self._rules:
            expression = rule.apply(expression, self._ctx)
        return expression.sql(dialect=self._ctx.dialect, comments=False)
```

Validation and rewriting share one mechanism — a rule either raises or returns a (possibly transformed) tree, and the final SQL is *rendered from the validated AST*, never echoed from model output. The select-only rule shows why tree-walking matters; a root-type check alone would leave three real write paths open:

```python
class SelectOnlyRule:
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
```

`_FORBIDDEN_NODES` covers DML, DDL, transaction control, `COPY`, `SELECT ... INTO` (a write dressed as a read), and `FOR UPDATE` locks. Multi-statement payloads die even earlier — `sqlglot.parse` returns a list, and any length other than one is a rejection before the first rule runs.

### Defense in depth: the layers underneath the prompt

The system prompt asks the model to behave; nothing downstream depends on it. Each layer assumes the ones above it have failed:

1. **AST guardrails** reject non-SELECTs, unknown tables, and sensitive stars, and rewrite the rest (qualified columns, injected `LIMIT`, stripped comments).
2. **EXPLAIN preflight** sends the validated SQL to the planner before execution — errors the static schema can't catch (grouping violations, type mismatches) surface here without reading a row.
3. **Read-only transaction + statement timeout** are enforced by the server, not the client:

```python
self._options = f"-c statement_timeout={int(statement_timeout_ms)}"
...
with psycopg.connect(
    self._dsn, options=self._options, connect_timeout=self._connect_timeout
) as conn:
    conn.read_only = True
```

Even a hypothetical guardrail bypass meets a transaction that cannot write and a query that cannot run past its budget. The timeout rides in on connection options, and because multi-statements were already rejected at parse time, the SQL text has no way to `SET` it back.

### The self-correction loop, and knowing when to stop

Retrying is only useful when the model can plausibly fix the problem. Every failure is classified by SQLSTATE (with message-pattern fallback), and the retry policy is an explicit, tested allowlist:

```python
# Categories the LLM has a realistic chance of fixing by rewriting the SQL.
# Timeouts, read-only violations, and permission errors are excluded on
# purpose: paraphrasing SQL rarely makes a slow query fast, and it should
# never be the mechanism that talks its way around a security boundary.
_RETRYABLE: frozenset[ErrorCategory] = frozenset(
    {
        ErrorCategory.SYNTAX,
        ErrorCategory.UNDEFINED_TABLE,
        ErrorCategory.UNDEFINED_COLUMN,
        ErrorCategory.AMBIGUOUS_COLUMN,
        ErrorCategory.GROUPING,
        ErrorCategory.UNKNOWN,
    }
)
```

The loop runs generate → validate → EXPLAIN → execute, and on a retryable failure sends back a minimal correction prompt: the question, the failing SQL, and the verbatim error — no fresh schema dump, no chatter. Two properties keep it honest: the budget counts *attempts*, not unique errors (a model that keeps producing the same `SELECT *` burns its attempts and stops), and non-retryable failures return immediately with the full round-by-round record (`stage`, `sql`, `error`, `category`) attached to the API response. The whole state machine is unit-tested with a scripted `FakeLLM` — fail-then-succeed at each stage, budget exhaustion, zero-budget, and stop-on-policy paths.

### Token-budgeted schema cards for wide databases

Dumping full DDL into the prompt works until the schema has 300 tables. Instead, each table is rendered as a dense card (name, comment, columns with types, PK/FK edges, `[sensitive]` flag), the cards are ranked by embedding similarity against the question, and a greedy pass packs the most relevant ones into a fixed budget:

```python
ranked = sorted(
    zip(cards, card_vectors, strict=True),
    key=lambda pair: cosine_similarity(question_vector, pair[1]),
    reverse=True,
)
selected: list[SchemaCard] = []
used = 0
for card, _vector in ranked:
    if selected and used + card.tokens > token_budget:
        break
    selected.append(card)
    used += card.tokens
```

Selection stops at the first card that no longer fits, keeping the result a relevance-ordered prefix — deterministic, no knapsack surprises — and the top card is always included even if it alone exceeds the budget. Token counts use a ~4-chars-per-token estimate on purpose: close enough for budgeting, and it avoids shipping a tokenizer (and its model-file download) into an offline test suite.

### Every external dependency has a deterministic fake

The LLM seam is one method:

```python
class LLMProvider(Protocol):
    def complete(self, prompt: str, *, system: str | None = None) -> str:
        """Return the model's text response for a single-turn prompt."""
        ...
```

Same pattern for `Embedder`, `QueryExecutor`, `SchemaProvider`, and `HistoryStore` — each protocol has a real implementation (Gemini/OpenAI, psycopg, `pg_catalog` introspection, SQLite) and a deterministic fake (`FakeLLM` with scripted responses and question-template matching, `FakeEmbedder` hashing words into a normalized vector, `FakeExecutor` with scriptable failure sequences, a fixture schema mirroring the docker-compose database). Wiring happens in exactly one place (`container.py`): no API key means fakes, no `DATABASE_URL` means fixtures — the rest of the codebase never reads the environment. The payoff: the *entire* behavior surface, including the correction loop's failure choreography and the demo itself, is exercised by fast, fully offline tests, and CI needs no secrets and no services.

## Project structure

```
text-to-sql-engine/
├── src/text_to_sql_engine/
│   ├── guardrails/        # AST rule chain: select-only, no-comments, allowlist,
│   │                      #   sensitive-star, qualify-columns, limit-inject
│   ├── schema/            # pg_catalog introspection, schema cards, cache, fixtures
│   ├── fewshot/           # exemplar store + 15 bundled (question, SQL) pairs
│   ├── llm/               # LLMProvider protocol, Gemini + OpenAI adapters, FakeLLM
│   ├── embeddings/        # Embedder protocol, Gemini + OpenAI adapters, FakeEmbedder
│   ├── execution/         # executors (Postgres/fake) + SQLSTATE classification
│   ├── correction.py      # the self-correction loop state machine
│   ├── pipeline.py        # end-to-end orchestration
│   ├── container.py       # the one place where real vs. fake is decided
│   ├── api.py             # FastAPI surface
│   └── demo.py            # offline-first demo
├── db/init/               # e-commerce schema + generated deterministic seed
├── scripts/               # seed generator (also prints the expected answers)
├── tests/                 # 141 offline tests
├── docker-compose.yml     # postgres:16 with schema + seed auto-applied
└── Makefile               # install / lint / type / test / demo / run / db-up
```

## Testing

```bash
make check   # ruff + mypy (strict) + pytest
```

141 tests run in under a second, entirely offline — no network, no database, no API keys, no model downloads. Coverage by area:

- **Guardrails** — every rule individually and through the pipeline: rejects UPDATE/DELETE/INSERT/DDL, DML inside CTEs, `SELECT ... INTO`, row locks, multi-statements, and unknown tables (including inside subqueries); strips comments; injects and preserves LIMITs; denies sensitive stars while expanding safe ones; fully qualifies columns and rejects unknown or ambiguous ones.
- **Correction loop** — scripted fail-then-succeed at the guardrail, EXPLAIN, and execute stages; budget exhaustion; zero-budget; immediate stop on non-retryable categories.
- **Error classification** — the SQLSTATE table, message-pattern fallback, precedence between them, and the retryability policy.
- **Schema cards** — card rendering (PK/FK/sensitive markers), similarity ranking, budget-packing edge cases, cache invalidation.
- **Retrieval** — few-shot top-k relevance, determinism, and the FakeEmbedder's similarity behavior.
- **API & demo** — request validation (`extra="forbid"`), rejection-as-data (refused queries are a 200 with `status=rejected`, not a 500), the history endpoint, and a full offline demo run asserted on its printed pipeline effects.

The live-database path (introspection SQL, read-only executor) is exercised manually via `make db-up && make demo`; its logic is deliberately thin, and the SQL it runs is identical to what the offline tests validate.

## Roadmap

- Column-level sensitivity: mask or drop flagged columns instead of rejecting the whole star.
- Gate execution on `EXPLAIN` cost output, not just success — refuse plans above a configurable cost ceiling.
- Persist corrected (question, SQL) pairs as new few-shot exemplars after human approval.
- Function denylist (e.g. `pg_sleep`, `pg_read_file`) as an additional guardrail rule.
- Multi-dialect support via sqlglot transpilation (DuckDB and SQLite backends for local analytics).
- Embedding cache + pgvector-backed retrieval for schemas with hundreds of tables.

## License

MIT — see [LICENSE](LICENSE).
