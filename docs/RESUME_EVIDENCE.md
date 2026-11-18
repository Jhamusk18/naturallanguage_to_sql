# Resume bullets: evidence map

Every number below is read from a committed result file. Reproduce with:

```bash
make db-local
LLM_PROVIDER=groq GROQ_MODEL=openai/gpt-oss-20b GROQ_REASONING_EFFORT=low LLM_FALLBACK=none \
  make ablation RESULTS=evaluation/results/v2-groq-gpt-oss-20b
python -m evaluation.compare evaluation/results/v2-groq-gpt-oss-20b/*.json
```

Setup: Groq free tier, temperature 0, PostgreSQL seed in `db/init/`, 67 questions in
`evaluation/questions.jsonl` (54 answerable with gold SQL, 6 unsafe, 4 nonexistent-column,
3 ambiguous). Same model, same day and same questions in every arm; only the prompt changes.

## Definitions

- **Query success (execution success):** the final SQL passed every guardrail and ran on
  PostgreSQL without error. Says nothing about correctness.
- **Accuracy:** the returned rows equal the gold SQL's rows (order-insensitive, column names ignored).
- **Hallucination:** the first-attempt SQL references a table or column not in the schema
  (static sqlglot check, `evaluation/hallucination.py`).
- **Correction OFF vs ON:** within one run with up to 2 retries, OFF = the first attempt,
  ON = the final result. At temperature 0 the first attempt is what a no-correction run produces.

## Bullet 1: "Built a natural-language-to-SQL system using an LLM to generate, validate and execute queries on 50+ analytical tasks"

**Supported.**
- Generate: `prompts.py` `PromptBuilder.initial`, `llm/groq_provider.py`, `llm/gemini_provider.py`.
- Validate: `guardrails/rules.py` (sqlglot AST rules), `guardrails/intent.py`, EXPLAIN preflight in `correction.py`.
- Execute: `execution/executor.py` (read-only transaction, statement timeout).
- 50+ tasks: 54 answerable questions with gold SQL; oracle check 54/54.

Function calling: implemented as an optional mode (`GENERATION_MODE=tools`, `src/text_to_sql_engine/tool_calling.py`):
the model calls `describe_table`, `submit_sql` and `cannot_answer`; failed submissions return as tool
results with the error and catalog facts. Tested offline (`tests/test_tool_calling.py`) and live on Groq.
Its full 54-question evaluation was cut off by the free daily token cap (13 questions scored, file marked
incomplete), so no tool-mode metric is claimed. The default mode is text generation.

## Bullet 2: "schema-aware prompting with metadata and sample rows, reducing hallucinated tables/columns by 40%"

Implemented: `schema/introspect.py` (types, PK/FK, comments, categorical values, PK-ordered diverse
sample rows, sensitive tables never sampled), `schema/cards.py` (`build_card`, `select_cards`),
`fewshot/store.py`.

Measured:

| Model | Names-only prompt | Schema-aware prompt | Relative reduction |
|---|---|---|---|
| gpt-oss-20b (v2) | 17/54 hallucinated | 0/54 | 100% |
| qwen3.8-27b (v1) | 34/54 | 0/54 | 100% |

- Accuracy: column names only 37/54 (68.5%) → full schema-aware 45/54 (83.3%) strict;
  41/54 (75.9%) → 48/54 (88.9%) column-tolerant (`evaluation/rescore.py`: extra columns allowed).
- Sample rows alone: 0 vs 3 rows gave the same result (0/54 hallucinations, 45/54 accurate) on this
  5-table schema. Their effect is not separately measurable here.

**The 40% figure is not what was measured.** The measured reduction is 100% (to zero). Accurate wording:
"…eliminating hallucinated tables/columns (17/54 → 0/54 first-try queries vs. a names-only prompt)
and raising answer accuracy from 69% to 83%."

## Bullet 3: "self-correction loop feeding execution errors back to the LLM, lifting query success from 72% to 90%"

Implemented: `correction.py` `CorrectionLoop.run` (guardrail → EXPLAIN → execute, max 2 retries,
non-retryable errors stop), `catalog_hint` (real tables/columns of the failed query),
`execution/errors.py` (error classification).

Measured (v2, gpt-oss-20b, paired OFF → ON):

| Prompt | Query success OFF → ON | Failed first attempts fixed |
|---|---|---|
| names only | 33.3% → 72.2% | 21/22 |
| column names | 98.1% → 100% | 1/1 |
| schema-aware | 98.1% → 100% | 1/1 |

Before the catalog-grounded correction prompt (v1, qwen): names-only 25.9% → 42.6%, 9/40 fixed.

**72% → 90% was not observed in any condition.** No prompt left first-try success near 72%:
with column information the model succeeds 98% of the time, and without it 33%. Accurate wording:
"…fixing 95% of failed queries (21/22) and lifting query success from 33% to 72% under minimal
schema context; with schema-aware prompting, 98% → 100%."

## Also measured

- Unsafe requests refused: 6/6 in every v2 arm; table row counts unchanged in every run.
- Nonexistent-column questions declined instead of answered: 3/4 (schema-aware); ambiguous: 3/3.
- Streamlit UI exercised in a browser against live Groq: `docs/screenshots/`.

## Second model: Gemini gemini-3.5-flash-lite (v3)

Same code, questions and four arms as v2, run on 2026-11-18 with Gemini `gemini-3.5-flash-lite`
(OpenAI-compatible endpoint, temperature 0, `LLM_FALLBACK=none`). Files:
`evaluation/results/v3-gemini-3.5-flash-lite/`. All four are complete (`incomplete: false`) with 0 fallback
calls. This is a separate experiment; its numbers are not pooled or compared row-by-row with the Groq runs.

```bash
LLM_PROVIDER=gemini GEMINI_API=openai_compat LLM_FALLBACK=none \
  make ablation RESULTS=evaluation/results/v3-gemini-3.5-flash-lite
python -m evaluation.rescore evaluation/results/v3-gemini-3.5-flash-lite/*.json
python -m evaluation.compare evaluation/results/v3-gemini-3.5-flash-lite/*.json
```

| Prompt | First-try hallucination | Query success OFF → ON | Strict accuracy OFF → ON | Column-tolerant OFF → ON | Failed first attempts fixed |
|---|---|---|---|---|---|
| names only | 10/54 | 7/54 → 17/54 | 5/54 → 12/54 | 5/54 → 12/54 | 10/10 |
| column names | 0/54 | 49/54 → 53/54 | 36/54 → 37/54 | 39/54 → 40/54 | 4/4 |
| schema-aware, no sample rows | 0/54 | 50/54 → 53/54 | 43/54 → 44/54 | 45/54 → 47/54 | 3/3 |
| schema-aware + sample rows | 0/54 | 51/54 → 54/54 | 42/54 → 43/54 | 44/54 → 46/54 | 3/3 |

| Prompt | Unsafe refused | Nonexistent not executed | Ambiguous not executed | Answerable falsely declined |
|---|---|---|---|---|
| names only | 6/6 | 4/4 | 3/3 | 37/54 |
| column names | 6/6 | 4/4 | 3/3 | 1/54 |
| schema-aware, no sample rows | 6/6 | 4/4 | 3/3 | 1/54 |
| schema-aware + sample rows | 6/6 | 4/4 | 3/3 | 0/54 |

Data unchanged in every arm.

Reading: hallucinations fall 10/54 → 0/54 (100% relative reduction, as on both Groq models). With names only,
Gemini mostly declines rather than guessing (37/54 false refusals), which is why its names-only query success
(13.0% → 31.5%) is lower than gpt-oss-20b's. Sample rows again made no measurable difference (43/54 vs 44/54
final strict accuracy, within run-to-run noise). Neither the 40% nor the 72% → 90% figure appears here either.

## Limits

One small schema (5 tables), 54 questions, single runs per arm (runs vary slightly even at
temperature 0), lexical rather than embedding retrieval on Groq.
