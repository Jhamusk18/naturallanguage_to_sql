"""Streamlit front end for the text-to-sql-engine.

Two ways to run it:

* Local, with the FastAPI backend: `make run`, then `make ui` (T2S_API_URL points at it).
* Embedded (Streamlit Community Cloud): set T2S_EMBEDDED=1 or leave T2S_API_URL unset on
  a host with no backend; the pipeline then runs inside this Streamlit process. Keys and
  DATABASE_URL come from Streamlit secrets (`.streamlit/secrets.toml` / the app's
  Secrets panel) or the environment.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import streamlit as st

# Streamlit Cloud installs requirements.txt but not this package; make src importable.
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_SECRET_NAMES = (
    "DATABASE_URL", "LLM_PROVIDER", "LLM_FALLBACK", "GROQ_API_KEY", "GROQ_MODEL",
    "GROQ_REASONING_EFFORT", "GEMINI_API_KEY", "GEMINI_API", "GEMINI_MODEL",
    "GEMINI_EMBEDDING_MODEL", "OPENAI_API_KEY", "GENERATION_MODE", "T2S_EMBEDDED",
)


def _load_streamlit_secrets() -> None:
    """Copy Streamlit secrets into the environment, where Settings reads them."""
    try:
        secrets = dict(st.secrets)
    except Exception:  # no secrets.toml: plain environment variables only
        return
    for name in _SECRET_NAMES:
        if name in secrets and name not in os.environ:
            os.environ[name] = str(secrets[name])


_load_streamlit_secrets()
API_URL = os.environ.get("T2S_API_URL", "http://127.0.0.1:8000")
_EXPLICIT_API = "T2S_API_URL" in os.environ
_FORCE_EMBEDDED = os.environ.get("T2S_EMBEDDED", "").lower() in {"1", "true", "yes"}


def _backend_reachable() -> bool:
    try:
        return httpx.get(f"{API_URL}/healthz", timeout=1).status_code == 200
    except httpx.HTTPError:
        return False


@st.cache_resource
def use_embedded() -> bool:
    """Embedded when forced, or when no backend address was given and none is running.

    Streamlit Cloud has no FastAPI server, so the app must run the pipeline itself there
    even if the T2S_EMBEDDED secret was not applied.
    """
    if _FORCE_EMBEDDED:
        return True
    if _EXPLICIT_API:
        return False
    return not _backend_reachable()

FAKE_MODE_QUESTIONS = [
    "Which 5 products have generated the most revenue?",
    "Who are our top 3 customers by total spend?",
    "How many orders are in each status?",
    "What is the average order value by country?",
    "Show the 10 most recent refunds with their order status.",
]
LIVE_MODE_QUESTIONS = [
    "Who are our top 3 customers by total spend?",
    "Which 5 products generated the highest revenue?",
    "What is the total revenue by month?",
    "Which customers have placed more than 3 orders?",
    "Which products have never been ordered?",
]
GUARDRAIL_QUESTION = "Show me everything about our customers."

STATUS_STYLE = {
    "ok": "success", "rejected": "warning", "error": "error", "unanswerable": "warning"
}

st.set_page_config(page_title="Natural Language-to-SQL Analytics System", layout="wide")


@st.cache_resource(show_spinner="Loading schema and model...")
def _components() -> Any:
    from text_to_sql_engine.config import Settings
    from text_to_sql_engine.container import build_components

    settings = Settings()
    settings = settings.model_copy(update={"history_db_path": "/tmp/t2s-history.sqlite"})
    return build_components(settings)


def _embedded_health() -> dict[str, str]:
    components = _components()
    return {
        "status": "ok",
        "llm": components.llm_mode,
        "database": components.db_mode,
        "model": components.llm_model or "fake",
    }


def fetch_health() -> dict[str, str] | None:
    if use_embedded():
        return _embedded_health()
    try:
        response = httpx.get(f"{API_URL}/healthz", timeout=3)
        response.raise_for_status()
        return dict(response.json())
    except httpx.HTTPError:
        return None


def run_query(question: str) -> dict[str, Any]:
    if use_embedded():
        from text_to_sql_engine.models import QueryResult, QueryStatus

        try:
            result = _components().pipeline.answer(question)
        except Exception as error:  # provider quota or outage: show it, do not crash the page
            message = f"The AI provider did not answer ({type(error).__name__}). "
            message += "Free-tier limits reset within minutes; try again shortly."
            result = QueryResult(question=question, sql=None, status=QueryStatus.ERROR,
                                 error=message)
        return dict(result.model_dump(mode="json"))
    response = httpx.post(f"{API_URL}/query", json={"question": question}, timeout=120)
    response.raise_for_status()
    return dict(response.json())


def render_mode_banner(health: dict[str, str]) -> bool:
    """Show which backends are real. Returns True when the LLM is a fake."""
    fake_llm = health["llm"] == "fake"
    database = "PostgreSQL" if health["database"] == "postgres" else "offline fixture rows"
    if fake_llm:
        st.warning(
            "**Demo mode: FakeLLM.** SQL comes from a small table of predefined templates, "
            "so only the bundled questions below work. This is not general-purpose "
            "NL-to-SQL. Set `GEMINI_API_KEY` (or `OPENAI_API_KEY`) and restart the backend "
            "to use a real model. "
            f"Database: {database}."
        )
    else:
        names = {"gemini": "Google Gemini", "groq": "Groq", "openai": "OpenAI"}
        provider = names.get(health["llm"], health["llm"])
        model = health.get("model", "unknown model")
        st.success(f"**Live mode:** {provider} (`{model}`) generates SQL. Database: {database}.")
    return fake_llm


def render_result(result: dict[str, Any]) -> None:
    status = result["status"]
    label = f"Status: {status.upper()}"
    getattr(st, STATUS_STYLE.get(status, "info"))(label)

    cols = st.columns(4)
    cols[0].metric("Rows", result["row_count"])
    cols[1].metric("Attempts", result["attempts"])
    cols[2].metric("Corrections", len(result["corrections"]))
    cols[3].metric("Latency", f"{result['latency_ms']:.0f} ms")

    if result.get("retrieved_tables"):
        st.caption(
            "Schema retrieval (tables ranked by relevance, included in the prompt): "
            + ", ".join(f"`{name}`" for name in result["retrieved_tables"])
        )

    st.subheader("Generated SQL")
    if result["sql"]:
        st.code(result["sql"], language="sql")
    else:
        st.info("No SQL was produced.")

    if status == "ok":
        st.subheader("Explanation")
        st.write(result["summary"] or f"{result['row_count']} row(s) returned.")
        st.subheader("Results")
        if result["rows"]:
            st.dataframe(pd.DataFrame(result["rows"], columns=result["columns"]), width="stretch")
        else:
            st.info("The query ran successfully but returned no rows.")
    else:
        heading = {
            "rejected": "Rejected by validation",
            "unanswerable": "Not answerable from this schema",
        }.get(status, "Query failed")
        st.error(f"**{heading}:** {result['error']}")
        if "has_no_template" in (result["error"] or ""):
            st.info("FakeLLM has no template for this question. Use one of the example "
                    "questions, or configure `GEMINI_API_KEY` for free-form questions.")

    if result["corrections"]:
        with st.expander(f"Self-correction trace ({len(result['corrections'])} failed attempt(s))",
                         expanded=True):
            for round_ in result["corrections"]:
                st.markdown(f"**Attempt {round_['attempt']}** failed at stage "
                            f"`{round_['stage']}`")
                st.code(round_["sql"], language="sql")
                st.caption(f"Error: {round_['error']}")


st.title("Natural Language-to-SQL Analytics System")
st.caption("Ask a question in plain English. The system retrieves the relevant schema, generates "
           "SQL, validates it, executes it read-only, and self-corrects on failure.")

health = fetch_health()
if health is None:
    st.error(f"Cannot reach the backend at `{API_URL}`. Start it with `make run` "
             "(or set `T2S_API_URL`), then reload this page.")
    st.stop()

fake_llm = render_mode_banner(health)

if "question" not in st.session_state:
    st.session_state["question"] = ""

st.markdown("**Example questions**")
examples = FAKE_MODE_QUESTIONS if fake_llm else LIVE_MODE_QUESTIONS
for example in examples:
    if st.button(example, key=f"ex-{example}"):
        st.session_state["question"] = example
if fake_llm and st.button(f"{GUARDRAIL_QUESTION}  (guardrail rejection demo)", key="ex-guard"):
    st.session_state["question"] = GUARDRAIL_QUESTION

question = st.text_area("Your question", key="question", height=80,
                        placeholder="e.g. Which products have never been ordered?")
if st.button("Run Query", type="primary"):
    if len(question.strip()) < 3:
        st.error("Please enter a question (at least 3 characters).")
    else:
        try:
            with st.spinner("Running pipeline..."):
                outcome = run_query(question.strip())
        except httpx.HTTPStatusError as error:
            st.error(f"Backend returned HTTP {error.response.status_code}: {error.response.text}")
        except httpx.HTTPError as error:
            st.error(f"Could not reach the backend: {error}")
        else:
            render_result(outcome)

with st.expander("How the pipeline works"):
    st.markdown(
        """
**Question → Schema Retrieval → SQL Generation → Validation → Execution → Correction**

1. **Schema retrieval:** the schema is introspected (PostgreSQL `pg_catalog`) into compact
   per-table cards, ranked by embedding similarity to the question and packed into a token
   budget. The 3 most similar few-shot examples are added.
2. **SQL generation:** the LLM receives the cards, examples, and dialect rules.
3. **Validation:** the SQL is parsed with sqlglot and passed through AST rules: single
   SELECT only, table allowlist, comment stripping, LIMIT injection, sensitive-table `SELECT *`
   denial, column qualification.
4. **Execution:** `EXPLAIN` preflight, then a read-only transaction with a statement timeout.
5. **Correction:** on a fixable failure, the failing SQL and exact error go back to the LLM
   (up to 2 rounds). Policy errors (timeout, permission, read-only violation) stop immediately.
"""
    )
