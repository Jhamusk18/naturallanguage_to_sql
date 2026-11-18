"""FastAPI surface: /query, /history, /schema/refresh, /healthz."""

from __future__ import annotations

from fastapi import FastAPI, Query
from pydantic import Field

from text_to_sql_engine import __version__
from text_to_sql_engine.config import Settings
from text_to_sql_engine.container import build_components
from text_to_sql_engine.models import HistoryEntry, QueryResult, StrictModel


class QueryRequest(StrictModel):
    """A natural-language question about the connected database."""

    question: str = Field(min_length=3, max_length=500)


def create_app(settings: Settings | None = None) -> FastAPI:
    components = build_components(settings)
    app = FastAPI(
        title="text-to-sql-engine",
        version=__version__,
        description=(
            "Natural language to safe, validated SQL: schema-aware prompting, "
            "AST guardrails, and a self-correction loop."
        ),
    )

    @app.post("/query", response_model=QueryResult)
    def query(request: QueryRequest) -> QueryResult:
        """Answer a question: generate, validate, execute, summarize."""
        return components.pipeline.answer(request.question)

    @app.get("/history", response_model=list[HistoryEntry])
    def history(limit: int = Query(default=20, ge=1, le=200)) -> list[HistoryEntry]:
        """Most recent queries, newest first."""
        return components.history.recent(limit)

    @app.post("/schema/refresh", status_code=204)
    def refresh_schema() -> None:
        """Invalidate the schema cache; the next query re-introspects."""
        components.schema_cache.invalidate()

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        """Liveness plus which backends (real or fake) are wired in."""
        info = {"status": "ok", "llm": components.llm_mode, "database": components.db_mode}
        if components.llm_model:
            info["model"] = components.llm_model
        return info

    return app


app = create_app()
