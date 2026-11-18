"""Builds the object graph from Settings.

Every external dependency is selected here, and only here: no API key means
FakeLLM + FakeEmbedder, no DATABASE_URL means fixture schema + FakeExecutor.
The rest of the codebase never checks the environment.
"""

from __future__ import annotations

from dataclasses import dataclass

from text_to_sql_engine.config import Settings
from text_to_sql_engine.demo_data import DEMO_SQL_TEMPLATES, FAKE_LLM_NO_TEMPLATE_SQL
from text_to_sql_engine.embeddings.base import Embedder
from text_to_sql_engine.embeddings.cached import CachedEmbedder
from text_to_sql_engine.embeddings.fake import FakeEmbedder
from text_to_sql_engine.embeddings.fallback import FallbackEmbedder
from text_to_sql_engine.embeddings.gemini_embedder import GeminiEmbedder
from text_to_sql_engine.embeddings.openai_embedder import OpenAIEmbedder
from text_to_sql_engine.execution.executor import FakeExecutor, PostgresExecutor, QueryExecutor
from text_to_sql_engine.fewshot.exemplars import DEMO_EXEMPLARS
from text_to_sql_engine.fewshot.store import FewShotStore
from text_to_sql_engine.history import HistoryStore, SqliteHistoryStore
from text_to_sql_engine.llm.base import LLMProvider
from text_to_sql_engine.llm.fake import FakeLLM
from text_to_sql_engine.llm.fallback import FallbackLLM
from text_to_sql_engine.llm.gemini_provider import GeminiProvider
from text_to_sql_engine.llm.groq_provider import GroqProvider
from text_to_sql_engine.llm.openai_provider import OpenAIProvider
from text_to_sql_engine.pipeline import TextToSqlPipeline
from text_to_sql_engine.schema.cache import SchemaCache, SchemaProvider, StaticSchemaProvider
from text_to_sql_engine.schema.fixtures import DEMO_SCHEMA
from text_to_sql_engine.schema.introspect import PostgresSchemaProvider

GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


@dataclass
class Components:
    """The wired application, plus mode labels for /healthz."""

    pipeline: TextToSqlPipeline
    schema_cache: SchemaCache
    history: HistoryStore
    llm_mode: str
    db_mode: str
    llm_model: str | None = None


def build_components(
    settings: Settings | None = None,
    *,
    llm: LLMProvider | None = None,
    executor: QueryExecutor | None = None,
    history: HistoryStore | None = None,
) -> Components:
    """Wire the pipeline. ``llm``/``executor``/``history`` overrides exist for
    the offline demo and tests; everything else follows the settings."""
    settings = settings or Settings()
    sensitive = settings.sensitive_table_set()

    embedder: Embedder
    llm_mode = settings.active_provider
    if llm_mode == "fake":
        if llm is None:
            llm = FakeLLM(
                sql_templates=DEMO_SQL_TEMPLATES, default=FAKE_LLM_NO_TEMPLATE_SQL
            )
        embedder = FakeEmbedder()
    elif llm_mode == "gemini":
        key = settings.gemini_api_key or ""
        gemini_embedder: Embedder
        if settings.gemini_api == "openai_compat":
            gemini_embedder = CachedEmbedder(
                OpenAIEmbedder(key, model=settings.gemini_embedding_model,
                               base_url=GEMINI_OPENAI_BASE_URL)
            )
        else:
            gemini_embedder = CachedEmbedder(
                GeminiEmbedder(key, model=settings.gemini_embedding_model)
            )
        embedder = gemini_embedder
        if llm is None:
            if settings.gemini_api == "openai_compat":
                llm = OpenAIProvider(key, model=settings.gemini_model,
                                     base_url=GEMINI_OPENAI_BASE_URL)
            else:
                llm = GeminiProvider(key, model=settings.gemini_model)
            if settings.llm_fallback == "auto" and settings.groq_api_key:
                llm = FallbackLLM(
                    llm,
                    GroqProvider(
                        settings.groq_api_key,
                        model=settings.groq_model,
                        reasoning_effort=settings.groq_reasoning_effort,
                    ),
                    primary_name=f"gemini:{settings.gemini_model}",
                    fallback_name=f"groq:{settings.groq_model}",
                )
                embedder = FallbackEmbedder(
                    gemini_embedder, CachedEmbedder(FakeEmbedder()), "gemini", "lexical"
                )
    elif llm_mode == "groq":
        if llm is None:
            llm = GroqProvider(
                settings.groq_api_key or "",
                model=settings.groq_model,
                reasoning_effort=settings.groq_reasoning_effort,
            )
        embedder = CachedEmbedder(FakeEmbedder())  # Groq has no embeddings API
    else:
        key = settings.openai_api_key or ""
        if llm is None:
            llm = OpenAIProvider(key, model=settings.openai_model)
        embedder = CachedEmbedder(OpenAIEmbedder(key, model=settings.openai_embedding_model))

    provider: SchemaProvider
    if settings.offline_db:
        provider = StaticSchemaProvider(DEMO_SCHEMA.with_sensitive(sensitive))
        if executor is None:
            executor = FakeExecutor()
        db_mode = "fixture"
    else:
        dsn = settings.database_url or ""
        provider = PostgresSchemaProvider(
            dsn, sensitive_tables=sensitive, sample_rows=settings.sample_rows_per_table
        )
        if executor is None:
            executor = PostgresExecutor(dsn, statement_timeout_ms=settings.statement_timeout_ms)
        db_mode = "postgres"

    schema_cache = SchemaCache(provider)
    history = history or SqliteHistoryStore(settings.history_db_path)
    pipeline = TextToSqlPipeline(
        schema_cache=schema_cache,
        embedder=embedder,
        fewshot=FewShotStore(DEMO_EXEMPLARS, embedder),
        llm=llm,
        executor=executor,
        history=history,
        token_budget=settings.schema_token_budget,
        default_limit=settings.default_row_limit,
        max_corrections=settings.max_correction_rounds,
        prompt_mode=settings.prompt_mode,
        summarize=settings.summarize_results,
        generation_mode=settings.generation_mode,
    )
    return Components(
        pipeline=pipeline,
        schema_cache=schema_cache,
        history=history,
        llm_mode=llm_mode,
        db_mode=db_mode,
        llm_model=settings.llm_model_name,
    )
