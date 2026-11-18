"""Environment-driven settings (pydantic-settings, reads `.env` when present)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime knobs. Every field maps 1:1 to an env var (see .env.example)."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str | None = None
    llm_provider: Literal["auto", "gemini", "groq", "openai"] = "auto"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.5-flash-lite"
    gemini_embedding_model: str = "gemini-embedding-2"
    # "native": google-genai SDK, key in the x-goog-api-key header.
    # "openai_compat": Gemini's OpenAI-compatible endpoint, key as Authorization: Bearer
    # (what a proxy-injected bearer credential provides); also enables tool calling.
    gemini_api: Literal["native", "openai_compat"] = "native"
    llm_fallback: Literal["auto", "none"] = "auto"
    groq_api_key: str | None = None
    groq_model: str = "qwen/qwen3.8-27b"
    # Only for reasoning models (e.g. openai/gpt-oss-20b): low | medium | high.
    groq_reasoning_effort: str | None = None
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    default_row_limit: int = Field(default=200, ge=1)
    max_correction_rounds: int = Field(default=2, ge=0)
    statement_timeout_ms: int = Field(default=5000, ge=100)
    schema_token_budget: int = Field(default=2000, ge=100)
    sample_rows_per_table: int = Field(default=3, ge=0, le=10)
    prompt_mode: Literal["schema_aware", "columns_only", "baseline"] = "schema_aware"
    # "text": SQL parsed from a text reply; "tools": LLM function calling
    # (describe_table / submit_sql / cannot_answer). Tools need Groq or OpenAI.
    generation_mode: Literal["text", "tools"] = "text"
    summarize_results: bool = True
    sensitive_tables: str = "customers"
    history_db_path: str = "artifacts/history.sqlite"

    def sensitive_table_set(self) -> set[str]:
        return {name.strip() for name in self.sensitive_tables.split(",") if name.strip()}

    @property
    def active_provider(
        self,
    ) -> Literal["gemini", "groq", "openai", "fake"]:
        """Which LLM backend is used. ``auto`` prefers Gemini, Groq, OpenAI, then the fake.

        An explicit ``LLM_PROVIDER`` without its API key falls back to the fake
        rather than silently using the other provider.
        """
        if self.llm_provider == "gemini":
            return "gemini" if self.gemini_api_key else "fake"
        if self.llm_provider == "groq":
            return "groq" if self.groq_api_key else "fake"
        if self.llm_provider == "openai":
            return "openai" if self.openai_api_key else "fake"
        if self.gemini_api_key:
            return "gemini"
        if self.groq_api_key:
            return "groq"
        return "openai" if self.openai_api_key else "fake"

    @property
    def llm_model_name(self) -> str | None:
        models = {
            "gemini": self.gemini_model,
            "groq": self.groq_model,
            "openai": self.openai_model,
        }
        return models.get(self.active_provider)

    @property
    def offline_llm(self) -> bool:
        return self.active_provider == "fake"

    @property
    def offline_db(self) -> bool:
        return not self.database_url
