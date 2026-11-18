"""Core domain models shared across the engine.

Every model is strict (``extra="forbid"``) so unknown fields are rejected at
the boundary instead of silently dropped.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    """Base model that rejects unknown fields everywhere it is used."""

    model_config = ConfigDict(extra="forbid")


class ColumnInfo(StrictModel):
    """A single column as discovered by introspection."""

    name: str
    data_type: str
    nullable: bool = True
    primary_key: bool = False
    comment: str | None = None
    # Distinct values of a low-cardinality text column (non-sensitive tables only).
    sample_values: list[str] = Field(default_factory=list)


class ForeignKey(StrictModel):
    """A foreign-key edge from one table's column to another's."""

    column: str
    references_table: str
    references_column: str


class TableInfo(StrictModel):
    """A table, its columns, and its outgoing foreign keys."""

    name: str
    columns: list[ColumnInfo]
    foreign_keys: list[ForeignKey] = Field(default_factory=list)
    comment: str | None = None
    sensitive: bool = False
    sample_rows: list[dict[str, str]] = Field(default_factory=list)

    def column_names(self) -> list[str]:
        return [column.name for column in self.columns]


class DatabaseSchema(StrictModel):
    """The introspected shape of a database, used by prompts and guardrails."""

    tables: list[TableInfo]

    def table_names(self) -> set[str]:
        return {table.name for table in self.tables}

    def sensitive_tables(self) -> set[str]:
        return {table.name for table in self.tables if table.sensitive}

    def get_table(self, name: str) -> TableInfo | None:
        for table in self.tables:
            if table.name == name:
                return table
        return None

    def to_sqlglot_schema(self) -> dict[str, object]:
        """Schema mapping in the shape sqlglot's qualifier expects.

        The value type is ``object`` because that is what sqlglot's
        ``ensure_schema`` accepts (nested mappings of table -> column -> type).
        """
        return {
            table.name: {column.name: column.data_type for column in table.columns}
            for table in self.tables
        }

    def with_sensitive(self, names: set[str]) -> DatabaseSchema:
        """Copy of this schema with the given tables (also) flagged sensitive."""
        tables = [
            table.model_copy(update={"sensitive": table.sensitive or table.name in names})
            for table in self.tables
        ]
        return DatabaseSchema(tables=tables)


class Exemplar(StrictModel):
    """A (question, SQL) pair used for few-shot prompting."""

    question: str
    sql: str


class QueryStatus(StrEnum):
    OK = "ok"
    REJECTED = "rejected"
    ERROR = "error"
    UNANSWERABLE = "unanswerable"  # the model declined: schema cannot answer / too vague


class CorrectionRound(StrictModel):
    """One failed attempt inside the self-correction loop."""

    attempt: int
    stage: str  # "guardrail" | "explain" | "execute"
    sql: str
    error: str
    category: str | None = None


class QueryResult(StrictModel):
    """Final answer returned by the pipeline and the API."""

    question: str
    sql: str | None
    status: QueryStatus
    columns: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0
    summary: str | None = None
    retrieved_tables: list[str] = Field(default_factory=list)
    attempts: int = 1
    corrections: list[CorrectionRound] = Field(default_factory=list)
    latency_ms: float = 0.0
    error: str | None = None


class HistoryEntry(StrictModel):
    """A persisted record of one answered (or failed) question."""

    id: int
    question: str
    sql: str | None
    status: QueryStatus
    latency_ms: float
    created_at: str
