"""Compact, token-budgeted 'schema cards' for prompting.

A card is a few dense lines describing one table: name, comment, columns with
types, PK/FK markers, and a [sensitive] flag. For wide databases the full DDL
would blow the context window, so cards are ranked by embedding similarity to
the question and greedily packed into a token budget — the most relevant
tables go in first, and the most relevant card is always included even if it
alone exceeds the budget.
"""

from __future__ import annotations

from dataclasses import dataclass

from text_to_sql_engine.embeddings.base import Embedder, cosine_similarity
from text_to_sql_engine.models import DatabaseSchema, TableInfo


@dataclass(frozen=True)
class SchemaCard:
    """One table rendered as prompt-ready text, with its token estimate."""

    table: str
    text: str
    tokens: int


def estimate_tokens(text: str) -> int:
    """Cheap deterministic proxy: ~4 characters per token.

    Good enough for budgeting; avoids shipping a tokenizer (and its model
    file download) just to count tokens.
    """
    return max(1, len(text) // 4)


def build_card(table: TableInfo) -> SchemaCard:
    """Render one table as a compact card."""
    header = f"table {table.name}"
    if table.comment:
        header += f" -- {table.comment}"
    if table.sensitive:
        header += " [sensitive]"
    fk_by_column = {fk.column: fk for fk in table.foreign_keys}
    lines = [header]
    for column in table.columns:
        line = f"  {column.name} {column.data_type}"
        if column.primary_key:
            line += " PK"
        fk = fk_by_column.get(column.name)
        if fk is not None:
            line += f" -> {fk.references_table}.{fk.references_column}"
        if column.comment:
            line += f" -- {column.comment}"
        if column.sample_values:
            line += " (values: " + ", ".join(column.sample_values) + ")"
        lines.append(line)
    for row in table.sample_rows:
        lines.append("  sample: " + ", ".join(f"{name}={value}" for name, value in row.items()))
    text = "\n".join(lines)
    return SchemaCard(table=table.name, text=text, tokens=estimate_tokens(text))


def build_name_only_card(table: TableInfo) -> SchemaCard:
    """Baseline card: the table name and nothing else (no columns, types, keys, samples)."""
    text = f"table {table.name}"
    return SchemaCard(table=table.name, text=text, tokens=estimate_tokens(text))


def build_columns_only_card(table: TableInfo) -> SchemaCard:
    """Middle ablation arm: table and column names only.

    No types, keys, comments, sample rows, or [sensitive] flag. This is what a
    naive "paste the column list into the prompt" approach looks like, so it
    isolates what the richer metadata + sample rows + exemplars add on top.
    """
    text = f"table {table.name}({', '.join(column.name for column in table.columns)})"
    return SchemaCard(table=table.name, text=text, tokens=estimate_tokens(text))


def select_cards(
    question: str,
    schema: DatabaseSchema,
    embedder: Embedder,
    token_budget: int,
) -> list[SchemaCard]:
    """Rank cards by similarity to the question and pack them into the budget.

    Selection stops at the first card that no longer fits, keeping the result
    a relevance-ordered prefix (deterministic, no knapsack surprises). At
    least one card is always returned.
    """
    cards = [build_card(table) for table in schema.tables]
    if not cards:
        return []
    vectors = embedder.embed([question, *[card.text for card in cards]])
    question_vector, card_vectors = vectors[0], vectors[1:]
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
    return selected
