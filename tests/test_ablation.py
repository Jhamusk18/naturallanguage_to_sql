"""Offline tests for the ablation machinery (samples, baseline prompt, hallucinations)."""

from __future__ import annotations

from pathlib import Path

from evaluation.hallucination import find_hallucinations
from evaluation.run import _first_attempt_sql

from text_to_sql_engine.config import Settings
from text_to_sql_engine.container import build_components
from text_to_sql_engine.llm.fake import FakeLLM
from text_to_sql_engine.models import CorrectionRound, QueryResult, QueryStatus, TableInfo
from text_to_sql_engine.schema.cards import (
    build_card,
    build_columns_only_card,
    build_name_only_card,
)
from text_to_sql_engine.schema.fixtures import DEMO_SCHEMA


def test_card_includes_sample_rows() -> None:
    table = TableInfo(
        name="products",
        columns=DEMO_SCHEMA.tables[1].columns,
        sample_rows=[{"id": "1", "name": "Mug"}],
    )
    assert "sample: id=1, name=Mug" in build_card(table).text


def test_card_lists_categorical_values() -> None:
    columns = [
        c.model_copy(update={"sample_values": ["books", "home"]}) if c.name == "category" else c
        for c in DEMO_SCHEMA.tables[1].columns
    ]
    table = TableInfo(name="products", columns=columns)
    assert "category text (values: books, home)" in build_card(table).text


def test_card_without_sample_rows_has_no_sample_lines() -> None:
    assert "sample:" not in build_card(DEMO_SCHEMA.tables[1]).text


def test_name_only_card_hides_columns() -> None:
    card = build_name_only_card(DEMO_SCHEMA.tables[1])
    assert card.text == "table products"


def test_columns_only_card_lists_names_without_types_or_samples() -> None:
    table = TableInfo(
        name="products",
        columns=DEMO_SCHEMA.tables[1].columns,
        sample_rows=[{"id": "1", "name": "Mug"}],
    )
    text = build_columns_only_card(table).text
    assert text.startswith("table products(") and "price_cents" in text
    assert "sample:" not in text and "integer" not in text.lower() and "PK" not in text


def test_columns_only_prompt_has_all_tables_and_no_exemplars(tmp_path: Path) -> None:
    llm = FakeLLM(script=["SELECT products.name FROM products"])
    settings = Settings(
        database_url=None, prompt_mode="columns_only", summarize_results=False,
        history_db_path=str(tmp_path / "h.sqlite"),
    )
    components = build_components(settings, llm=llm)
    components.pipeline.answer("List product names")
    prompt = llm.calls[0]
    assert "price_cents" in prompt and "Q:" not in prompt
    assert all(f"table {t.name}(" in prompt for t in DEMO_SCHEMA.tables)


def test_detects_unknown_table() -> None:
    found = find_hallucinations("SELECT * FROM shipments", DEMO_SCHEMA)
    assert found.tables == ["shipments"] and found.any


def test_detects_unknown_column() -> None:
    found = find_hallucinations("SELECT products.color FROM products", DEMO_SCHEMA)
    assert found.columns and "color" in found.columns[0]


def test_valid_sql_and_cte_are_clean() -> None:
    assert not find_hallucinations("SELECT name FROM products", DEMO_SCHEMA).any
    cte = "WITH t AS (SELECT id FROM orders) SELECT id FROM t"
    assert not find_hallucinations(cte, DEMO_SCHEMA).any


def test_unparseable_sql_is_not_a_hallucination() -> None:
    assert not find_hallucinations("SELEC FROM WHERE", DEMO_SCHEMA).any


def test_first_attempt_sql_prefers_the_failed_first_attempt() -> None:
    failed = QueryResult(
        question="q", sql="SELECT 2", status=QueryStatus.OK, attempts=2,
        corrections=[CorrectionRound(attempt=1, stage="explain", sql="SELECT 1", error="e")],
    )
    clean = QueryResult(question="q", sql="SELECT 3", status=QueryStatus.OK, attempts=1)
    assert _first_attempt_sql(failed) == "SELECT 1"
    assert _first_attempt_sql(clean) == "SELECT 3"


def test_baseline_prompt_has_names_only_and_no_exemplars(tmp_path: Path) -> None:
    llm = FakeLLM(script=["SELECT products.name FROM products"])
    settings = Settings(
        database_url=None, prompt_mode="baseline", summarize_results=False,
        history_db_path=str(tmp_path / "h.sqlite"),
    )
    components = build_components(settings, llm=llm)
    components.pipeline.answer("List product names")
    prompt = llm.calls[0]
    assert "table products" in prompt and "price_cents" not in prompt
    assert "Q:" not in prompt
    assert len(llm.calls) == 1  # summarization disabled


def test_schema_aware_prompt_has_columns_and_exemplars(tmp_path: Path) -> None:
    llm = FakeLLM(script=["SELECT products.name FROM products"])
    settings = Settings(database_url=None, history_db_path=str(tmp_path / "h.sqlite"))
    components = build_components(settings, llm=llm)
    components.pipeline.answer("List product names")
    assert "price_cents" in llm.calls[0] and "Q:" in llm.calls[0]


def test_cached_embedder_embeds_each_text_once() -> None:
    from text_to_sql_engine.embeddings.cached import CachedEmbedder

    class Counting:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def embed(self, texts):  # type: ignore[no-untyped-def]
            self.calls.append(list(texts))
            return [[float(len(t))] for t in texts]

    inner = Counting()
    cached = CachedEmbedder(inner)  # type: ignore[arg-type]
    assert cached.embed(["a", "bb", "a"]) == [[1.0], [2.0], [1.0]]
    assert cached.embed(["bb", "ccc"]) == [[2.0], [3.0]]
    assert inner.calls == [["a", "bb"], ["ccc"]]
