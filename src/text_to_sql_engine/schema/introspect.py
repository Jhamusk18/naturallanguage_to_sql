"""Live PostgreSQL introspection: tables, columns, types, PKs, FKs, comments."""

from __future__ import annotations

from typing import LiteralString

import psycopg
from psycopg import sql
from psycopg.rows import tuple_row

from text_to_sql_engine.models import ColumnInfo, DatabaseSchema, ForeignKey, TableInfo

_TEXT_TYPES = {"text", "character varying", "character"}
_MAX_CATEGORIES = 10

_TABLES_SQL: LiteralString = """
SELECT c.relname, obj_description(c.oid, 'pg_class')
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'public' AND c.relkind = 'r'
ORDER BY c.relname
"""

_COLUMNS_SQL: LiteralString = """
SELECT c.relname,
       a.attname,
       format_type(a.atttypid, a.atttypmod),
       NOT a.attnotnull,
       col_description(c.oid, a.attnum)
FROM pg_attribute a
JOIN pg_class c ON c.oid = a.attrelid
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'public' AND c.relkind = 'r' AND a.attnum > 0 AND NOT a.attisdropped
ORDER BY c.relname, a.attnum
"""

_PRIMARY_KEYS_SQL: LiteralString = """
SELECT c.relname, a.attname
FROM pg_index i
JOIN pg_class c ON c.oid = i.indrelid
JOIN pg_namespace n ON n.oid = c.relnamespace
JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum = ANY (i.indkey)
WHERE i.indisprimary AND n.nspname = 'public'
"""

_FOREIGN_KEYS_SQL: LiteralString = """
SELECT src.relname,
       src_col.attname,
       tgt.relname,
       tgt_col.attname
FROM pg_constraint con
JOIN pg_class src ON src.oid = con.conrelid
JOIN pg_namespace n ON n.oid = src.relnamespace
JOIN pg_class tgt ON tgt.oid = con.confrelid
JOIN unnest(con.conkey) WITH ORDINALITY AS sk(attnum, ord) ON TRUE
JOIN unnest(con.confkey) WITH ORDINALITY AS tk(attnum, ord) ON tk.ord = sk.ord
JOIN pg_attribute src_col ON src_col.attrelid = src.oid AND src_col.attnum = sk.attnum
JOIN pg_attribute tgt_col ON tgt_col.attrelid = tgt.oid AND tgt_col.attnum = tk.attnum
WHERE con.contype = 'f' AND n.nspname = 'public'
ORDER BY src.relname, sk.ord
"""


class PostgresSchemaProvider:
    """Reads the `public` schema of a live database into a DatabaseSchema.

    Sensitivity is policy, not metadata Postgres knows about, so it is
    supplied by configuration (SENSITIVE_TABLES) and stamped on here.
    """

    def __init__(
        self,
        dsn: str,
        sensitive_tables: set[str] | None = None,
        sample_rows: int = 0,
    ) -> None:
        self._dsn = dsn
        self._sensitive = sensitive_tables or set()
        self._sample_rows = sample_rows

    def load(self) -> DatabaseSchema:
        with psycopg.connect(self._dsn, connect_timeout=5) as conn:
            conn.read_only = True
            with conn.cursor(row_factory=tuple_row) as cur:
                cur.execute(_TABLES_SQL)
                table_rows = cur.fetchall()
                cur.execute(_COLUMNS_SQL)
                column_rows = cur.fetchall()
                cur.execute(_PRIMARY_KEYS_SQL)
                pk_rows = cur.fetchall()
                cur.execute(_FOREIGN_KEYS_SQL)
                fk_rows = cur.fetchall()
        primary_keys = {(table, column) for table, column in pk_rows}
        columns_by_table: dict[str, list[ColumnInfo]] = {}
        for table, name, data_type, nullable, comment in column_rows:
            columns_by_table.setdefault(table, []).append(
                ColumnInfo(
                    name=name,
                    data_type=data_type,
                    nullable=nullable,
                    primary_key=(table, name) in primary_keys,
                    comment=comment,
                )
            )
        fks_by_table: dict[str, list[ForeignKey]] = {}
        for table, column, ref_table, ref_column in fk_rows:
            fks_by_table.setdefault(table, []).append(
                ForeignKey(column=column, references_table=ref_table, references_column=ref_column)
            )
        samples: dict[str, list[dict[str, str]]] = {}
        if self._sample_rows > 0:
            with psycopg.connect(self._dsn, connect_timeout=5) as conn:
                conn.read_only = True
                with conn.cursor(row_factory=tuple_row) as cur:
                    for name, _comment in table_rows:
                        if name in self._sensitive:
                            continue  # never send sensitive data to the model
                        columns = columns_by_table.get(name, [])
                        for column in columns:
                            if column.data_type in _TEXT_TYPES:
                                column.sample_values = self._categorical_values(
                                    cur, name, column.name
                                )
                        samples[name] = self._fetch_samples(cur, name, columns)
        tables = [
            TableInfo(
                name=name,
                comment=comment,
                columns=columns_by_table.get(name, []),
                foreign_keys=fks_by_table.get(name, []),
                sensitive=name in self._sensitive,
                sample_rows=samples.get(name, []),
            )
            for name, comment in table_rows
        ]
        return DatabaseSchema(tables=tables)

    def _categorical_values(
        self, cur: psycopg.Cursor[tuple[object, ...]], table: str, column: str
    ) -> list[str]:
        """Distinct values of a text column, most frequent first, if it is categorical.

        Categorical means at most ``_MAX_CATEGORIES`` distinct values that
        actually repeat (fewer distinct values than rows). This tells the model
        that ``orders.status`` holds 'paid'/'shipped'/... without dumping
        free-text or identifier columns into the prompt.
        """
        query = sql.SQL(
            "SELECT {col}::text, COUNT(*) FROM {tbl} WHERE {col} IS NOT NULL "
            "GROUP BY 1 ORDER BY 2 DESC, 1 LIMIT {lim}"
        ).format(
            col=sql.Identifier(column),
            tbl=sql.Identifier(table),
            lim=sql.Literal(_MAX_CATEGORIES + 1),
        )
        cur.execute(query)
        rows = cur.fetchall()
        repeats = any(int(str(count)) > 1 for _value, count in rows)
        if not rows or len(rows) > _MAX_CATEGORIES or not repeats:
            return []
        return [_clip(value, 30) for value, _count in rows]

    def _fetch_samples(
        self,
        cur: psycopg.Cursor[tuple[object, ...]],
        table: str,
        columns: list[ColumnInfo],
    ) -> list[dict[str, str]]:
        """A few deterministic, diverse rows of a non-sensitive table.

        Rows are ordered by primary key (stable across runs). When the table
        has a categorical text column, one row per category is preferred, so
        the samples show different values instead of the first N rows, which
        often look alike. Values are stringified and truncated. Sensitive
        tables are never sampled: sample rows go into the LLM prompt, so they
        would leak PII to the model provider.
        """
        order: list[sql.Composable] = [sql.Identifier(c.name) for c in columns if c.primary_key]
        order = order or [sql.SQL("1")]
        order_by = sql.SQL(", ").join(order)
        categorical = next((c for c in columns if c.sample_values), None)
        if categorical is not None:
            spread = sql.Identifier(categorical.name)
            query = sql.SQL(
                "SELECT * FROM (SELECT DISTINCT ON ({spread}) * FROM {tbl} "
                "ORDER BY {spread}, {order}) AS s ORDER BY {order} LIMIT {lim}"
            ).format(
                spread=spread, tbl=sql.Identifier(table), order=order_by,
                lim=sql.Literal(self._sample_rows),
            )
        else:
            query = sql.SQL("SELECT * FROM {tbl} ORDER BY {order} LIMIT {lim}").format(
                tbl=sql.Identifier(table), order=order_by, lim=sql.Literal(self._sample_rows)
            )
        cur.execute(query)
        names = [column.name for column in cur.description or []]
        return [
            {name: _clip(value) for name, value in zip(names, row, strict=True)}
            for row in cur.fetchall()
        ]


def _clip(value: object, limit: int = 40) -> str:
    text = "NULL" if value is None else str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"
