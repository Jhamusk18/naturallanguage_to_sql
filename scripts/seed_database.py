"""Create the demo `shop` schema and data in any PostgreSQL database (e.g. a free Neon one).

    DATABASE_URL=postgresql://... python scripts/seed_database.py

Drops and recreates the five demo tables, then loads db/init/01_schema.sql and
02_seed.sql. Use a database that holds nothing else.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import psycopg

TABLES = ("refunds", "order_items", "orders", "products", "customers")


def main() -> int:
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        print(__doc__)
        return 2
    init = Path(__file__).resolve().parents[1] / "db" / "init"
    with psycopg.connect(dsn, autocommit=True) as conn:
        for table in TABLES:
            conn.execute(f'DROP TABLE IF EXISTS "{table}" CASCADE')
        for name in ("01_schema.sql", "02_seed.sql"):
            conn.execute((init / name).read_text())
        counts = {t: conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone() for t in TABLES}
    print("Seeded:", {t: c[0] if c else 0 for t, c in counts.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
