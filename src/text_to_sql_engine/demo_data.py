"""Bundled demo questions, FakeLLM SQL templates, and offline fixture rows.

The fixture rows are the *actual* answers for db/init/02_seed.sql, computed
by scripts/generate_seed.py — so the offline demo and a live docker-compose
database give the same numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

from text_to_sql_engine.execution.executor import ExecutionResult


@dataclass(frozen=True)
class DemoCase:
    question: str
    result: ExecutionResult


# Needle (matched against the question) -> SQL the FakeLLM "generates".
# Template 2 deliberately omits LIMIT (showing injection); template 6 is a
# deliberate SELECT * over a sensitive table (showing rejection).
DEMO_SQL_TEMPLATES: dict[str, str] = {
    "generated the most revenue": (
        "SELECT products.name, sum(order_items.quantity * order_items.unit_price_cents) "
        "AS revenue_cents FROM order_items "
        "JOIN products ON products.id = order_items.product_id "
        "GROUP BY products.name ORDER BY revenue_cents DESC LIMIT 5"
    ),
    "orders are in each status": (
        "SELECT status, count(*) AS order_count FROM orders "
        "GROUP BY status ORDER BY order_count DESC"
    ),
    "average order value by country": (
        "SELECT customers.country, avg(orders.total_cents) AS avg_order_cents "
        "FROM orders JOIN customers ON customers.id = orders.customer_id "
        "GROUP BY customers.country ORDER BY avg_order_cents DESC"
    ),
    "most recent refunds": (
        "SELECT refunds.id, refunds.amount_cents, refunds.reason, "
        "orders.status AS order_status, refunds.created_at "
        "FROM refunds JOIN orders ON orders.id = refunds.order_id "
        "ORDER BY refunds.created_at DESC LIMIT 10"
    ),
    "top 3 customers by total spend": (
        "SELECT customers.full_name, sum(orders.total_cents) AS total_spent_cents "
        "FROM orders JOIN customers ON customers.id = orders.customer_id "
        "GROUP BY customers.full_name ORDER BY total_spent_cents DESC LIMIT 3"
    ),
    "everything about our customers": "SELECT * FROM customers",
}

# FakeLLM fallback for questions with no template. It references a table that
# cannot exist, so the table allowlist rejects it with a self-explanatory
# message instead of the demo silently "succeeding" with a placeholder query.
FAKE_LLM_NO_TEMPLATE_SQL = (
    "SELECT * FROM fake_llm_has_no_template_for_this_question_set_OPENAI_API_KEY"
)

DEMO_CASES: list[DemoCase] = [
    DemoCase(
        question="Which 5 products have generated the most revenue?",
        result=ExecutionResult(
            columns=["name", "revenue_cents"],
            rows=[
                {"name": "4K Action Camera", "revenue_cents": 878400},
                {"name": "Air Purifier", "revenue_cents": 298800},
                {"name": "Trail Running Shoes", "revenue_cents": 166600},
                {"name": "Yoga Mat Premium", "revenue_cents": 103500},
                {"name": "Smart Home Hub", "revenue_cents": 99000},
            ],
        ),
    ),
    DemoCase(
        question="How many orders are in each status?",
        result=ExecutionResult(
            columns=["status", "order_count"],
            rows=[
                {"status": "completed", "order_count": 11},
                {"status": "shipped", "order_count": 7},
                {"status": "paid", "order_count": 5},
                {"status": "pending", "order_count": 4},
                {"status": "cancelled", "order_count": 3},
            ],
        ),
    ),
    DemoCase(
        question="What is the average order value by country?",
        result=ExecutionResult(
            columns=["country", "avg_order_cents"],
            rows=[
                {"country": "PH", "avg_order_cents": 153600.00},
                {"country": "CA", "avg_order_cents": 75460.00},
                {"country": "GB", "avg_order_cents": 42500.00},
                {"country": "US", "avg_order_cents": 38600.00},
                {"country": "DE", "avg_order_cents": 25420.00},
            ],
        ),
    ),
    DemoCase(
        question="Show the 10 most recent refunds with their order status.",
        result=ExecutionResult(
            columns=["id", "amount_cents", "reason", "order_status", "created_at"],
            rows=[
                {
                    "id": 6, "amount_cents": 17600, "reason": "changed mind",
                    "order_status": "completed", "created_at": "2026-04-15",
                },
                {
                    "id": 5, "amount_cents": 18900, "reason": "damaged in transit",
                    "order_status": "shipped", "created_at": "2026-03-07",
                },
                {
                    "id": 4, "amount_cents": 3225, "reason": "late delivery",
                    "order_status": "shipped", "created_at": "2026-02-19",
                },
                {
                    "id": 3, "amount_cents": 5400, "reason": "changed mind",
                    "order_status": "completed", "created_at": "2026-02-03",
                },
                {
                    "id": 2, "amount_cents": 7250, "reason": "wrong item shipped",
                    "order_status": "completed", "created_at": "2026-01-21",
                },
                {
                    "id": 1, "amount_cents": 18900, "reason": "damaged in transit",
                    "order_status": "completed", "created_at": "2026-01-08",
                },
            ],
        ),
    ),
    DemoCase(
        question="Who are our top 3 customers by total spend?",
        result=ExecutionResult(
            columns=["full_name", "total_spent_cents"],
            rows=[
                {"full_name": "Luis Santos", "total_spent_cents": 579000},
                {"full_name": "Felix Moreau", "total_spent_cents": 300800},
                {"full_name": "Carmen Diaz", "total_spent_cents": 235800},
            ],
        ),
    ),
    DemoCase(
        # Deliberately rejected: the FakeLLM "generates" SELECT * over the
        # sensitive customers table, and the guardrails refuse it.
        question="Show me everything about our customers.",
        result=ExecutionResult(),
    ),
]


@dataclass(frozen=True)
class CorrectionShowcase:
    """A scripted fail-then-succeed run that makes the correction loop visible.

    The first SQL is *statically* valid — every table and column exists, so
    the AST guardrails pass it — but it breaks Postgres grouping rules. That
    is exactly the class of error the EXPLAIN preflight exists to catch: the
    database reports it before any row is read, the loop feeds the error back,
    and the second attempt succeeds.
    """

    question: str
    bad_sql: str
    good_sql: str
    explain_error_message: str
    explain_error_sqlstate: str
    result: ExecutionResult


CORRECTION_SHOWCASE = CorrectionShowcase(
    question="What is the total spend per country?",
    bad_sql=(
        "SELECT customers.country, customers.full_name, "
        "sum(orders.total_cents) AS spent_cents "
        "FROM orders JOIN customers ON customers.id = orders.customer_id "
        "GROUP BY customers.country ORDER BY spent_cents DESC"
    ),
    good_sql=(
        "SELECT customers.country, sum(orders.total_cents) AS spent_cents "
        "FROM orders JOIN customers ON customers.id = orders.customer_id "
        "GROUP BY customers.country ORDER BY spent_cents DESC"
    ),
    # The exact message PostgreSQL emits for this statement (SQLSTATE 42803).
    explain_error_message=(
        'column "customers.full_name" must appear in the GROUP BY clause '
        "or be used in an aggregate function"
    ),
    explain_error_sqlstate="42803",
    result=ExecutionResult(
        columns=["country", "spent_cents"],
        rows=[
            {"country": "PH", "spent_cents": 768000},
            {"country": "CA", "spent_cents": 377300},
            {"country": "US", "spent_cents": 347400},
            {"country": "GB", "spent_cents": 255000},
            {"country": "DE", "spent_cents": 127100},
        ],
    ),
)
