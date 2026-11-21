"""Bundled few-shot exemplars for the demo e-commerce schema.

Fifteen (question, SQL) pairs covering aggregation, joins, anti-joins, date
math, and window-free analytics — enough variety for retrieval to matter.
"""

from __future__ import annotations

from text_to_sql_engine.models import Exemplar

DEMO_EXEMPLARS: list[Exemplar] = [
    Exemplar(
        question="What share of all orders were cancelled?",
        sql=(
            "SELECT round(100.0 * sum(CASE WHEN orders.status = 'cancelled' THEN 1 ELSE 0 END)"
            " / count(*), 1) AS cancelled_pct FROM orders"
        ),
    ),
    Exemplar(
        question="How many orders are in each status?",
        sql=(
            "SELECT orders.status, count(*) AS order_count FROM orders "
            "GROUP BY orders.status ORDER BY order_count DESC"
        ),
    ),
    Exemplar(
        question="What are the five most expensive products?",
        sql=(
            "SELECT products.name, products.price_cents FROM products "
            "ORDER BY products.price_cents DESC LIMIT 5"
        ),
    ),
    Exemplar(
        question="What is the total revenue from completed orders?",
        sql=(
            "SELECT sum(orders.total_cents) AS revenue_cents FROM orders "
            "WHERE orders.status = 'completed'"
        ),
    ),
    Exemplar(
        question="Which products sold the most units?",
        sql=(
            "SELECT products.name, sum(order_items.quantity) AS units_sold "
            "FROM order_items JOIN products ON products.id = order_items.product_id "
            "GROUP BY products.name ORDER BY units_sold DESC LIMIT 10"
        ),
    ),
    Exemplar(
        question="Which products appear in more than one order?",
        sql=(
            "SELECT products.name, count(DISTINCT order_items.order_id) AS order_count "
            "FROM products JOIN order_items ON order_items.product_id = products.id "
            "GROUP BY products.name HAVING count(DISTINCT order_items.order_id) > 1"
        ),
    ),
    Exemplar(
        question="How many orders were placed in the last 7 days?",
        sql=(
            "SELECT count(*) AS recent_orders FROM orders "
            "WHERE orders.created_at >= now() - INTERVAL '7 days'"
        ),
    ),
    Exemplar(
        question="Which customers have never placed an order?",
        sql=(
            "SELECT customers.full_name, customers.email FROM customers "
            "LEFT JOIN orders ON orders.customer_id = customers.id "
            "WHERE orders.id IS NULL"
        ),
    ),
    Exemplar(
        question="What is the refund rate per product category?",
        sql=(
            "SELECT products.category, "
            "count(DISTINCT refunds.order_id)::float / count(DISTINCT orders.id) AS refund_rate "
            "FROM orders "
            "JOIN order_items ON order_items.order_id = orders.id "
            "JOIN products ON products.id = order_items.product_id "
            "LEFT JOIN refunds ON refunds.order_id = orders.id "
            "GROUP BY products.category"
        ),
    ),
    Exemplar(
        question="What are the most common refund reasons?",
        sql=(
            "SELECT refunds.reason, count(*) AS refund_count FROM refunds "
            "GROUP BY refunds.reason ORDER BY refund_count DESC"
        ),
    ),
    Exemplar(
        question="Show monthly revenue for the current year.",
        sql=(
            "SELECT date_trunc('month', orders.created_at) AS month, "
            "sum(orders.total_cents) AS revenue_cents FROM orders "
            "WHERE orders.created_at >= date_trunc('year', now()) "
            "GROUP BY month ORDER BY month"
        ),
    ),
    Exemplar(
        question="What is the single largest order ever placed?",
        sql=(
            "SELECT orders.id, orders.total_cents FROM orders "
            "ORDER BY orders.total_cents DESC LIMIT 1"
        ),
    ),
    Exemplar(
        question="Which products have never been sold?",
        sql=(
            "SELECT products.name FROM products "
            "LEFT JOIN order_items ON order_items.product_id = products.id "
            "WHERE order_items.id IS NULL"
        ),
    ),
    Exemplar(
        question="What is the average number of items per order?",
        sql=(
            "SELECT avg(item_counts.item_count) AS avg_items_per_order FROM ("
            "SELECT order_items.order_id, sum(order_items.quantity) AS item_count "
            "FROM order_items GROUP BY order_items.order_id"
            ") AS item_counts"
        ),
    ),
    Exemplar(
        question="How much money has been refunded per country?",
        sql=(
            "SELECT customers.country, sum(refunds.amount_cents) AS refunded_cents "
            "FROM refunds "
            "JOIN orders ON orders.id = refunds.order_id "
            "JOIN customers ON customers.id = orders.customer_id "
            "GROUP BY customers.country ORDER BY refunded_cents DESC"
        ),
    ),
]
