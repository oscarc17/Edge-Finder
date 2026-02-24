from __future__ import annotations

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st

from polymarket_edge.config import settings
from polymarket_edge.scoring import example_edge_summary, score_edges


st.set_page_config(page_title="Polymarket Edge Dashboard", layout="wide")
st.title("Polymarket Edge Research Dashboard")

conn = duckdb.connect(settings.db_path, read_only=True)

try:
    bt = conn.execute("SELECT ts, equity FROM backtest_timeseries ORDER BY ts").df()
except Exception:
    bt = pd.DataFrame()

try:
    cat = conn.execute(
        """
        SELECT
            t.category,
            SUM(t.notional) AS traded_notional
        FROM (
            SELECT
                bt.token_id,
                bt.notional,
                m.category
            FROM backtest_trades bt
            LEFT JOIN market_outcomes mo
                ON bt.token_id = mo.token_id
            LEFT JOIN markets m
                ON mo.market_id = m.market_id
        ) t
        GROUP BY t.category
        ORDER BY traded_notional DESC
        """
    ).df()
except Exception:
    cat = pd.DataFrame()

try:
    raw_edges = conn.execute(
        """
        SELECT
            edge_name,
            MIN(metric_value) FILTER (WHERE metric_name ILIKE '%pvalue%') AS p_value,
            AVG(metric_value) FILTER (WHERE metric_name = 'capacity_usd') AS capacity_usd,
            AVG(metric_value) FILTER (WHERE metric_name = 'stability') AS stability,
            AVG(metric_value) FILTER (WHERE metric_name = 'mean_return') AS mean_return
        FROM edge_results
        GROUP BY edge_name
        """
    ).df()
except Exception:
    raw_edges = pd.DataFrame()

if raw_edges.empty:
    raw_edges = example_edge_summary()

scored = score_edges(raw_edges.fillna({"p_value": 1.0, "capacity_usd": 0.0, "stability": 0.0, "mean_return": 0.0}))

col1, col2, col3 = st.columns(3)
col1.metric("Tracked Edges", len(scored))
col2.metric("Top Edge Score", f"{scored['edge_score'].max():.3f}" if not scored.empty else "n/a")
col3.metric("DB Path", settings.db_path)

st.subheader("Edge Ranking")
st.dataframe(
    scored[
        [
            "edge_name",
            "edge_score",
            "significance_score",
            "capacity_score",
            "stability_score",
            "mean_return",
        ]
    ].sort_values("edge_score", ascending=False),
    use_container_width=True,
)

if not bt.empty:
    bt["ts"] = pd.to_datetime(bt["ts"])
    fig = px.line(bt, x="ts", y="equity", title="Backtest Equity Curve")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No backtest equity data yet. Run `scripts/run_research.py` after collecting data.")

if not cat.empty:
    fig_cat = px.bar(cat, x="category", y="traded_notional", title="Backtest Notional by Market Category")
    st.plotly_chart(fig_cat, use_container_width=True)

st.caption("Metrics are examples until ingestion and research jobs populate the database.")
