from __future__ import annotations

from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd

from polymarket_edge.config import settings


SCHEMA_SQL: Iterable[str] = (
    """
    CREATE TABLE IF NOT EXISTS markets (
        market_id VARCHAR PRIMARY KEY,
        event_id VARCHAR,
        slug VARCHAR,
        question VARCHAR,
        description VARCHAR,
        category VARCHAR,
        tags VARCHAR,
        condition_id VARCHAR,
        start_ts TIMESTAMP,
        end_ts TIMESTAMP,
        close_ts TIMESTAMP,
        active BOOLEAN,
        closed BOOLEAN,
        archived BOOLEAN,
        accepting_orders BOOLEAN,
        created_ts TIMESTAMP,
        updated_ts TIMESTAMP,
        last_seen_ts TIMESTAMP,
        raw_json VARCHAR
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS market_outcomes (
        market_id VARCHAR,
        token_id VARCHAR,
        outcome_index INTEGER,
        outcome_label VARCHAR,
        winner BOOLEAN,
        PRIMARY KEY (market_id, token_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS orderbook_snapshots (
        snapshot_ts TIMESTAMP,
        market_id VARCHAR,
        token_id VARCHAR,
        best_bid DOUBLE,
        best_ask DOUBLE,
        mid DOUBLE,
        spread DOUBLE,
        spread_bps DOUBLE,
        bid_depth DOUBLE,
        ask_depth DOUBLE,
        liquidity_score DOUBLE,
        PRIMARY KEY (snapshot_ts, token_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS price_history (
        token_id VARCHAR,
        market_id VARCHAR,
        ts TIMESTAMP,
        price DOUBLE,
        PRIMARY KEY (token_id, ts)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS trades (
        trade_id VARCHAR PRIMARY KEY,
        market_id VARCHAR,
        token_id VARCHAR,
        trader VARCHAR,
        side VARCHAR,
        price DOUBLE,
        size DOUBLE,
        notional DOUBLE,
        trade_ts TIMESTAMP,
        tx_hash VARCHAR,
        raw_json VARCHAR
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS holders (
        snapshot_ts TIMESTAMP,
        market_id VARCHAR,
        wallet VARCHAR,
        outcome VARCHAR,
        shares DOUBLE,
        pct DOUBLE,
        PRIMARY KEY (snapshot_ts, market_id, wallet, outcome)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS resolutions (
        market_id VARCHAR PRIMARY KEY,
        resolved_ts TIMESTAMP,
        winner_token_id VARCHAR,
        winner_outcome_index INTEGER,
        winner_label VARCHAR,
        resolution_source VARCHAR,
        raw_json VARCHAR
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS resolved_markets (
        market_id VARCHAR PRIMARY KEY,
        event_id VARCHAR,
        question VARCHAR,
        category VARCHAR,
        resolved_ts TIMESTAMP,
        winner_token_id VARCHAR,
        winner_label VARCHAR,
        resolution_source VARCHAR,
        close_ts TIMESTAMP,
        last_backfill_ts TIMESTAMP,
        raw_json VARCHAR
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS market_links (
        link_id VARCHAR PRIMARY KEY,
        market_id_a VARCHAR,
        market_id_b VARCHAR,
        relation_type VARCHAR,
        link_confidence DOUBLE,
        heuristic_source VARCHAR,
        created_ts TIMESTAMP,
        updated_ts TIMESTAMP,
        metadata_json VARCHAR
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS outcome_links (
        market_id VARCHAR PRIMARY KEY,
        yes_token_id VARCHAR,
        no_token_id VARCHAR,
        link_source VARCHAR,
        created_ts TIMESTAMP,
        updated_ts TIMESTAMP,
        metadata_json VARCHAR
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS market_sets (
        set_id VARCHAR,
        market_id VARCHAR,
        outcome_label VARCHAR,
        weight DOUBLE,
        set_type VARCHAR,
        link_confidence DOUBLE,
        created_ts TIMESTAMP,
        updated_ts TIMESTAMP,
        metadata_json VARCHAR,
        PRIMARY KEY (set_id, market_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS returns (
        token_id VARCHAR,
        ts TIMESTAMP,
        ret_1h DOUBLE,
        ret_6h DOUBLE,
        ret_24h DOUBLE,
        PRIMARY KEY (token_id, ts)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS edge_results (
        run_ts TIMESTAMP,
        edge_name VARCHAR,
        entity_id VARCHAR,
        metric_name VARCHAR,
        metric_value DOUBLE,
        meta_json VARCHAR
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS backtest_runs (
        run_id VARCHAR PRIMARY KEY,
        run_ts TIMESTAMP,
        strategy_name VARCHAR,
        start_ts TIMESTAMP,
        end_ts TIMESTAMP,
        total_pnl DOUBLE,
        sharpe DOUBLE,
        max_drawdown DOUBLE,
        metadata_json VARCHAR
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS backtest_timeseries (
        run_id VARCHAR,
        ts TIMESTAMP,
        equity DOUBLE,
        cash DOUBLE,
        gross_exposure DOUBLE,
        net_exposure DOUBLE,
        PRIMARY KEY (run_id, ts)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS backtest_trades (
        run_id VARCHAR,
        ts TIMESTAMP,
        token_id VARCHAR,
        market_id VARCHAR,
        qty DOUBLE,
        fill_price DOUBLE,
        fee_paid DOUBLE,
        slippage_paid DOUBLE,
        notional DOUBLE,
        side VARCHAR
    );
    """,
)


INDEX_SQL: Iterable[str] = (
    "CREATE INDEX IF NOT EXISTS idx_orderbook_token_ts ON orderbook_snapshots(token_id, snapshot_ts);",
    "CREATE INDEX IF NOT EXISTS idx_orderbook_market_ts ON orderbook_snapshots(market_id, snapshot_ts);",
    "CREATE INDEX IF NOT EXISTS idx_trades_market_ts ON trades(market_id, trade_ts);",
    "CREATE INDEX IF NOT EXISTS idx_trades_trader_ts ON trades(trader, trade_ts);",
    "CREATE INDEX IF NOT EXISTS idx_returns_token_ts ON returns(token_id, ts);",
    "CREATE INDEX IF NOT EXISTS idx_resolutions_market ON resolutions(market_id);",
    "CREATE INDEX IF NOT EXISTS idx_resolved_markets_ts ON resolved_markets(resolved_ts);",
    "CREATE INDEX IF NOT EXISTS idx_market_links_a ON market_links(market_id_a);",
    "CREATE INDEX IF NOT EXISTS idx_market_links_b ON market_links(market_id_b);",
    "CREATE INDEX IF NOT EXISTS idx_outcome_links_market ON outcome_links(market_id);",
    "CREATE INDEX IF NOT EXISTS idx_market_sets_set ON market_sets(set_id);",
    "CREATE INDEX IF NOT EXISTS idx_market_sets_market ON market_sets(market_id);",
)


def get_connection(db_path: str | None = None) -> duckdb.DuckDBPyConnection:
    target_path = db_path or settings.db_path
    Path(target_path).parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(target_path)
    conn.execute("PRAGMA threads=4;")
    return conn


def init_db(conn: duckdb.DuckDBPyConnection) -> None:
    for stmt in SCHEMA_SQL:
        conn.execute(stmt)
    for stmt in INDEX_SQL:
        conn.execute(stmt)


def upsert_dataframe(conn: duckdb.DuckDBPyConnection, table: str, frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    temp_view = f"_tmp_{table}"
    conn.register(temp_view, frame)
    try:
        conn.execute(f"INSERT OR REPLACE INTO {table} SELECT * FROM {temp_view}")
    except duckdb.BinderException:
        conn.execute(f"INSERT INTO {table} SELECT * FROM {temp_view}")
    conn.unregister(temp_view)
    return len(frame)
