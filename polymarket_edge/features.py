from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import duckdb
import numpy as np
import pandas as pd


def load_yes_token_map(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return conn.execute(
        """
        WITH ranked AS (
            SELECT
                market_id,
                token_id,
                outcome_label,
                outcome_index,
                ROW_NUMBER() OVER (
                    PARTITION BY market_id
                    ORDER BY
                        CASE
                            WHEN lower(coalesce(outcome_label, '')) = 'yes' THEN 0
                            ELSE 1
                        END,
                        outcome_index
                ) AS rn
            FROM market_outcomes
        )
        SELECT market_id, token_id
        FROM ranked
        WHERE rn = 1
        """
    ).df()


def load_latest_yes_probabilities(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return conn.execute(
        """
        WITH yes_token AS (
            SELECT
                market_id,
                token_id,
                ROW_NUMBER() OVER (
                    PARTITION BY market_id
                    ORDER BY
                        CASE
                            WHEN lower(coalesce(outcome_label, '')) = 'yes' THEN 0
                            ELSE 1
                        END,
                        outcome_index
                ) AS rn
            FROM market_outcomes
        ),
        latest AS (
            SELECT
                os.market_id,
                os.token_id,
                os.snapshot_ts,
                os.mid,
                ROW_NUMBER() OVER (
                    PARTITION BY os.token_id
                    ORDER BY os.snapshot_ts DESC
                ) AS rn
            FROM orderbook_snapshots os
            JOIN yes_token yt
                ON os.token_id = yt.token_id
               AND yt.rn = 1
            WHERE os.mid IS NOT NULL
        )
        SELECT
            l.market_id,
            l.token_id,
            l.snapshot_ts,
            l.mid AS prob,
            m.event_id,
            m.question,
            m.category,
            m.close_ts
        FROM latest l
        JOIN markets m
            ON l.market_id = m.market_id
        WHERE l.rn = 1
        """
    ).df()


def load_yes_probability_panel(conn: duckdb.DuckDBPyConnection, market_ids: Iterable[str]) -> pd.DataFrame:
    ids = list(market_ids)
    if not ids:
        return pd.DataFrame()
    id_values = ",".join([f"'{x}'" for x in ids])
    return conn.execute(
        f"""
        WITH yes_token AS (
            SELECT
                market_id,
                token_id,
                ROW_NUMBER() OVER (
                    PARTITION BY market_id
                    ORDER BY
                        CASE
                            WHEN lower(coalesce(outcome_label, '')) = 'yes' THEN 0
                            ELSE 1
                        END,
                        outcome_index
                ) AS rn
            FROM market_outcomes
            WHERE market_id IN ({id_values})
        )
        SELECT
            os.snapshot_ts,
            os.market_id,
            os.mid AS prob
        FROM orderbook_snapshots os
        JOIN yes_token yt
            ON os.token_id = yt.token_id
           AND yt.rn = 1
        WHERE os.mid IS NOT NULL
          AND os.market_id IN ({id_values})
        ORDER BY os.snapshot_ts
        """
    ).df()


def build_pre_resolution_frame(conn: duckdb.DuckDBPyConnection, horizon_hours: int = 24) -> pd.DataFrame:
    horizon_hours = int(horizon_hours)
    orderbook_frame = conn.execute(
        f"""
        WITH target AS (
            SELECT
                r.market_id,
                r.winner_token_id,
                COALESCE(r.resolved_ts, m.close_ts) AS resolved_ts,
                COALESCE(r.resolved_ts, m.close_ts) - INTERVAL '{horizon_hours} hours' AS target_ts,
                m.category,
                m.question,
                m.description
            FROM resolutions r
            JOIN markets m
                ON r.market_id = m.market_id
            WHERE COALESCE(r.resolved_ts, m.close_ts) IS NOT NULL
        ),
        nearest AS (
            SELECT
                t.market_id,
                t.winner_token_id,
                t.resolved_ts,
                t.target_ts,
                t.category,
                t.question,
                t.description,
                os.token_id,
                os.mid,
                os.spread_bps,
                os.liquidity_score,
                os.snapshot_ts,
                ROW_NUMBER() OVER (
                    PARTITION BY t.market_id, os.token_id
                    ORDER BY ABS(date_diff('second', os.snapshot_ts, t.target_ts))
                ) AS rn
            FROM target t
            JOIN orderbook_snapshots os
                ON os.market_id = t.market_id
               AND os.mid IS NOT NULL
        )
        SELECT
            market_id,
            token_id,
            winner_token_id,
            resolved_ts,
            target_ts,
            category,
            question,
            description,
            snapshot_ts,
            mid AS prob,
            spread_bps,
            liquidity_score
        FROM nearest
        WHERE rn = 1
        """
    ).df()

    trade_frame = conn.execute(
        f"""
        WITH target AS (
            SELECT
                r.market_id,
                r.winner_token_id,
                COALESCE(r.resolved_ts, m.close_ts) AS resolved_ts,
                COALESCE(r.resolved_ts, m.close_ts) - INTERVAL '{horizon_hours} hours' AS target_ts,
                m.category,
                m.question,
                m.description
            FROM resolutions r
            JOIN markets m
                ON r.market_id = m.market_id
            WHERE COALESCE(r.resolved_ts, m.close_ts) IS NOT NULL
        ),
        nearest AS (
            SELECT
                t.market_id,
                t.winner_token_id,
                t.resolved_ts,
                t.target_ts,
                t.category,
                t.question,
                t.description,
                tr.token_id,
                tr.price AS prob,
                CAST(NULL AS DOUBLE) AS spread_bps,
                tr.notional AS liquidity_score,
                tr.trade_ts AS snapshot_ts,
                ROW_NUMBER() OVER (
                    PARTITION BY t.market_id, tr.token_id
                    ORDER BY ABS(date_diff('second', tr.trade_ts, t.target_ts))
                ) AS rn
            FROM target t
            JOIN trades tr
                ON tr.market_id = t.market_id
        )
        SELECT
            market_id,
            token_id,
            winner_token_id,
            resolved_ts,
            target_ts,
            category,
            question,
            description,
            snapshot_ts,
            prob,
            spread_bps,
            liquidity_score
        FROM nearest
        WHERE rn = 1
        """
    ).df()

    if orderbook_frame.empty and trade_frame.empty:
        return pd.DataFrame()

    if orderbook_frame.empty:
        frame = trade_frame
    elif trade_frame.empty:
        frame = orderbook_frame
    else:
        seen = set(zip(orderbook_frame["market_id"], orderbook_frame["token_id"]))
        fallback = trade_frame[
            ~trade_frame.apply(lambda r: (r["market_id"], r["token_id"]) in seen, axis=1)
        ]
        frame = pd.concat([orderbook_frame, fallback], ignore_index=True)

    frame["y"] = (frame["token_id"] == frame["winner_token_id"]).astype(float)
    frame["abs_error"] = (frame["prob"] - frame["y"]).abs()
    frame["signed_error"] = frame["prob"] - frame["y"]
    frame["edge_long"] = frame["y"] - frame["prob"]
    frame["hours_to_resolution"] = (
        (pd.to_datetime(frame["resolved_ts"]) - pd.to_datetime(frame["snapshot_ts"])).dt.total_seconds() / 3600.0
    )
    return frame


@dataclass
class RegressionSummary:
    slope: float
    intercept: float
    r2: float


def simple_linear_fit(x: pd.Series, y: pd.Series) -> RegressionSummary:
    valid = ~(x.isna() | y.isna())
    x_vals = x[valid].astype(float).to_numpy()
    y_vals = y[valid].astype(float).to_numpy()
    if len(x_vals) < 3:
        return RegressionSummary(slope=np.nan, intercept=np.nan, r2=np.nan)
    slope, intercept = np.polyfit(x_vals, y_vals, 1)
    pred = slope * x_vals + intercept
    ss_res = float(np.sum((y_vals - pred) ** 2))
    ss_tot = float(np.sum((y_vals - np.mean(y_vals)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    return RegressionSummary(slope=float(slope), intercept=float(intercept), r2=r2)
