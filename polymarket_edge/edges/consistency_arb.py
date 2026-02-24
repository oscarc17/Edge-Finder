from __future__ import annotations

from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd

from polymarket_edge.research.linking import refresh_market_links


@dataclass(frozen=True)
class ConsistencyArbConfig:
    cost_buffer_bps: float = 35.0
    min_violation_prob: float = 0.005
    link_confidence_min: float = 0.65


def _latest_orderbook(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return conn.execute(
        """
        WITH latest_ts AS (
            SELECT MAX(snapshot_ts) AS ts FROM orderbook_snapshots
        )
        SELECT
            o.snapshot_ts AS ts,
            o.market_id,
            o.token_id,
            mo.outcome_label,
            mo.outcome_index,
            o.best_bid,
            o.best_ask,
            o.mid,
            o.spread,
            o.spread_bps,
            o.bid_depth,
            o.ask_depth
        FROM orderbook_snapshots o
        JOIN latest_ts lt
            ON o.snapshot_ts = lt.ts
        LEFT JOIN market_outcomes mo
            ON o.market_id = mo.market_id
           AND o.token_id = mo.token_id
        WHERE o.mid IS NOT NULL
        """
    ).df()


def _complement_parity(latest: pd.DataFrame, cfg: ConsistencyArbConfig) -> pd.DataFrame:
    if latest.empty:
        return pd.DataFrame()
    t = latest.copy()
    t["label_norm"] = t["outcome_label"].astype(str).str.lower().str.strip()
    yes = t[t["label_norm"] == "yes"].copy()
    no = t[t["label_norm"] == "no"].copy()
    if yes.empty or no.empty:
        return pd.DataFrame()
    pair = yes.merge(
        no[["market_id", "ts", "mid", "spread", "spread_bps", "bid_depth", "ask_depth"]].rename(
            columns={
                "mid": "mid_no",
                "spread": "spread_no",
                "spread_bps": "spread_bps_no",
                "bid_depth": "bid_depth_no",
                "ask_depth": "ask_depth_no",
            }
        ),
        on=["market_id", "ts"],
        how="inner",
    )
    if pair.empty:
        return pd.DataFrame()
    pair["mid_yes"] = pd.to_numeric(pair["mid"], errors="coerce")
    pair["mid_no"] = pd.to_numeric(pair["mid_no"], errors="coerce")
    pair["sum_prob"] = pair["mid_yes"] + pair["mid_no"]
    pair["violation_prob"] = (pair["sum_prob"] - 1.0).abs()
    pair["cost_buffer_return"] = (
        (pd.to_numeric(pair["spread"], errors="coerce").fillna(0.0) + pd.to_numeric(pair["spread_no"], errors="coerce").fillna(0.0))
        + (2.0 * cfg.cost_buffer_bps / 10_000.0)
    )
    pair["expected_gross_return"] = pair["violation_prob"]
    pair["expected_net_return"] = pair["expected_gross_return"] - pair["cost_buffer_return"]
    pair["trade_side"] = np.where(pair["sum_prob"] > 1.0, "BUY_CHEAPER_LEG", "SELL_RICHER_LEG")
    pair["hedged"] = 1
    out = pair[pair["violation_prob"] >= float(cfg.min_violation_prob)].copy()
    if out.empty:
        return out
    return out[
        [
            "ts",
            "market_id",
            "mid_yes",
            "mid_no",
            "sum_prob",
            "violation_prob",
            "cost_buffer_return",
            "expected_gross_return",
            "expected_net_return",
            "trade_side",
            "hedged",
        ]
    ].sort_values("expected_net_return", ascending=False)


def _multi_outcome_constraints(latest: pd.DataFrame, cfg: ConsistencyArbConfig) -> pd.DataFrame:
    if latest.empty:
        return pd.DataFrame()
    t = latest.copy()
    agg = (
        t.groupby(["ts", "market_id"], observed=True)
        .agg(
            n_outcomes=("token_id", "count"),
            sum_prob=("mid", "sum"),
            avg_spread=("spread", "mean"),
            max_prob=("mid", "max"),
            min_prob=("mid", "min"),
        )
        .reset_index()
    )
    if agg.empty:
        return agg
    agg["violation_prob"] = np.where(
        agg["n_outcomes"] >= 2,
        np.maximum(0.0, (agg["sum_prob"] - 1.0).abs()),
        0.0,
    )
    agg["expected_gross_return"] = agg["violation_prob"]
    agg["cost_buffer_return"] = agg["avg_spread"].fillna(0.0) + (cfg.cost_buffer_bps / 10_000.0)
    agg["expected_net_return"] = agg["expected_gross_return"] - agg["cost_buffer_return"]
    return agg[agg["violation_prob"] >= cfg.min_violation_prob].sort_values("expected_net_return", ascending=False)


def _time_window_consistency(conn: duckdb.DuckDBPyConnection, cfg: ConsistencyArbConfig) -> pd.DataFrame:
    try:
        links = refresh_market_links(conn)
    except Exception:
        links = pd.DataFrame()
    if links.empty:
        return pd.DataFrame()
    links = links[
        (links["relation_type"].astype(str).str.contains("time_window", case=False, na=False))
        & (pd.to_numeric(links["link_confidence"], errors="coerce").fillna(0.0) >= float(cfg.link_confidence_min))
    ].copy()
    if links.empty:
        return pd.DataFrame()
    latest = _latest_orderbook(conn)
    if latest.empty:
        return pd.DataFrame()
    yes = latest[latest["outcome_label"].astype(str).str.lower().eq("yes")].copy()
    if yes.empty:
        return pd.DataFrame()
    yes_px = yes.groupby("market_id", observed=True)["mid"].mean().rename("mid_yes").reset_index()
    m = conn.execute(
        "SELECT market_id, question, category, close_ts, created_ts FROM markets"
    ).df()
    rows = []
    for r in links.itertuples(index=False):
        a = str(r.market_id_a)
        b = str(r.market_id_b)
        pa = yes_px[yes_px["market_id"] == a]
        pb = yes_px[yes_px["market_id"] == b]
        if pa.empty or pb.empty:
            continue
        ma = m[m["market_id"] == a]
        mb = m[m["market_id"] == b]
        if ma.empty or mb.empty:
            continue
        close_a = pd.to_datetime(ma["close_ts"].iloc[0], errors="coerce")
        close_b = pd.to_datetime(mb["close_ts"].iloc[0], errors="coerce")
        if pd.isna(close_a) or pd.isna(close_b) or close_a == close_b:
            continue
        p_a = float(pa["mid_yes"].iloc[0])
        p_b = float(pb["mid_yes"].iloc[0])
        if close_a < close_b:
            violation = max(0.0, p_a - p_b)
            earlier, later = a, b
        else:
            violation = max(0.0, p_b - p_a)
            earlier, later = b, a
        if violation < cfg.min_violation_prob:
            continue
        rows.append(
            {
                "market_id_earlier": earlier,
                "market_id_later": later,
                "prob_earlier": p_a if earlier == a else p_b,
                "prob_later": p_b if later == b else p_a,
                "violation_prob": float(violation),
                "expected_gross_return": float(violation),
                "expected_net_return": float(violation - (cfg.cost_buffer_bps / 10_000.0)),
                "link_confidence": float(r.link_confidence),
                "relation_type": str(r.relation_type),
                "hedged": 1,
            }
        )
    return pd.DataFrame(rows).sort_values("expected_net_return", ascending=False) if rows else pd.DataFrame()


def run(conn: duckdb.DuckDBPyConnection, config: ConsistencyArbConfig = ConsistencyArbConfig()) -> dict[str, pd.DataFrame]:
    latest = _latest_orderbook(conn)
    complement = _complement_parity(latest, config)
    multi = _multi_outcome_constraints(latest, config)
    time_consistency = _time_window_consistency(conn, config)
    summary = pd.DataFrame(
        [
            {
                "module": "consistency_arb",
                "n_complement_candidates": float(len(complement)),
                "n_multioutcome_candidates": float(len(multi)),
                "n_timewindow_candidates": float(len(time_consistency)),
                "avg_expected_net_return": float(
                    pd.concat(
                        [
                            complement.get("expected_net_return", pd.Series(dtype=float)),
                            multi.get("expected_net_return", pd.Series(dtype=float)),
                            time_consistency.get("expected_net_return", pd.Series(dtype=float)),
                        ],
                        ignore_index=True,
                    ).mean()
                )
                if (not complement.empty or not multi.empty or not time_consistency.empty)
                else 0.0,
            }
        ]
    )
    return {
        "summary": summary,
        "complement_parity": complement,
        "multi_outcome_constraints": multi,
        "time_window_consistency": time_consistency,
    }

