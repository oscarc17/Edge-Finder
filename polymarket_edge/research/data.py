from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

EXTREME_MID_CUTOFF = 0.99
EXTREME_DEPTH_MIN = 50.0
MAX_REASONABLE_SPREAD_BPS = 2_500.0


def _market_base(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return conn.execute(
        """
        SELECT
            m.market_id,
            m.event_id,
            m.question,
            m.description,
            m.category,
            COALESCE(m.close_ts, m.end_ts) AS target_close_ts,
            m.created_ts,
            r.resolved_ts
        FROM markets m
        LEFT JOIN resolutions r
            ON m.market_id = r.market_id
        """
    ).df()


def _snapshot_panel(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return conn.execute(
        """
        SELECT
            os.snapshot_ts AS ts,
            os.market_id,
            os.token_id,
            os.best_bid,
            os.best_ask,
            os.mid,
            os.spread,
            os.spread_bps,
            os.bid_depth,
            os.ask_depth,
            os.liquidity_score
        FROM orderbook_snapshots os
        WHERE os.mid IS NOT NULL
        """
    ).df()


def _winner_map(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return conn.execute(
        """
        SELECT
            mo.market_id,
            mo.token_id,
            CASE
                WHEN r.winner_token_id IS NULL THEN NULL
                WHEN CAST(mo.token_id AS VARCHAR) = CAST(r.winner_token_id AS VARCHAR) THEN 1
                ELSE 0
            END AS winner,
            CASE WHEN r.winner_token_id IS NULL THEN 0 ELSE 1 END AS resolution_known
        FROM market_outcomes mo
        LEFT JOIN resolutions r
            ON mo.market_id = r.market_id
        """
    ).df()


def _yes_token_map(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
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
                        CASE WHEN lower(coalesce(outcome_label, '')) = 'yes' THEN 0 ELSE 1 END,
                        outcome_index
                ) AS rn
            FROM market_outcomes
        )
        SELECT market_id, token_id
        FROM ranked
        WHERE rn = 1
        """
    ).df()


def _trade_agg(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return conn.execute(
        """
        WITH hourly AS (
            SELECT
                date_trunc('hour', trade_ts) AS hr,
                token_id,
                SUM(notional) AS trade_notional,
                COUNT(*) AS trade_count,
                SUM(CASE WHEN upper(side)='BUY' THEN notional ELSE 0 END) AS buy_notional,
                SUM(CASE WHEN upper(side)='SELL' THEN notional ELSE 0 END) AS sell_notional
            FROM trades
            WHERE trade_ts IS NOT NULL
              AND token_id IS NOT NULL
            GROUP BY 1,2
        )
        SELECT *
        FROM hourly
        """
    ).df()


def _add_forward_targets(frame: pd.DataFrame, horizons: tuple[int, ...] = (1, 3, 6)) -> pd.DataFrame:
    out = frame.sort_values(["token_id", "ts"]).copy()
    for h in horizons:
        out[f"future_mid_{h}"] = out.groupby("token_id", observed=True)["mid"].shift(-h)
        out[f"future_spread_{h}"] = out.groupby("token_id", observed=True)["spread"].shift(-h)
        out[f"forward_ret_{h}"] = out[f"future_mid_{h}"] / out["mid"] - 1.0
    return out


def _add_event_category_encoding(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["category"] = out["category"].fillna("unknown")
    out["category_code"] = pd.factorize(out["category"])[0].astype(float)
    dummies = pd.get_dummies(out["category"], prefix="cat", dtype=float)
    return pd.concat([out, dummies], axis=1)


def _add_dynamic_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.sort_values(["token_id", "ts"]).copy()
    grp = out.groupby("token_id", observed=True)
    out["velocity_1"] = grp["mid"].pct_change(1)
    out["velocity_3"] = grp["mid"].pct_change(3)
    out["acceleration"] = out["velocity_1"] - grp["velocity_1"].shift(1)
    out["prob_vol_6"] = grp["velocity_1"].transform(lambda s: s.rolling(6, min_periods=3).std())
    out["prob_vol_24"] = grp["velocity_1"].transform(lambda s: s.rolling(24, min_periods=6).std())
    out["depth_total"] = out["bid_depth"].fillna(0.0) + out["ask_depth"].fillna(0.0)
    out["book_imbalance"] = (out["bid_depth"].fillna(0.0) - out["ask_depth"].fillna(0.0)) / (
        out["depth_total"].replace(0.0, np.nan)
    )
    out["book_imbalance"] = out["book_imbalance"].fillna(0.0)
    out["spread_to_vol_6"] = out["spread_bps"].fillna(0.0) / (out["prob_vol_6"].abs() * 10000.0 + 1.0)
    return out


def _add_microstructure_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "trade_buy_sell_imbalance_h" in out.columns:
        out["order_flow_imbalance_h"] = out["trade_buy_sell_imbalance_h"].fillna(0.0)
    else:
        out["order_flow_imbalance_h"] = 0.0

    if "trade_count_h" in out.columns:
        out["trade_frequency_h"] = out["trade_count_h"].fillna(0.0)
    else:
        out["trade_frequency_h"] = 0.0

    vol = out["prob_vol_6"].abs().fillna(0.0)
    q1 = float(vol.quantile(0.33))
    q2 = float(vol.quantile(0.66))
    out["vol_regime"] = np.where(vol <= q1, 0.0, np.where(vol <= q2, 1.0, 2.0))
    out["is_high_vol_regime"] = (out["vol_regime"] >= 2.0).astype(float)

    liq = out["liquidity_log"].fillna(0.0) if "liquidity_log" in out.columns else np.log1p(out["liquidity_score"].fillna(0.0))
    liq_q = float(pd.Series(liq).quantile(0.33))
    out["is_low_liq_regime"] = (liq <= liq_q).astype(float)
    return out


def _add_clock_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["ts"] = pd.to_datetime(out["ts"])
    out["target_close_ts"] = pd.to_datetime(out["target_close_ts"])
    out["resolved_ts"] = pd.to_datetime(out["resolved_ts"])
    out["created_ts"] = pd.to_datetime(out["created_ts"])

    ref_close = out["resolved_ts"].combine_first(out["target_close_ts"])
    out["time_to_resolution_h"] = (ref_close - out["ts"]).dt.total_seconds() / 3600.0
    out["market_age_h"] = (out["ts"] - out["created_ts"]).dt.total_seconds() / 3600.0
    return out


def _add_temporal_dynamics(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.sort_values(["token_id", "ts"]).copy()
    ttr = out["time_to_resolution_h"].fillna(9_999.0).clip(lower=0.0)
    out["ttr_log"] = np.log1p(ttr)
    out["ttr_sq"] = np.square(np.minimum(ttr, 24.0 * 30.0))
    out["near_resolution_24h"] = (ttr <= 24.0).astype(float)
    out["near_resolution_72h"] = (ttr <= 72.0).astype(float)

    grp = out.groupby("token_id", observed=True)
    vel = out["velocity_1"].fillna(0.0)
    fast = grp["velocity_1"].transform(lambda s: s.fillna(0.0).ewm(span=3, min_periods=2).mean())
    slow = grp["velocity_1"].transform(lambda s: s.fillna(0.0).ewm(span=12, min_periods=4).mean())
    out["momentum_decay"] = (fast - slow).fillna(0.0)
    out["momentum_decay_rate"] = (out["momentum_decay"] / (np.abs(slow) + 1e-6)).clip(lower=-5.0, upper=5.0)
    out["velocity_abs"] = np.abs(vel)
    return out


def _add_cross_market_context(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.sort_values(["ts", "market_id", "token_id"]).copy()
    out["event_id"] = out["event_id"].fillna("unknown_event")
    out["category"] = out["category"].fillna("unknown")

    event_grp = out.groupby(["event_id", "ts"], observed=True)
    out["event_mean_mid"] = event_grp["mid"].transform("mean")
    out["related_prob_spread"] = (out["mid"] - out["event_mean_mid"]).fillna(0.0)
    out["event_prob_dispersion"] = event_grp["mid"].transform(lambda s: float(s.max() - s.min()) if len(s) else 0.0).fillna(0.0)

    cat_grp = out.groupby(["category", "ts"], observed=True)
    out["category_mean_mid"] = cat_grp["mid"].transform("mean")
    out["category_prob_spread"] = (out["mid"] - out["category_mean_mid"]).fillna(0.0)

    out["ret_1"] = out.groupby("token_id", observed=True)["mid"].pct_change().fillna(0.0)
    cat_ret = (
        out.groupby(["category", "ts"], observed=True)["ret_1"]
        .mean()
        .rename("category_ret_1")
        .reset_index()
    )
    out = out.merge(cat_ret, on=["category", "ts"], how="left")
    out["category_ret_1"] = out["category_ret_1"].fillna(0.0)

    corr_rows: list[dict[str, float | str]] = []
    for token_id, g in out.groupby("token_id", observed=True):
        if len(g) < 12:
            corr = 0.0
        else:
            token_ret = pd.to_numeric(g["ret_1"], errors="coerce").fillna(0.0)
            cat_bm = pd.to_numeric(g["category_ret_1"], errors="coerce").fillna(0.0)
            corr = float(token_ret.corr(cat_bm)) if float(token_ret.std(ddof=0)) > 1e-12 else 0.0
        corr_rows.append({"token_id": token_id, "token_corr_category": corr})
    corr_map = pd.DataFrame(corr_rows)
    out = out.merge(corr_map, on="token_id", how="left")
    out["token_corr_category"] = out["token_corr_category"].fillna(0.0)

    ranks = out[["token_id", "token_corr_category"]].drop_duplicates().copy()
    ranks["corr_rank"] = ranks["token_corr_category"].rank(method="average", pct=True)
    ranks["corr_cluster"] = np.floor(ranks["corr_rank"] * 4.0).clip(lower=0.0, upper=3.0)
    out = out.merge(ranks[["token_id", "corr_cluster"]], on="token_id", how="left")
    out["corr_cluster"] = out["corr_cluster"].fillna(0.0)
    return out.drop(columns=[c for c in ["ret_1", "corr_rank"] if c in out.columns])


def _join_trade_flow(frame: pd.DataFrame, trade_agg: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if trade_agg.empty:
        out["trade_notional_h"] = 0.0
        out["trade_count_h"] = 0.0
        out["trade_buy_sell_imbalance_h"] = 0.0
        return out

    t = trade_agg.copy()
    t["hr"] = pd.to_datetime(t["hr"])
    out["hr"] = out["ts"].dt.floor("h")
    out = out.merge(t, left_on=["hr", "token_id"], right_on=["hr", "token_id"], how="left")
    out["trade_notional_h"] = out["trade_notional"].fillna(0.0)
    out["trade_count_h"] = out["trade_count"].fillna(0.0)
    imbalance = (out["buy_notional"].fillna(0.0) - out["sell_notional"].fillna(0.0)) / (
        out["trade_notional_h"].replace(0.0, np.nan)
    )
    out["trade_buy_sell_imbalance_h"] = imbalance.fillna(0.0)
    drop_cols = ["hr", "trade_notional", "trade_count", "buy_notional", "sell_notional"]
    return out.drop(columns=[c for c in drop_cols if c in out.columns])


def _apply_orderbook_sanity(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    has_bid = out["best_bid"].notna()
    has_ask = out["best_ask"].notna()
    two_sided = has_bid & has_ask

    out["best_bid_present"] = has_bid.astype(float)
    out["best_ask_present"] = has_ask.astype(float)
    out["book_two_sided"] = two_sided.astype(float)
    out["book_reliable"] = two_sided.astype(float)

    # Canonical tradable mid/spread are only defined for two-sided books.
    out["mid"] = np.where(two_sided, 0.5 * (out["best_bid"].astype(float) + out["best_ask"].astype(float)), np.nan)
    out["spread"] = np.where(two_sided, out["best_ask"].astype(float) - out["best_bid"].astype(float), np.nan)
    out["spread_bps"] = np.where(
        two_sided & (out["mid"] > 0.0),
        (out["spread"] / out["mid"]) * 10000.0,
        np.nan,
    )
    reasonable_spread = pd.to_numeric(out["spread_bps"], errors="coerce").fillna(np.inf) <= float(MAX_REASONABLE_SPREAD_BPS)
    out["spread_reasonable"] = reasonable_spread.astype(float)

    depth_total = out["bid_depth"].fillna(0.0) + out["ask_depth"].fillna(0.0)
    out["depth_total_raw"] = depth_total
    mid = pd.to_numeric(out["mid"], errors="coerce")
    extreme_mid = (mid >= EXTREME_MID_CUTOFF) | (mid <= (1.0 - EXTREME_MID_CUTOFF))
    extreme_depth_ok = depth_total >= float(EXTREME_DEPTH_MIN)
    out["mid_extreme"] = extreme_mid.fillna(False).astype(float)
    out["extreme_depth_ok"] = extreme_depth_ok.fillna(False).astype(float)

    tradable = two_sided & reasonable_spread & (~extreme_mid | extreme_depth_ok)
    out["book_tradable"] = tradable.astype(float)
    return out


def build_orderbook_sanity_reports(conn: duckdb.DuckDBPyConnection) -> dict[str, pd.DataFrame]:
    snaps = _snapshot_panel(conn)
    if snaps.empty:
        empty = pd.DataFrame()
        return {"orderbook_missingness": empty, "orderbook_distribution": empty}

    s = snaps.copy()
    s["best_bid_present"] = s["best_bid"].notna()
    s["best_ask_present"] = s["best_ask"].notna()
    s["two_sided"] = s["best_bid_present"] & s["best_ask_present"]
    s["mid_extreme"] = pd.to_numeric(s["mid"], errors="coerce").fillna(0.5).pipe(
        lambda x: (x >= EXTREME_MID_CUTOFF) | (x <= (1.0 - EXTREME_MID_CUTOFF))
    )
    s["depth_total"] = pd.to_numeric(s["bid_depth"], errors="coerce").fillna(0.0) + pd.to_numeric(s["ask_depth"], errors="coerce").fillna(0.0)
    s["spread_zero"] = pd.to_numeric(s["spread"], errors="coerce").fillna(0.0) <= 0.0

    per_market = (
        s.groupby("market_id", observed=True)
        .agg(
            n_snapshots=("token_id", "count"),
            bid_missing_rate=("best_bid_present", lambda x: float(1.0 - np.mean(x))),
            ask_missing_rate=("best_ask_present", lambda x: float(1.0 - np.mean(x))),
            two_sided_rate=("two_sided", "mean"),
            spread_zero_rate=("spread_zero", "mean"),
            mid_extreme_rate=("mid_extreme", "mean"),
            avg_mid=("mid", "mean"),
            avg_spread=("spread", "mean"),
            avg_depth_total=("depth_total", "mean"),
        )
        .reset_index()
        .sort_values(["bid_missing_rate", "ask_missing_rate", "mid_extreme_rate"], ascending=[False, False, False])
    )

    dist_rows: list[dict[str, float | str]] = []
    mid = pd.to_numeric(s["mid"], errors="coerce").dropna()
    spread = pd.to_numeric(s["spread"], errors="coerce").dropna()
    if len(mid):
        mid_bins = pd.cut(
            mid,
            bins=[0.0, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1.0],
            include_lowest=True,
        )
        counts = mid_bins.value_counts(dropna=False).sort_index()
        total = float(max(1, counts.sum()))
        for bucket, count in counts.items():
            dist_rows.append(
                {
                    "distribution_type": "mid",
                    "bucket": str(bucket),
                    "count": float(count),
                    "pct": float(count) / total,
                }
            )
    if len(spread):
        spread_bins = pd.cut(
            spread,
            bins=[-1e-9, 0.0, 0.001, 0.002, 0.005, 0.01, np.inf],
            include_lowest=True,
        )
        counts = spread_bins.value_counts(dropna=False).sort_index()
        total = float(max(1, counts.sum()))
        for bucket, count in counts.items():
            dist_rows.append(
                {
                    "distribution_type": "spread",
                    "bucket": str(bucket),
                    "count": float(count),
                    "pct": float(count) / total,
                }
            )
    presence = (
        s.groupby(["best_bid_present", "best_ask_present"], observed=True)
        .size()
        .reset_index(name="count")
    )
    total_presence = float(max(1.0, float(presence["count"].sum()))) if not presence.empty else 1.0
    for row in presence.itertuples(index=False):
        dist_rows.append(
            {
                "distribution_type": "quote_presence",
                "bucket": f"bid={bool(row.best_bid_present)}_ask={bool(row.best_ask_present)}",
                "count": float(row.count),
                "pct": float(row.count) / total_presence,
            }
        )
    distribution = pd.DataFrame(dist_rows)
    return {"orderbook_missingness": per_market, "orderbook_distribution": distribution}


def build_feature_panel(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    snaps = _snapshot_panel(conn)
    if snaps.empty:
        return snaps
    markets = _market_base(conn)
    winners = _winner_map(conn)
    trade_agg = _trade_agg(conn)

    panel = snaps.merge(markets, on="market_id", how="left")
    panel = _apply_orderbook_sanity(panel)
    panel = panel[panel["book_tradable"] > 0.0].copy()
    if panel.empty:
        return panel
    panel = panel.merge(winners, on=["market_id", "token_id"], how="left")
    panel["winner"] = pd.to_numeric(panel.get("winner"), errors="coerce")
    panel["resolution_known"] = pd.to_numeric(panel.get("resolution_known"), errors="coerce").fillna(0.0)
    panel = _add_clock_features(panel)
    panel = _add_dynamic_features(panel)
    panel = _join_trade_flow(panel, trade_agg)
    panel = _add_event_category_encoding(panel)
    panel = _add_forward_targets(panel, horizons=(1, 3, 6, 12))

    # Common normalization-friendly transforms.
    panel["liquidity_log"] = np.log1p(panel["liquidity_score"].fillna(0.0))
    panel["depth_log"] = np.log1p(panel["depth_total"].fillna(0.0))
    panel["spread_bps"] = panel["spread_bps"].fillna(0.0)
    panel["prob_vol_6"] = panel["prob_vol_6"].fillna(0.0)
    panel["prob_vol_24"] = panel["prob_vol_24"].fillna(0.0)
    panel["velocity_1"] = panel["velocity_1"].fillna(0.0)
    panel["velocity_3"] = panel["velocity_3"].fillna(0.0)
    panel["acceleration"] = panel["acceleration"].fillna(0.0)
    panel["time_to_resolution_h"] = panel["time_to_resolution_h"].fillna(9_999.0)
    panel["market_age_h"] = panel["market_age_h"].fillna(0.0)
    panel = _add_microstructure_features(panel)
    panel = _add_temporal_dynamics(panel)
    panel = _add_cross_market_context(panel)

    return panel.sort_values(["ts", "token_id"]).reset_index(drop=True)


def build_yes_feature_panel(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    panel = build_feature_panel(conn)
    if panel.empty:
        return panel
    yes_map = _yes_token_map(conn)
    yes = panel.merge(yes_map, on=["market_id", "token_id"], how="inner")
    return yes.sort_values(["ts", "market_id"]).reset_index(drop=True)
