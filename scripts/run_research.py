from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from polymarket_edge.backtest.engine import BacktestConfig, BacktestEngine, BacktestResult
from polymarket_edge.db import get_connection, init_db, upsert_dataframe
from polymarket_edge.edges import cross_market, liquidity_premium, momentum_reversion, resolution_rules, whale_behavior
from polymarket_edge.scoring import score_edges


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _signal_from_momentum(result: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frame = result.get("signal_frame", pd.DataFrame()).copy()
    summary = result.get("summary", pd.DataFrame())
    if frame.empty:
        return pd.DataFrame(columns=["ts", "token_id", "signal"])
    frame = frame.rename(columns={"snapshot_ts": "ts"}).copy()
    dominant = "momentum"
    if not summary.empty:
        dominant = str(summary.iloc[0].get("dominant_style", "momentum"))

    frame["signal_raw"] = frame["ret_lb"].astype(float)
    if dominant == "mean_reversion":
        frame["signal_raw"] *= -1.0

    # If short-horizon returns are constant, fall back to a level-based proxy
    # so the backtest can still evaluate execution and risk plumbing.
    if float(frame["signal_raw"].std(ddof=0)) < 1e-12 and "mid" in frame.columns:
        level = frame["mid"].astype(float) - 0.5
        frame["signal_raw"] = -level if dominant == "mean_reversion" else level

    def _zscore(x: pd.Series) -> pd.Series:
        std = float(x.std(ddof=0))
        if std == 0:
            return pd.Series(np.zeros(len(x)), index=x.index)
        return (x - float(x.mean())) / std

    frame["signal_cs"] = frame.groupby("ts", observed=True)["signal_raw"].transform(_zscore)
    ts_counts = frame.groupby("ts", observed=True)["token_id"].transform("count")
    global_std = float(frame["signal_raw"].std(ddof=0))
    if global_std == 0:
        frame["signal_global"] = 0.0
    else:
        frame["signal_global"] = (frame["signal_raw"] - float(frame["signal_raw"].mean())) / global_std

    frame["signal"] = np.where(ts_counts >= 3, frame["signal_cs"], frame["signal_global"])
    frame["signal"] = frame["signal"].clip(-2, 2) / 2
    return frame[["ts", "token_id", "signal"]]


def _price_frame_for_backtest(conn) -> pd.DataFrame:
    return conn.execute(
        """
        SELECT
            os.snapshot_ts AS ts,
            os.token_id,
            os.market_id,
            m.category,
            os.mid,
            os.spread,
            os.bid_depth,
            os.ask_depth
        FROM orderbook_snapshots os
        LEFT JOIN markets m
            ON os.market_id = m.market_id
        WHERE os.mid IS NOT NULL
        ORDER BY os.snapshot_ts, os.token_id
        """
    ).df()


def _to_metric_rows(run_ts: datetime, edge_name: str, metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.floating)) and np.isfinite(value):
            rows.append(
                {
                    "run_ts": run_ts,
                    "edge_name": edge_name,
                    "entity_id": edge_name,
                    "metric_name": key,
                    "metric_value": float(value),
                    "meta_json": None,
                }
            )
        elif value is not None and not (isinstance(value, float) and np.isnan(value)):
            rows.append(
                {
                    "run_ts": run_ts,
                    "edge_name": edge_name,
                    "entity_id": edge_name,
                    "metric_name": key,
                    "metric_value": np.nan,
                    "meta_json": json.dumps({"value": str(value)}),
                }
            )
    return rows


def main() -> None:
    conn = get_connection()
    init_db(conn)
    run_ts = datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)

    cross_df = cross_market.run(conn)
    rr = resolution_rules.run(conn)
    liq = liquidity_premium.run(conn)
    mom = momentum_reversion.run(conn)
    whale = whale_behavior.run(conn)

    price_frame = _price_frame_for_backtest(conn)
    signal_frame = _signal_from_momentum(mom)
    nonzero_signals = int((signal_frame["signal"].abs() > 1e-9).sum()) if not signal_frame.empty else 0
    if nonzero_signals == 0:
        empty = pd.DataFrame()
        bt_result = BacktestResult(
            summary={"total_pnl": 0.0, "sharpe": 0.0, "max_drawdown": 0.0},
            timeseries=empty,
            trades=empty,
            category_pnl=empty,
        )
    else:
        backtester = BacktestEngine(BacktestConfig())
        bt_result = backtester.run(price_frame, signal_frame)

    run_id = str(uuid.uuid4())
    if not bt_result.timeseries.empty:
        bt_ts = bt_result.timeseries.copy()
        bt_ts.insert(0, "run_id", run_id)
        upsert_dataframe(conn, "backtest_timeseries", bt_ts)
    if not bt_result.trades.empty:
        bt_tr = bt_result.trades.copy()
        bt_tr.insert(0, "run_id", run_id)
        bt_tr = bt_tr[
            [
                "run_id",
                "ts",
                "token_id",
                "market_id",
                "qty",
                "fill_price",
                "fee_paid",
                "slippage_paid",
                "notional",
                "side",
            ]
        ]
        upsert_dataframe(conn, "backtest_trades", bt_tr)
    conn.execute(
        """
        INSERT OR REPLACE INTO backtest_runs
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            run_id,
            run_ts,
            "momentum_or_reversion",
            bt_result.timeseries["ts"].min() if not bt_result.timeseries.empty else None,
            bt_result.timeseries["ts"].max() if not bt_result.timeseries.empty else None,
            _safe_float(bt_result.summary.get("total_pnl")),
            _safe_float(bt_result.summary.get("sharpe")),
            _safe_float(bt_result.summary.get("max_drawdown")),
            json.dumps({"config": BacktestConfig().__dict__}),
        ],
    )

    edge_summary_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []

    if not cross_df.empty:
        p_val = float(cross_df["adf_pvalue"].dropna().median()) if cross_df["adf_pvalue"].notna().any() else 1.0
        stability = float(np.mean(cross_df["adf_pvalue"] < 0.05)) if cross_df["adf_pvalue"].notna().any() else 0.0
        mean_return = float((cross_df["spread_zscore"].abs() / 100.0).mean())
        capacity = float(5000 + 100 * len(cross_df))
        row = {
            "edge_name": "cross_market_inefficiency",
            "p_value": p_val,
            "capacity_usd": capacity,
            "stability": stability,
            "mean_return": mean_return,
        }
        edge_summary_rows.append(row)
        metric_rows.extend(_to_metric_rows(run_ts, row["edge_name"], row))

    rr_summary = rr.get("summary", pd.DataFrame())
    if not rr_summary.empty:
        s = rr_summary.iloc[0]
        row = {
            "edge_name": "resolution_rule_mispricing",
            "p_value": _safe_float(s.get("spearman_pvalue"), 1.0),
            "capacity_usd": float(12000),
            "stability": float(max(0.0, min(1.0, abs(_safe_float(s.get("spearman_corr_abs_error"), 0.0))))),
            "mean_return": float(max(0.0, _safe_float(s.get("spearman_corr_abs_error"), 0.0) * 0.05)),
        }
        edge_summary_rows.append(row)
        metric_rows.extend(_to_metric_rows(run_ts, row["edge_name"], row))

    liq_summary = liq.get("summary", pd.DataFrame())
    if not liq_summary.empty:
        s = liq_summary.iloc[0]
        slope = _safe_float(s.get("slope_abs_error_vs_neglogliq"), 0.0)
        row = {
            "edge_name": "liquidity_premium",
            "p_value": _safe_float(s.get("linreg_pvalue"), 1.0),
            "capacity_usd": float(9000),
            "stability": float(max(0.0, min(1.0, abs(_safe_float(s.get("linreg_r"), 0.0))))),
            "mean_return": float(max(0.0, slope * 0.20)),
        }
        edge_summary_rows.append(row)
        metric_rows.extend(_to_metric_rows(run_ts, row["edge_name"], row))

    mom_summary = mom.get("summary", pd.DataFrame())
    if not mom_summary.empty:
        s = mom_summary.iloc[0]
        bucket = mom.get("bucket_detail", pd.DataFrame())
        stability = 0.5
        if not bucket.empty:
            dominant = str(s.get("dominant_style"))
            stability = float(np.mean(bucket["style"] == dominant))
        row = {
            "edge_name": "momentum_vs_mean_reversion",
            "p_value": _safe_float(s.get("pvalue"), 1.0),
            "capacity_usd": float(15000),
            "stability": stability,
            "mean_return": float(abs(_safe_float(s.get("beta"), 0.0)) * 0.5),
        }
        edge_summary_rows.append(row)
        metric_rows.extend(_to_metric_rows(run_ts, row["edge_name"], row))

    whale_summary = whale.get("summary", pd.DataFrame())
    if not whale_summary.empty:
        s = whale_summary.iloc[0]
        alpha = _safe_float(s.get("mean_alpha_whale"), 0.0)
        row = {
            "edge_name": "whale_behavior",
            "p_value": _safe_float(s.get("whale_minus_non_pvalue"), 1.0),
            "capacity_usd": float(25000),
            "stability": float(0.6 if alpha > 0 else 0.3),
            "mean_return": float(max(0.0, alpha)),
        }
        edge_summary_rows.append(row)
        metric_rows.extend(_to_metric_rows(run_ts, row["edge_name"], row))

    if metric_rows:
        upsert_dataframe(conn, "edge_results", pd.DataFrame(metric_rows))

    edge_summary = pd.DataFrame(edge_summary_rows)
    scored = score_edges(edge_summary) if not edge_summary.empty else edge_summary

    out_dir = Path("data/research_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not edge_summary.empty:
        edge_summary.to_csv(out_dir / "edge_summary.csv", index=False)
    if not scored.empty:
        scored.to_csv(out_dir / "edge_scores.csv", index=False)
    if not cross_df.empty:
        cross_df.to_csv(out_dir / "cross_market_pairs.csv", index=False)
    if not rr.get("market_candidates", pd.DataFrame()).empty:
        rr["market_candidates"].to_csv(out_dir / "resolution_candidates.csv", index=False)
    if not bt_result.timeseries.empty:
        bt_result.timeseries.to_csv(out_dir / "backtest_timeseries.csv", index=False)
    if not bt_result.trades.empty:
        bt_result.trades.to_csv(out_dir / "backtest_trades.csv", index=False)

    print("Research run complete.")
    print(f"Signal diagnostics: rows={len(signal_frame)}, nonzero={nonzero_signals}")
    print(f"Backtest summary: {bt_result.summary}")
    if not scored.empty:
        print(scored[["edge_name", "edge_score", "significance_score", "capacity_score", "stability_score"]])
    else:
        print("No edge scores generated yet. Ingest more history first.")


if __name__ == "__main__":
    main()
