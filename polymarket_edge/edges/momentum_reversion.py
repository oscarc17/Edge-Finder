from __future__ import annotations

from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class MomentumConfig:
    lookback_hours: int = 3
    forward_hours: int = 3
    max_hours_to_close: int = 24 * 365
    min_samples: int = 50


def _safe_linregress(x: pd.Series, y: pd.Series) -> tuple[float, float, float, float]:
    x_num = pd.to_numeric(x, errors="coerce").dropna()
    y_num = pd.to_numeric(y, errors="coerce").dropna()
    joined = pd.DataFrame({"x": x_num, "y": y_num}).dropna()
    if joined.empty:
        return 0.0, 0.0, 0.0, 1.0
    if float(joined["x"].std(ddof=0)) < 1e-12:
        return 0.0, 0.0, 0.0, 1.0
    beta, _intercept, r_val, p_val, _ = stats.linregress(joined["x"], joined["y"])
    return float(beta), float(_intercept), float(r_val), float(p_val)


def _load_panel(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return conn.execute(
        """
        SELECT
            os.snapshot_ts,
            os.market_id,
            os.token_id,
            os.mid,
            COALESCE(m.close_ts, m.end_ts) AS close_ts,
            m.category
        FROM orderbook_snapshots os
        JOIN markets m
            ON os.market_id = m.market_id
        WHERE os.mid IS NOT NULL
        """
    ).df()


def run(conn: duckdb.DuckDBPyConnection, config: MomentumConfig = MomentumConfig()) -> dict[str, pd.DataFrame]:
    panel = _load_panel(conn)
    if panel.empty:
        return {"summary": pd.DataFrame(), "bucket_detail": pd.DataFrame(), "signal_frame": pd.DataFrame()}

    panel["snapshot_ts"] = pd.to_datetime(panel["snapshot_ts"])
    panel["close_ts"] = pd.to_datetime(panel["close_ts"])
    panel = panel.sort_values(["token_id", "snapshot_ts"])

    panel["ret_lb"] = panel.groupby("token_id")["mid"].pct_change(config.lookback_hours)
    panel["ret_fw"] = panel.groupby("token_id")["mid"].shift(-config.forward_hours) / panel["mid"] - 1.0
    panel["hours_to_close"] = (panel["close_ts"] - panel["snapshot_ts"]).dt.total_seconds() / 3600.0
    frame = panel[
        panel["hours_to_close"].between(1, config.max_hours_to_close)
        & panel["ret_lb"].notna()
        & panel["ret_fw"].notna()
    ].copy()
    if len(frame) < config.min_samples:
        return {"summary": pd.DataFrame(), "bucket_detail": pd.DataFrame(), "signal_frame": frame}

    beta, intercept, r_val, p_val = _safe_linregress(frame["ret_lb"], frame["ret_fw"])
    style = "momentum" if beta > 0 else "mean_reversion"

    bins = [0, 24, 72, config.max_hours_to_close]
    labels = ["0-24h", "24-72h", "72h+"]
    frame["h2c_bucket"] = pd.cut(frame["hours_to_close"], bins=bins, labels=labels, include_lowest=True)

    bucket_rows: list[dict[str, object]] = []
    for bucket, g in frame.groupby("h2c_bucket", observed=True):
        if len(g) < 30:
            continue
        b, c, rv, pv = _safe_linregress(g["ret_lb"], g["ret_fw"])
        bucket_rows.append(
            {
                "bucket": str(bucket),
                "n": len(g),
                "beta": b,
                "corr": rv,
                "pvalue": pv,
                "style": "momentum" if b > 0 else "mean_reversion",
            }
        )

    summary = pd.DataFrame(
        [
            {
                "edge_name": "momentum_vs_mean_reversion",
                "n_samples": len(frame),
                "beta": beta,
                "corr": r_val,
                "pvalue": p_val,
                "dominant_style": style,
                "lookback_hours": config.lookback_hours,
                "forward_hours": config.forward_hours,
            }
        ]
    )

    signal_frame = frame[["snapshot_ts", "market_id", "token_id", "mid", "ret_lb", "ret_fw", "hours_to_close", "category"]]
    return {"summary": summary, "bucket_detail": pd.DataFrame(bucket_rows), "signal_frame": signal_frame}
