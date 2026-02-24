from __future__ import annotations

from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd
from scipy import stats
import warnings

from polymarket_edge.features import build_pre_resolution_frame


@dataclass
class LiquidityPremiumConfig:
    horizon_hours: int = 24
    min_samples: int = 20
    buckets: int = 5


def _safe_ttest_ind(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    if float(a.std(ddof=0)) < 1e-12 and float(b.std(ddof=0)) < 1e-12:
        return 0.0, 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
    return float(t_stat), float(p_val)


def run(conn: duckdb.DuckDBPyConnection, config: LiquidityPremiumConfig = LiquidityPremiumConfig()) -> dict[str, pd.DataFrame]:
    frame = build_pre_resolution_frame(conn, horizon_hours=config.horizon_hours)
    required = {"abs_error", "liquidity_score"}
    if frame.empty or not required.issubset(frame.columns):
        return {"summary": pd.DataFrame(), "bucket_detail": pd.DataFrame()}
    frame = frame.dropna(subset=["abs_error", "liquidity_score"])
    if frame.empty or len(frame) < config.min_samples:
        return {"summary": pd.DataFrame(), "bucket_detail": pd.DataFrame()}

    frame["liq_bucket"] = pd.qcut(frame["liquidity_score"].rank(method="first"), q=config.buckets, labels=False)
    bucket = (
        frame.groupby("liq_bucket", observed=True)
        .agg(
            mean_abs_error=("abs_error", "mean"),
            median_abs_error=("abs_error", "median"),
            avg_liquidity=("liquidity_score", "mean"),
            n=("market_id", "count"),
        )
        .reset_index()
        .sort_values("liq_bucket")
    )

    frame["neg_log_liq"] = -np.log1p(frame["liquidity_score"])
    x = frame["neg_log_liq"].astype(float)
    y = frame["abs_error"].astype(float)
    if float(x.std(ddof=0)) < 1e-12:
        slope, r_val, p_val = 0.0, 0.0, 1.0
    else:
        slope, _intercept, r_val, p_val, _ = stats.linregress(x, y)
    low = frame[frame["liq_bucket"] == frame["liq_bucket"].min()]["abs_error"]
    high = frame[frame["liq_bucket"] == frame["liq_bucket"].max()]["abs_error"]
    if len(low) > 5 and len(high) > 5:
        t_stat, t_p = _safe_ttest_ind(low, high)
    else:
        t_stat, t_p = np.nan, np.nan

    summary = pd.DataFrame(
        [
            {
                "edge_name": "liquidity_premium",
                "n_samples": len(frame),
                "slope_abs_error_vs_neglogliq": slope,
                "linreg_r": r_val,
                "linreg_pvalue": p_val,
                "low_vs_high_tstat": t_stat,
                "low_vs_high_pvalue": t_p,
                "horizon_hours": config.horizon_hours,
            }
        ]
    )
    return {"summary": summary, "bucket_detail": bucket}
