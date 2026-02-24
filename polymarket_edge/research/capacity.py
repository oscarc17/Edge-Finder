from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_capacity(
    trades: pd.DataFrame,
    *,
    max_participation: float = 0.15,
    impact_coeff: float = 0.10,
) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(
            [
                {
                    "estimated_daily_capacity_usd": 0.0,
                    "median_trade_capacity_usd": 0.0,
                    "active_markets": 0.0,
                    "avg_impact_bps": 0.0,
                }
            ]
        )

    t = trades.copy()
    for col in ["mid", "depth_total", "notional"]:
        if col not in t.columns:
            t[col] = 0.0
    t["ts"] = pd.to_datetime(t["ts"])
    cap_trade = max_participation * t["depth_total"].clip(lower=1.0) * t["mid"].clip(lower=1e-4)
    impact = impact_coeff * (t["notional"].clip(lower=0.0) / (t["depth_total"].clip(lower=1.0) * t["mid"].clip(lower=1e-4)))
    impact_bps = impact * 10_000.0

    by_day = t.assign(day=t["ts"].dt.floor("d")).groupby("day", observed=True)["notional"].sum()
    daily_capacity = float(np.nanmedian(by_day)) if len(by_day) else 0.0

    return pd.DataFrame(
        [
            {
                "estimated_daily_capacity_usd": daily_capacity,
                "median_trade_capacity_usd": float(np.nanmedian(cap_trade)) if len(cap_trade) else 0.0,
                "active_markets": float(t["market_id"].nunique()) if "market_id" in t.columns else 0.0,
                "avg_impact_bps": float(np.nanmean(impact_bps)) if len(impact_bps) else 0.0,
            }
        ]
    )
