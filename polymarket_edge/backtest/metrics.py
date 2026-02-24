from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 24 * 365) -> float:
    r = returns.dropna()
    if len(r) < 3 or float(r.std(ddof=0)) == 0.0:
        return 0.0
    return float((r.mean() / r.std(ddof=0)) * np.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    dd = (equity / running_max) - 1.0
    return float(dd.min())


def summarize_equity(timeseries: pd.DataFrame) -> dict[str, float]:
    if timeseries.empty:
        return {"total_pnl": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}
    series = timeseries.sort_values("ts").copy()
    series["ret"] = series["equity"].pct_change().replace([np.inf, -np.inf], np.nan)
    total_pnl = float(series["equity"].iloc[-1] - series["equity"].iloc[0])
    return {
        "total_pnl": total_pnl,
        "sharpe": sharpe_ratio(series["ret"]),
        "max_drawdown": max_drawdown(series["equity"]),
    }
