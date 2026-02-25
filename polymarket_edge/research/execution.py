from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from polymarket_edge.research.direction import (
    DIRECTION_LONG_NO,
    DIRECTION_LONG_YES,
)


@dataclass(frozen=True)
class ExecutionCostInputs:
    fee_rate: float = 0.0
    entry_spread: float | pd.Series = 0.0
    exit_spread: float | pd.Series = 0.0
    entry_impact: float | pd.Series = 0.0
    exit_impact: float | pd.Series = 0.0
    entry_vol_slippage: float | pd.Series = 0.0
    exit_vol_slippage: float | pd.Series = 0.0
    extra_cost_pnl: float | pd.Series = 0.0


def _as_series(value: object, index: pd.Index, *, default: float = 0.0) -> pd.Series:
    if isinstance(value, pd.Series):
        return pd.to_numeric(value, errors="coerce").reindex(index).fillna(default)
    if isinstance(value, np.ndarray):
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            return pd.Series(float(arr), index=index, dtype=float)
        if len(arr) != len(index):
            raise ValueError("Array length does not match index length")
        return pd.Series(arr, index=index, dtype=float)
    return pd.Series(float(value if value is not None else default), index=index, dtype=float)


def _normalize_direction_series(direction: pd.Series | str | int | float, index: pd.Index) -> pd.Series:
    if isinstance(direction, pd.Series):
        d = direction.reindex(index)
    else:
        d = pd.Series(direction, index=index)
    if pd.api.types.is_numeric_dtype(d):
        s = pd.to_numeric(d, errors="coerce").fillna(0.0)
        return pd.Series(
            np.where(s > 0.0, DIRECTION_LONG_YES, np.where(s < 0.0, DIRECTION_LONG_NO, "FLAT")),
            index=index,
            dtype=object,
        )
    d = d.astype(str).str.upper().str.strip()
    d = d.replace({"YES": DIRECTION_LONG_YES, "NO": DIRECTION_LONG_NO})
    out = pd.Series("FLAT", index=index, dtype=object)
    out[d == DIRECTION_LONG_YES] = DIRECTION_LONG_YES
    out[d == DIRECTION_LONG_NO] = DIRECTION_LONG_NO
    return out


def compute_trade_return(
    entry: pd.Series | np.ndarray | float,
    exit: pd.Series | np.ndarray | float,
    side: pd.Series | str | int | float,
    costs: ExecutionCostInputs,
    *,
    qty: pd.Series | np.ndarray | float = 1.0,
    prices_are_yes_mid: bool = True,
    min_notional: float = 1e-9,
) -> pd.DataFrame:
    """Compute executable trade returns in consistent return units.

    Parameters
    ----------
    entry, exit
        If ``prices_are_yes_mid`` is True, these are YES-side mid prices and ``side`` selects
        LONG_YES vs LONG_NO (complement token). If False, they are already traded-instrument
        executable mids in the traded side's price space.
    side
        LONG_YES / LONG_NO (or +1 / -1). Used for direction semantics and long-NO conversion.
    costs
        Per-side costs in price units plus fee rate.
    qty
        Quantity in contracts.
    """

    if isinstance(entry, pd.Series):
        index = entry.index
    elif isinstance(exit, pd.Series):
        index = exit.index
    elif isinstance(qty, pd.Series):
        index = qty.index
    elif isinstance(side, pd.Series):
        index = side.index
    else:
        index = pd.RangeIndex(1)

    entry_yes = _as_series(entry, index)
    exit_yes = _as_series(exit, index)
    qty_s = _as_series(qty, index, default=1.0).clip(lower=0.0)
    dlab = _normalize_direction_series(side, index)
    is_long_yes = dlab == DIRECTION_LONG_YES
    is_long_no = dlab == DIRECTION_LONG_NO

    if prices_are_yes_mid:
        entry_mid = pd.Series(np.where(is_long_yes, entry_yes, np.where(is_long_no, 1.0 - entry_yes, np.nan)), index=index, dtype=float)
        exit_mid = pd.Series(np.where(is_long_yes, exit_yes, np.where(is_long_no, 1.0 - exit_yes, np.nan)), index=index, dtype=float)
    else:
        entry_mid = entry_yes.astype(float)
        exit_mid = exit_yes.astype(float)

    entry_mid = entry_mid.clip(lower=1e-6, upper=1 - 1e-6)
    exit_mid = exit_mid.clip(lower=1e-6, upper=1 - 1e-6)

    entry_spread = _as_series(costs.entry_spread, index).clip(lower=0.0)
    exit_spread = _as_series(costs.exit_spread, index).clip(lower=0.0)
    entry_impact = _as_series(costs.entry_impact, index).clip(lower=0.0)
    exit_impact = _as_series(costs.exit_impact, index).clip(lower=0.0)
    entry_vol = _as_series(costs.entry_vol_slippage, index).clip(lower=0.0)
    exit_vol = _as_series(costs.exit_vol_slippage, index).clip(lower=0.0)
    extra_cost_pnl = _as_series(costs.extra_cost_pnl, index).clip(lower=0.0)
    fee_rate = float(costs.fee_rate)

    entry_price = (entry_mid + entry_spread + entry_impact + entry_vol).clip(lower=1e-6, upper=0.999999)
    exit_price = (exit_mid - exit_spread - exit_impact - exit_vol).clip(lower=1e-6, upper=0.999999)

    gross_pnl = (exit_price - entry_price) * qty_s
    fees = fee_rate * (entry_price * qty_s + exit_price * qty_s)
    spread_component = (entry_spread + exit_spread) * qty_s
    impact_component = (entry_impact + exit_impact) * qty_s
    vol_slippage_component = (entry_vol + exit_vol) * qty_s
    net_pnl = gross_pnl - fees - extra_cost_pnl
    notional = (entry_price * qty_s).clip(lower=float(min_notional))
    trade_return = (net_pnl / notional).astype(float)
    gross_return = (gross_pnl / notional).astype(float)
    mid_return = (((exit_mid - entry_mid) * qty_s) / notional).astype(float)

    out = pd.DataFrame(index=index)
    out["direction"] = dlab.astype(str)
    out["entry_mid"] = entry_mid.astype(float)
    out["exit_mid"] = exit_mid.astype(float)
    out["entry_price"] = entry_price.astype(float)
    out["exit_price"] = exit_price.astype(float)
    out["qty"] = qty_s.astype(float)
    out["notional"] = notional.astype(float)
    out["gross_pnl"] = gross_pnl.astype(float)
    out["fees"] = fees.astype(float)
    out["spread_component"] = spread_component.astype(float)
    out["impact_component"] = impact_component.astype(float)
    out["vol_slippage_component"] = vol_slippage_component.astype(float)
    out["extra_cost_pnl"] = extra_cost_pnl.astype(float)
    out["net_pnl"] = net_pnl.astype(float)
    out["gross_return"] = gross_return
    out["mid_return"] = mid_return
    out["trade_return"] = trade_return
    return out

