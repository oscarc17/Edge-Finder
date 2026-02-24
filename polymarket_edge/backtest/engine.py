from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from polymarket_edge.backtest.metrics import summarize_equity


@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    fee_bps: float = 20.0
    impact_coeff: float = 0.10
    max_impact: float = 0.05
    max_participation: float = 0.25
    max_gross_exposure: float = 1.0
    max_weight_per_token: float = 0.05


@dataclass
class BacktestResult:
    summary: dict[str, float]
    timeseries: pd.DataFrame
    trades: pd.DataFrame
    category_pnl: pd.DataFrame


class BacktestEngine:
    """
    Signal-driven simulator.
    `signals` columns: ts, token_id, signal in [-1, 1]
    `prices` columns: ts, token_id, market_id, category, mid, spread, bid_depth, ask_depth
    """

    def __init__(self, config: BacktestConfig = BacktestConfig()) -> None:
        self.config = config

    @staticmethod
    def _sanitize(prices: pd.DataFrame, signals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        p = prices.copy()
        s = signals.copy()
        needed_p = {"ts", "token_id", "mid"}
        needed_s = {"ts", "token_id", "signal"}
        if not needed_p.issubset(p.columns):
            return pd.DataFrame(columns=list(needed_p)), pd.DataFrame(columns=list(needed_s))
        if not needed_s.issubset(s.columns):
            return p, pd.DataFrame(columns=list(needed_s))
        p["ts"] = pd.to_datetime(p["ts"])
        s["ts"] = pd.to_datetime(s["ts"])
        p = p.dropna(subset=["ts", "token_id", "mid"]).sort_values(["ts", "token_id"])
        s = s.dropna(subset=["ts", "token_id", "signal"]).sort_values(["ts", "token_id"])
        p["spread"] = p["spread"].fillna(0.0).clip(lower=0.0)
        p["depth"] = np.minimum(p["bid_depth"].fillna(0.0), p["ask_depth"].fillna(0.0)).clip(lower=1.0)
        return p, s

    def run(self, prices: pd.DataFrame, signals: pd.DataFrame) -> BacktestResult:
        prices, signals = self._sanitize(prices, signals)
        if prices.empty or signals.empty:
            empty = pd.DataFrame()
            return BacktestResult(
                summary={"total_pnl": 0.0, "sharpe": 0.0, "max_drawdown": 0.0},
                timeseries=empty,
                trades=empty,
                category_pnl=empty,
            )

        timeline = sorted(set(prices["ts"]).union(set(signals["ts"])))
        price_by_ts = {ts: g.set_index("token_id") for ts, g in prices.groupby("ts")}
        signal_by_ts = {ts: g.set_index("token_id") for ts, g in signals.groupby("ts")}

        cash = float(self.config.initial_capital)
        positions: dict[str, float] = {}
        target_signal: dict[str, float] = {}
        market_lookup: dict[str, tuple[str, str]] = {}
        last_mid: dict[str, float] = {}
        category_running: dict[str, float] = {}

        trades_log: list[dict[str, Any]] = []
        ts_log: list[dict[str, Any]] = []

        fee_rate = self.config.fee_bps / 10_000.0
        equity = cash

        for ts in timeline:
            prev_mid = last_mid.copy()
            px_frame = price_by_ts.get(ts)
            if px_frame is not None:
                for token_id, row in px_frame.iterrows():
                    last_mid[token_id] = float(row["mid"])
                    market_lookup[token_id] = (
                        str(row.get("market_id")) if row.get("market_id") is not None else "",
                        str(row.get("category")) if row.get("category") is not None else "unknown",
                    )

            sig_frame = signal_by_ts.get(ts)
            if sig_frame is not None:
                for token_id, row in sig_frame.iterrows():
                    target_signal[token_id] = float(np.clip(row["signal"], -1.0, 1.0))

            if not last_mid:
                continue

            marked = sum(qty * last_mid.get(tok, 0.0) for tok, qty in positions.items())
            equity = cash + marked
            gross_budget = max(0.0, self.config.max_gross_exposure * equity)

            for token_id, sig in target_signal.items():
                mid = last_mid.get(token_id)
                if mid is None or mid <= 0:
                    continue
                target_weight = sig * self.config.max_weight_per_token
                target_notional = target_weight * gross_budget
                current_qty = positions.get(token_id, 0.0)
                target_qty = target_notional / mid
                row = px_frame.loc[token_id] if px_frame is not None and token_id in px_frame.index else None
                spread = float(row["spread"]) if row is not None else 0.0
                depth = float(row["depth"]) if row is not None else 1.0
                raw_d_qty = target_qty - current_qty
                max_step_qty = max(1e-9, self.config.max_participation * depth)
                d_qty = float(np.clip(raw_d_qty, -max_step_qty, max_step_qty))
                if abs(d_qty) < 1e-9:
                    continue
                side = 1.0 if d_qty > 0 else -1.0
                impact = min(self.config.max_impact, self.config.impact_coeff * (abs(d_qty) / depth))
                half_spread = spread / 2.0
                slippage = half_spread + impact
                fill = float(np.clip(mid + side * slippage, 0.001, 0.999))
                notional = abs(d_qty) * fill
                fee = fee_rate * notional

                cash -= d_qty * fill
                cash -= fee
                positions[token_id] = current_qty + d_qty

                market_id, category = market_lookup.get(token_id, ("", "unknown"))
                category_running.setdefault(category, 0.0)
                trades_log.append(
                    {
                        "ts": ts,
                        "token_id": token_id,
                        "market_id": market_id,
                        "category": category,
                        "qty": d_qty,
                        "fill_price": fill,
                        "fee_paid": fee,
                        "slippage_paid": abs(d_qty) * slippage,
                        "notional": notional,
                        "side": "BUY" if d_qty > 0 else "SELL",
                    }
                )

            pnl_step_by_cat: dict[str, float] = {}
            if px_frame is not None:
                for token_id, row in px_frame.iterrows():
                    prev = prev_mid.get(token_id)
                    if prev is None:
                        continue
                    qty = positions.get(token_id, 0.0)
                    if qty == 0:
                        continue
                    mid = float(row["mid"])
                    pnl = qty * (mid - prev)
                    category = market_lookup.get(token_id, ("", "unknown"))[1]
                    pnl_step_by_cat[category] = pnl_step_by_cat.get(category, 0.0) + pnl

            for cat, pnl in pnl_step_by_cat.items():
                category_running[cat] = category_running.get(cat, 0.0) + pnl

            marked = sum(qty * last_mid.get(tok, 0.0) for tok, qty in positions.items())
            equity = cash + marked
            gross = sum(abs(qty * last_mid.get(tok, 0.0)) for tok, qty in positions.items())
            net = sum(qty * last_mid.get(tok, 0.0) for tok, qty in positions.items())
            ts_log.append(
                {
                    "ts": ts,
                    "equity": equity,
                    "cash": cash,
                    "gross_exposure": gross,
                    "net_exposure": net,
                }
            )

        timeseries = pd.DataFrame(ts_log).sort_values("ts")
        trades = pd.DataFrame(trades_log).sort_values("ts") if trades_log else pd.DataFrame(trades_log)
        category_pnl = (
            pd.DataFrame([{"category": cat, "pnl": pnl} for cat, pnl in category_running.items()]).sort_values("pnl", ascending=False)
            if category_running
            else pd.DataFrame(columns=["category", "pnl"])
        )
        summary = summarize_equity(timeseries) if not timeseries.empty else {"total_pnl": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}
        return BacktestResult(summary=summary, timeseries=timeseries, trades=trades, category_pnl=category_pnl)
