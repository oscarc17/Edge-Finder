from __future__ import annotations

from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd

from polymarket_edge.research.data import build_yes_feature_panel


@dataclass(frozen=True)
class MicrostructureMMConfig:
    min_depth_total: float = 200.0
    max_prob_vol_6: float = 0.06
    min_mid: float = 0.10
    max_mid: float = 0.90
    base_notional: float = 100.0
    inventory_cap_usd: float = 1_000.0
    adverse_momentum_cutoff: float = 0.02
    fee_bps: float = 20.0
    fill_liquidity_coef: float = 0.9
    fill_spread_coef: float = 1.2
    queue_penalty: float = 0.5
    max_return_clip: float = 0.05


def _fill_prob(depth: pd.Series, spread_bps: pd.Series, flow: pd.Series, cfg: MicrostructureMMConfig) -> pd.Series:
    liq_term = cfg.fill_liquidity_coef * np.log1p(depth.clip(lower=0.0) / 100.0)
    spr_term = -cfg.fill_spread_coef * (spread_bps.clip(lower=0.0) / 200.0)
    flow_term = 0.4 * np.log1p(flow.clip(lower=0.0))
    logit = (liq_term + spr_term + flow_term - cfg.queue_penalty).clip(lower=-20.0, upper=20.0)
    return 1.0 / (1.0 + np.exp(-logit))


def run(
    conn: duckdb.DuckDBPyConnection,
    config: MicrostructureMMConfig = MicrostructureMMConfig(),
    *,
    panel: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    p = panel.copy() if isinstance(panel, pd.DataFrame) else build_yes_feature_panel(conn)
    if p.empty:
        empty = pd.DataFrame()
        return {"summary": empty, "trades": empty, "live_quotes": empty}

    df = p.copy()
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df = df.sort_values(["token_id", "ts"])
    required = {"mid", "spread", "depth_total", "future_mid_1"}
    if not required.issubset(df.columns):
        empty = pd.DataFrame()
        return {"summary": empty, "trades": empty, "live_quotes": empty}

    df["depth_total"] = pd.to_numeric(df["depth_total"], errors="coerce").fillna(0.0)
    df["spread_bps"] = pd.to_numeric(df.get("spread_bps"), errors="coerce").fillna(0.0)
    df["prob_vol_6"] = pd.to_numeric(df.get("prob_vol_6"), errors="coerce").fillna(0.0).abs()
    df["velocity_1"] = pd.to_numeric(df.get("velocity_1"), errors="coerce").fillna(0.0)
    df["trade_notional_h"] = pd.to_numeric(df.get("trade_notional_h"), errors="coerce").fillna(0.0)
    df["time_to_resolution_h"] = pd.to_numeric(df.get("time_to_resolution_h"), errors="coerce").fillna(9999.0)

    tradable = (
        (pd.to_numeric(df.get("book_tradable"), errors="coerce").fillna(1.0) > 0.0)
        & (df["depth_total"] >= float(config.min_depth_total))
        & (df["prob_vol_6"] <= float(config.max_prob_vol_6))
        & (pd.to_numeric(df["mid"], errors="coerce").between(float(config.min_mid), float(config.max_mid)))
    )
    # Adverse selection filter: avoid quoting against strong short-term directional pressure.
    adverse = df["velocity_1"].abs() >= float(config.adverse_momentum_cutoff)
    mm = df[tradable & (~adverse)].copy()
    if mm.empty:
        return {
            "summary": pd.DataFrame([{"module": "microstructure_mm", "n_quotes": 0.0, "n_trades": 0.0, "expectancy": 0.0, "avg_fill_prob": 0.0}]),
            "trades": pd.DataFrame(),
            "live_quotes": pd.DataFrame(),
        }

    # Fair value shifts away from informed flow direction.
    fair = pd.to_numeric(mm["mid"], errors="coerce").fillna(0.5) - 0.25 * mm["velocity_1"].clip(-0.05, 0.05)
    fair = fair.clip(lower=0.001, upper=0.999)
    half_spread = (pd.to_numeric(mm["spread"], errors="coerce").fillna(0.0) / 2.0).clip(lower=0.001, upper=0.05)
    widen = 1.0 + 1.5 * (mm["prob_vol_6"] / (mm["prob_vol_6"].median() + 1e-6))
    near_close = (mm["time_to_resolution_h"] <= 24.0).astype(float)
    half_spread = (half_spread * widen * (1.0 + 0.5 * near_close)).clip(lower=0.001, upper=0.10)
    half_spread = np.minimum(half_spread, 0.25 * pd.to_numeric(mm["mid"], errors="coerce").fillna(0.5))

    bid_quote = (fair - half_spread).clip(lower=0.001, upper=0.999)
    ask_quote = (fair + half_spread).clip(lower=0.001, upper=0.999)
    mm["quote_bid"] = bid_quote
    mm["quote_ask"] = ask_quote
    mm["quote_width"] = ask_quote - bid_quote

    fill_bid = _fill_prob(mm["depth_total"], mm["spread_bps"], mm["trade_notional_h"], config)
    fill_ask = _fill_prob(mm["depth_total"], mm["spread_bps"], mm["trade_notional_h"], config)
    # Inventory proxy decays quoting size near close and in same-direction momentum.
    size_notional = float(config.base_notional) * (1.0 - 0.4 * near_close) * (1.0 - 0.2 * mm["prob_vol_6"].clip(0.0, 1.0))
    size_notional = size_notional.clip(lower=10.0, upper=float(config.base_notional))

    future_mid = pd.to_numeric(mm["future_mid_1"], errors="coerce").fillna(mm["mid"])
    fee_rate = float(config.fee_bps) / 10_000.0
    # Conservative spread-capture proxy in YES-price return units.
    adverse_bid = (mm["quote_bid"] - future_mid).clip(lower=0.0)
    adverse_ask = (future_mid - mm["quote_ask"]).clip(lower=0.0)
    bid_capture = (half_spread - adverse_bid) / mm["mid"].clip(lower=1e-3)
    ask_capture = (half_spread - adverse_ask) / mm["mid"].clip(lower=1e-3)
    gross_return = 0.5 * (fill_bid * bid_capture + fill_ask * ask_capture)
    gross_return = gross_return.clip(lower=-float(config.max_return_clip), upper=float(config.max_return_clip))
    fee_return = 2.0 * fee_rate * (fill_bid + fill_ask) * 0.5
    net_return = gross_return - fee_return

    trades = mm.copy()
    trades["fill_prob_bid"] = fill_bid.astype(float)
    trades["fill_prob_ask"] = fill_ask.astype(float)
    trades["expected_fill_probability"] = 0.5 * (trades["fill_prob_bid"] + trades["fill_prob_ask"])
    trades["quoted_notional"] = size_notional.astype(float)
    trades["expected_gross_return"] = gross_return.astype(float)
    trades["expected_cost_return"] = fee_return.astype(float)
    trades["expected_net_return"] = net_return.astype(float)
    trades["realized_return_proxy"] = net_return.astype(float)
    trades["strategy_rationale"] = "passive_spread_capture_with_adverse_selection_filter"
    trades["inventory_cap_usd"] = float(config.inventory_cap_usd)

    live_cols = [
        c
        for c in [
            "ts",
            "market_id",
            "token_id",
            "category",
            "mid",
            "quote_bid",
            "quote_ask",
            "quote_width",
            "expected_fill_probability",
            "expected_net_return",
            "quoted_notional",
            "strategy_rationale",
        ]
        if c in trades.columns
    ]
    live_quotes = trades[live_cols].sort_values("expected_net_return", ascending=False).head(200)

    summary = pd.DataFrame(
        [
            {
                "module": "microstructure_mm",
                "n_quotes": float(len(mm)),
                "n_trades": float(len(trades)),
                "expectancy": float(pd.to_numeric(trades["realized_return_proxy"], errors="coerce").mean()) if len(trades) else 0.0,
                "avg_fill_prob": float(pd.to_numeric(trades["expected_fill_probability"], errors="coerce").mean()) if len(trades) else 0.0,
                "avg_quote_width": float(pd.to_numeric(trades["quote_width"], errors="coerce").mean()) if len(trades) else 0.0,
                "avg_expected_net_return": float(pd.to_numeric(trades["expected_net_return"], errors="coerce").mean()) if len(trades) else 0.0,
            }
        ]
    )
    return {"summary": summary, "trades": trades, "live_quotes": live_quotes}
