from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from polymarket_edge.backtest.metrics import max_drawdown, sharpe_ratio
from polymarket_edge.research.direction import DIRECTION_LONG_NO, DIRECTION_LONG_YES
from polymarket_edge.research.semantics import PredictionSpace, attach_prediction_contract


@dataclass
class PairedBacktestConfig:
    holding_periods: tuple[int, ...] = (1, 3, 6)
    fee_bps: float = 20.0
    impact_coeff: float = 0.08
    max_impact: float = 0.05
    base_trade_notional: float = 250.0
    max_participation: float = 0.10
    vol_slippage_coeff: float = 0.15
    fill_alpha_liquidity: float = 1.75
    fill_beta_spread: float = 1.0
    fill_gamma_flow: float = 0.50
    fill_risk_buffer_bps: float = 5.0
    staleness_tolerance_ms: float = 1_000.0
    execution_regime: str = "base"  # optimistic | base | pessimistic
    random_seed: int = 7


def _regime_params(cfg: PairedBacktestConfig) -> dict[str, float]:
    regime = str(cfg.execution_regime).lower().strip()
    if regime == "optimistic":
        return {
            "impact_coeff": cfg.impact_coeff * 0.7,
            "max_impact": cfg.max_impact * 0.8,
            "vol_slippage_coeff": cfg.vol_slippage_coeff * 0.7,
            "fill_alpha": cfg.fill_alpha_liquidity + 0.5,
            "fill_beta": cfg.fill_beta_spread * 0.8,
            "fill_gamma": cfg.fill_gamma_flow * 1.2,
            "fill_risk_buffer_bps": cfg.fill_risk_buffer_bps * 0.5,
        }
    if regime == "pessimistic":
        return {
            "impact_coeff": cfg.impact_coeff * 1.4,
            "max_impact": cfg.max_impact * 1.4,
            "vol_slippage_coeff": cfg.vol_slippage_coeff * 1.5,
            "fill_alpha": cfg.fill_alpha_liquidity - 0.5,
            "fill_beta": cfg.fill_beta_spread * 1.4,
            "fill_gamma": cfg.fill_gamma_flow * 0.7,
            "fill_risk_buffer_bps": cfg.fill_risk_buffer_bps * 2.0,
        }
    return {
        "impact_coeff": cfg.impact_coeff,
        "max_impact": cfg.max_impact,
        "vol_slippage_coeff": cfg.vol_slippage_coeff,
        "fill_alpha": cfg.fill_alpha_liquidity,
        "fill_beta": cfg.fill_beta_spread,
        "fill_gamma": cfg.fill_gamma_flow,
        "fill_risk_buffer_bps": cfg.fill_risk_buffer_bps,
    }


def _drawdown_duration(equity: pd.Series) -> int:
    if equity.empty:
        return 0
    peaks = equity.cummax()
    under = equity < peaks
    cur = 0
    longest = 0
    for flag in under.astype(bool).tolist():
        if flag:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0
    return int(longest)


def _empty_summary(horizon: int, regime: str) -> dict[str, float | str]:
    return {
        "holding_period": float(horizon),
        "trade_count": 0.0,
        "win_rate": 0.0,
        "expectancy": 0.0,
        "expectancy_per_trade": 0.0,
        "total_pnl": 0.0,
        "sharpe": 0.0,
        "max_drawdown": 0.0,
        "drawdown_duration": 0.0,
        "ret_p10": 0.0,
        "ret_p50": 0.0,
        "ret_p90": 0.0,
        "avg_expected_fill_pct": 0.0,
        "avg_realized_fill_pct": 0.0,
        "avg_actual_fill_proxy": 0.0,
        "avg_fill_calibration_error": 0.0,
        "avg_time_to_fill_h": 0.0,
        "avg_realized_slippage_bps": 0.0,
        "avg_gross_return": 0.0,
        "avg_expected_ev": 0.0,
        "avg_ev_gap": 0.0,
        "ev_hit_rate": 0.0,
        "gross_pnl_total": 0.0,
        "fees_total": 0.0,
        "spread_cost_total": 0.0,
        "impact_cost_total": 0.0,
        "vol_slippage_cost_total": 0.0,
        "observed_cost_bps": 0.0,
        "breakeven_cost_bps": 0.0,
        "execution_regime": regime,
        "return_per_day": 0.0,
        "return_per_event": 0.0,
        "n_days": 0.0,
        "n_events": 0.0,
        "hedged_fill_rate": 0.0,
        "fully_hedged_trade_count": 0.0,
        "avg_pair_fill_pct": 0.0,
        "avg_expected_pair_fill": 0.0,
    }


def _simulate_paired_one_horizon(
    signal_frame: pd.DataFrame,
    *,
    horizon: int,
    threshold: float,
    cfg: PairedBacktestConfig,
) -> tuple[pd.DataFrame, dict[str, float | str]]:
    regime = str(cfg.execution_regime).lower().strip()
    p = _regime_params(cfg)
    required = {
        "ts",
        "market_id",
        "signal",
        "confidence",
        "ask_yes",
        "ask_no",
        "bid_yes",
        "bid_no",
        "spread_yes",
        "spread_no",
        "depth_yes",
        "depth_no",
        f"future_bid_yes_{horizon}",
        f"future_ask_yes_{horizon}",
        f"future_bid_no_{horizon}",
        f"future_ask_no_{horizon}",
    }
    if signal_frame.empty or not required.issubset(signal_frame.columns):
        return pd.DataFrame(), _empty_summary(horizon, regime)

    df = signal_frame.copy()
    df["pair_staleness_ms"] = pd.to_numeric(df.get("pair_staleness_ms", 0.0), errors="coerce").fillna(0.0)
    df = df[df["pair_staleness_ms"] <= float(cfg.staleness_tolerance_ms)].copy()
    sig = pd.to_numeric(df["signal"], errors="coerce").fillna(0.0)
    df = df[sig.abs() >= float(threshold)].copy()
    if df.empty:
        return pd.DataFrame(), _empty_summary(horizon, regime)

    needed_future = [f"future_bid_yes_{horizon}", f"future_ask_yes_{horizon}", f"future_bid_no_{horizon}", f"future_ask_no_{horizon}"]
    for col in needed_future:
        df = df[pd.to_numeric(df[col], errors="coerce").notna()]
    if df.empty:
        return pd.DataFrame(), _empty_summary(horizon, regime)

    sign = np.sign(pd.to_numeric(df["signal"], errors="coerce").fillna(0.0))
    is_long_set = sign > 0
    pair_direction = pd.Series(np.where(is_long_set, "LONG_SET", "SHORT_SET"), index=df.index, dtype=object)
    direction = pd.Series(np.where(is_long_set, DIRECTION_LONG_YES, DIRECTION_LONG_NO), index=df.index, dtype=object)

    conf = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.5).clip(lower=0.05, upper=1.0)
    ask_yes = pd.to_numeric(df["ask_yes"], errors="coerce").clip(lower=1e-6, upper=0.999)
    ask_no = pd.to_numeric(df["ask_no"], errors="coerce").clip(lower=1e-6, upper=0.999)
    bid_yes = pd.to_numeric(df["bid_yes"], errors="coerce").clip(lower=1e-6, upper=0.999)
    bid_no = pd.to_numeric(df["bid_no"], errors="coerce").clip(lower=1e-6, upper=0.999)
    spread_yes = pd.to_numeric(df["spread_yes"], errors="coerce").fillna(0.0).clip(lower=0.0)
    spread_no = pd.to_numeric(df["spread_no"], errors="coerce").fillna(0.0).clip(lower=0.0)
    depth_yes = pd.to_numeric(df["depth_yes"], errors="coerce").fillna(0.0).clip(lower=0.0)
    depth_no = pd.to_numeric(df["depth_no"], errors="coerce").fillna(0.0).clip(lower=0.0)

    side_depth_yes = np.where(is_long_set, depth_yes, depth_yes)
    side_depth_no = np.where(is_long_set, depth_no, depth_no)
    pair_depth = pd.Series(np.minimum(side_depth_yes, side_depth_no), index=df.index).clip(lower=1.0)
    target_qty = np.minimum(cfg.base_trade_notional * conf, cfg.max_participation * pair_depth).clip(lower=1.0)
    notional = target_qty.copy()  # $1 collateral per paired contract

    spread_penalty = ((spread_yes + spread_no) * 10_000.0).clip(lower=0.0)
    flow_proxy = pd.to_numeric(df.get("trade_frequency_pair_h", 0.0), errors="coerce").fillna(0.0)
    size_ratio = (target_qty / pair_depth).clip(lower=0.0)
    leg_ratio_yes = (target_qty / depth_yes.clip(lower=1.0)).clip(lower=0.0)
    leg_ratio_no = (target_qty / depth_no.clip(lower=1.0)).clip(lower=0.0)
    fill_logit_yes = p["fill_alpha"] - 2.0 * leg_ratio_yes - p["fill_beta"] * (spread_penalty / 200.0) + p["fill_gamma"] * np.log1p(flow_proxy)
    fill_logit_no = p["fill_alpha"] - 2.0 * leg_ratio_no - p["fill_beta"] * (spread_penalty / 200.0) + p["fill_gamma"] * np.log1p(flow_proxy)
    fill_prob_yes = pd.Series(1.0 / (1.0 + np.exp(-np.clip(fill_logit_yes, -20.0, 20.0))), index=df.index).clip(lower=0.0, upper=1.0)
    fill_prob_no = pd.Series(1.0 / (1.0 + np.exp(-np.clip(fill_logit_no, -20.0, 20.0))), index=df.index).clip(lower=0.0, upper=1.0)
    expected_pair_fill = (fill_prob_yes * fill_prob_no).clip(lower=0.0, upper=1.0)
    min_fill_prob_pair = np.minimum(fill_prob_yes, fill_prob_no)

    queue_penalty = (1.0 - conf) * (0.25 + size_ratio)
    realized_fill_yes = (fill_prob_yes * (1.0 - 0.5 * queue_penalty)).clip(lower=0.0, upper=1.0)
    realized_fill_no = (fill_prob_no * (1.0 - 0.5 * queue_penalty)).clip(lower=0.0, upper=1.0)
    q_yes = target_qty * realized_fill_yes
    q_no = target_qty * realized_fill_no
    q_pair = np.minimum(q_yes, q_no)
    q_unhedged_yes = (q_yes - q_pair).clip(lower=0.0)
    q_unhedged_no = (q_no - q_pair).clip(lower=0.0)
    realized_pair_fill_pct = (q_pair / target_qty).clip(lower=0.0, upper=1.0)
    realized_fill_pct = (0.5 * (realized_fill_yes + realized_fill_no)).clip(lower=0.0, upper=1.0)
    fully_hedged = (realized_pair_fill_pct >= 0.95).astype(float)
    expected_fill_pct = expected_pair_fill.astype(float)
    actual_fill_proxy = np.minimum(realized_fill_yes, realized_fill_no).astype(float)

    long_edge = (1.0 - (ask_yes + ask_no)).astype(float)
    short_edge = ((bid_yes + bid_no) - 1.0).astype(float)
    locked_edge = pd.Series(np.where(is_long_set, long_edge, short_edge), index=df.index, dtype=float)
    gross_pair_pnl = q_pair * locked_edge

    fby = pd.to_numeric(df[f"future_bid_yes_{horizon}"], errors="coerce").clip(lower=1e-6, upper=0.999)
    fay = pd.to_numeric(df[f"future_ask_yes_{horizon}"], errors="coerce").clip(lower=1e-6, upper=0.999)
    fbn = pd.to_numeric(df[f"future_bid_no_{horizon}"], errors="coerce").clip(lower=1e-6, upper=0.999)
    fan = pd.to_numeric(df[f"future_ask_no_{horizon}"], errors="coerce").clip(lower=1e-6, upper=0.999)
    future_spread_yes = (fay - fby).clip(lower=0.0)
    future_spread_no = (fan - fbn).clip(lower=0.0)

    unhedged_yes_pnl = pd.Series(np.where(is_long_set, (fby - ask_yes), (bid_yes - fay)), index=df.index) * q_unhedged_yes
    unhedged_no_pnl = pd.Series(np.where(is_long_set, (fbn - ask_no), (bid_no - fan)), index=df.index) * q_unhedged_no

    fee_rate = cfg.fee_bps / 10_000.0
    # Entry notionals are only for executed quantities on each leg.
    entry_notional_yes = pd.Series(np.where(is_long_set, ask_yes, bid_yes), index=df.index) * q_yes
    entry_notional_no = pd.Series(np.where(is_long_set, ask_no, bid_no), index=df.index) * q_no
    unwind_notional_yes = pd.Series(np.where(is_long_set, fby, fay), index=df.index) * q_unhedged_yes
    unwind_notional_no = pd.Series(np.where(is_long_set, fbn, fan), index=df.index) * q_unhedged_no
    fees = fee_rate * (entry_notional_yes + entry_notional_no + unwind_notional_yes + unwind_notional_no)

    pair_spread_cost = 0.0 * q_pair  # entry spread is already baked into bid/ask parity edge
    unwind_spread_cost = 0.5 * future_spread_yes * q_unhedged_yes + 0.5 * future_spread_no * q_unhedged_no
    spread_component = pair_spread_cost + unwind_spread_cost

    pair_vol = (
        pd.to_numeric(df.get("pair_volatility", 0.0), errors="coerce").fillna(0.0).abs()
        if "pair_volatility" in df.columns
        else pd.to_numeric(df.get("prob_vol_pair", 0.0), errors="coerce").fillna(0.0).abs()
    )
    impact_rate = np.minimum(p["max_impact"], p["impact_coeff"] * size_ratio)
    vol_slippage_rate = p["vol_slippage_coeff"] * pair_vol * (1.0 + size_ratio)
    impact_component = (entry_notional_yes + entry_notional_no + unwind_notional_yes + unwind_notional_no) * impact_rate
    vol_slippage_component = (entry_notional_yes + entry_notional_no + unwind_notional_yes + unwind_notional_no) * vol_slippage_rate

    unwind_buffer = (p["fill_risk_buffer_bps"] / 10_000.0) * (q_unhedged_yes + q_unhedged_no)
    gross_pnl = gross_pair_pnl + unhedged_yes_pnl + unhedged_no_pnl
    net_pnl = gross_pnl - fees - spread_component - impact_component - vol_slippage_component - unwind_buffer
    trade_return = (net_pnl / notional.clip(lower=1.0)).astype(float)
    gross_return = (gross_pnl / notional.clip(lower=1.0)).astype(float)

    expected_fee_return = fee_rate * (
        pd.Series(np.where(is_long_set, ask_yes + ask_no, bid_yes + bid_no), index=df.index)
        * expected_pair_fill
    )
    expected_spread_cost_return = (p["fill_risk_buffer_bps"] / 10_000.0) * (1.0 - expected_pair_fill)
    expected_impact_return = (impact_rate * np.minimum(fill_prob_yes, fill_prob_no)).astype(float)
    expected_vol_slippage_return = (vol_slippage_rate * np.minimum(fill_prob_yes, fill_prob_no)).astype(float)
    expected_cost_return = (
        expected_fee_return + expected_spread_cost_return + expected_impact_return + expected_vol_slippage_return
    ).astype(float)
    expected_gross_return = (locked_edge * expected_pair_fill).astype(float)
    expected_ev = (expected_gross_return - expected_cost_return).astype(float)
    ev_gap = trade_return - expected_ev

    pair_mid = ((ask_yes + ask_no + bid_yes + bid_no) / 2.0).clip(lower=1e-6, upper=0.999999)
    future_pair_mid = ((fby + fbn + fay + fan) / 2.0).clip(lower=1e-6, upper=0.999999)
    forward_ret_1 = (future_pair_mid - pair_mid).fillna(0.0)

    pair_fill_prob = expected_pair_fill.astype(float)
    hedging_delay_proxy = (1.0 - realized_pair_fill_pct).clip(lower=0.0, upper=1.0) * float(horizon)

    trades = df.copy()
    trades["paired_trade_id"] = (
        pd.to_datetime(trades["ts"], errors="coerce").astype(str)
        + "|"
        + trades["market_id"].astype(str)
        + "|hp="
        + str(int(horizon))
    )
    trades["holding_period"] = float(horizon)
    trades["direction"] = direction.astype(str)
    trades["pair_direction"] = pair_direction.astype(str)
    trades["direction_sign"] = sign.astype(float)
    trades["traded_side"] = "BASKET"
    trades["position_type"] = np.where(is_long_set, "long_complement_set", "short_complement_set")
    trades["prediction_space"] = PredictionSpace.RETURN_NEXT_H.value
    trades["model_target_type"] = PredictionSpace.RETURN_NEXT_H.value
    trades["model_horizon"] = float(horizon)
    trades["horizon"] = float(horizon)
    trades["mid"] = pair_mid.astype(float)
    trades["future_mid_1"] = future_pair_mid.astype(float)
    trades["forward_ret_1"] = forward_ret_1.astype(float)
    trades["book_tradable"] = 1.0
    trades["depth_total"] = pair_depth.astype(float)
    trades["notional"] = notional.astype(float)
    trades["target_notional"] = target_qty.astype(float)
    trades["required_notional"] = notional.astype(float)
    trades["capacity_estimate"] = (cfg.max_participation * pair_depth).astype(float)
    trades["qty_target_pair"] = target_qty.astype(float)
    trades["qty_yes_filled"] = q_yes.astype(float)
    trades["qty_no_filled"] = q_no.astype(float)
    trades["qty_pair_hedged"] = q_pair.astype(float)
    trades["qty_unhedged_yes"] = q_unhedged_yes.astype(float)
    trades["qty_unhedged_no"] = q_unhedged_no.astype(float)
    trades["leg_fill_pct_yes"] = realized_fill_yes.astype(float)
    trades["leg_fill_pct_no"] = realized_fill_no.astype(float)
    trades["fill_prob_yes"] = fill_prob_yes.astype(float)
    trades["fill_prob_no"] = fill_prob_no.astype(float)
    trades["min_fill_prob_pair"] = min_fill_prob_pair.astype(float)
    trades["expected_pair_fill"] = pair_fill_prob
    trades["expected_fill_pct"] = expected_fill_pct.astype(float)
    trades["realized_fill_pct"] = realized_fill_pct.astype(float)
    trades["actual_fill_proxy"] = actual_fill_proxy.astype(float)
    trades["actual_pair_fill_proxy"] = actual_fill_proxy.astype(float)
    trades["realized_pair_fill_pct"] = realized_pair_fill_pct.astype(float)
    trades["fully_hedged"] = fully_hedged.astype(float)
    trades["fully_hedged_pair"] = fully_hedged.astype(float)
    trades["time_to_fill_h"] = (1.0 - pair_fill_prob).clip(lower=0.0, upper=1.0) * float(horizon)
    trades["hedging_delay_proxy"] = hedging_delay_proxy.astype(float)
    trades["pair_staleness_ms"] = trades["pair_staleness_ms"].astype(float)
    trades["pair_staleness_penalty_return"] = (trades["pair_staleness_ms"] / 1000.0) * 1e-4
    trades["execution_regime"] = regime
    trades["gross_pnl"] = gross_pnl.astype(float)
    trades["gross_return"] = gross_return.astype(float)
    trades["fees"] = fees.astype(float)
    trades["spread_component"] = spread_component.astype(float)
    trades["impact_component"] = impact_component.astype(float)
    trades["vol_slippage_component"] = vol_slippage_component.astype(float)
    trades["unwind_penalty_component"] = unwind_buffer.astype(float)
    trades["net_pnl"] = net_pnl.astype(float)
    trades["trade_return"] = trade_return.astype(float)
    trades["expected_gross"] = expected_gross_return.astype(float)
    trades["expected_gross_return"] = expected_gross_return.astype(float)
    trades["expected_cost"] = expected_cost_return.astype(float)
    trades["expected_cost_return"] = expected_cost_return.astype(float)
    trades["expected_fee_return"] = expected_fee_return.astype(float)
    trades["expected_spread_cost_return"] = expected_spread_cost_return.astype(float)
    trades["expected_impact_return"] = expected_impact_return.astype(float)
    trades["expected_vol_slippage_return"] = expected_vol_slippage_return.astype(float)
    trades["expected_net_ev"] = expected_ev.astype(float)
    trades["expected_ev"] = expected_ev.astype(float)
    trades["expected_return"] = expected_ev.astype(float)
    trades["ev_gap"] = ev_gap.astype(float)
    trades = attach_prediction_contract(trades, expected_return_col="expected_return")

    pnl_ts = trades.groupby("ts", observed=True)["net_pnl"].sum().sort_index().cumsum()
    equity = 100_000.0 + pnl_ts
    ret_ts = pnl_ts.diff().fillna(0.0) / 100_000.0
    cost_return = (gross_return - trade_return).fillna(0.0)
    summary = _empty_summary(horizon, regime)
    summary.update(
        {
            "trade_count": float(len(trades)),
            "win_rate": float((trades["trade_return"] > 0).mean()),
            "expectancy": float(trades["trade_return"].mean()),
            "expectancy_per_trade": float(trades["trade_return"].mean()),
            "total_pnl": float(trades["net_pnl"].sum()),
            "sharpe": sharpe_ratio(ret_ts, periods_per_year=24 * 365),
            "max_drawdown": max_drawdown(equity),
            "drawdown_duration": float(_drawdown_duration(equity)),
            "ret_p10": float(trades["trade_return"].quantile(0.10)),
            "ret_p50": float(trades["trade_return"].quantile(0.50)),
            "ret_p90": float(trades["trade_return"].quantile(0.90)),
            "avg_expected_fill_pct": float(trades["expected_fill_pct"].mean()),
            "avg_realized_fill_pct": float(trades["realized_fill_pct"].mean()),
            "avg_actual_fill_proxy": float(trades["actual_fill_proxy"].mean()),
            "avg_fill_calibration_error": float((trades["expected_fill_pct"] - trades["actual_fill_proxy"]).mean()),
            "avg_time_to_fill_h": float(trades["time_to_fill_h"].mean()),
            "avg_realized_slippage_bps": float((trades["spread_component"] / trades["notional"].clip(lower=1.0)).mean() * 10_000.0),
            "avg_gross_return": float(trades["gross_return"].mean()),
            "avg_expected_ev": float(trades["expected_ev"].mean()),
            "avg_ev_gap": float(trades["ev_gap"].mean()),
            "ev_hit_rate": float((trades.loc[trades["expected_ev"] > 0.0, "trade_return"] > 0.0).mean()) if float((trades["expected_ev"] > 0.0).sum()) > 0 else 0.0,
            "gross_pnl_total": float(trades["gross_pnl"].sum()),
            "fees_total": float(trades["fees"].sum()),
            "spread_cost_total": float(trades["spread_component"].sum()),
            "impact_cost_total": float(trades["impact_component"].sum()),
            "vol_slippage_cost_total": float(trades["vol_slippage_component"].sum()),
            "observed_cost_bps": float(cost_return.mean() * 10_000.0),
            "breakeven_cost_bps": float(max(0.0, trades["gross_return"].mean()) * 10_000.0),
            "hedged_fill_rate": float(trades["fully_hedged"].mean()),
            "fully_hedged_trade_count": float(trades["fully_hedged"].sum()),
            "avg_pair_fill_pct": float(trades["realized_pair_fill_pct"].mean()),
            "avg_expected_pair_fill": float(trades["expected_pair_fill"].mean()),
        }
    )
    if "ts" in trades.columns and len(trades):
        t = trades.copy()
        t["day"] = pd.to_datetime(t["ts"], errors="coerce").dt.floor("D")
        day_ret = t.groupby("day", observed=True)["net_pnl"].sum() / 100_000.0
        event_ret = t.groupby("market_id", observed=True)["net_pnl"].sum() / 100_000.0
        summary["return_per_day"] = float(day_ret.mean()) if len(day_ret) else 0.0
        summary["return_per_event"] = float(event_ret.mean()) if len(event_ret) else 0.0
        summary["n_days"] = float(len(day_ret))
        summary["n_events"] = float(len(event_ret))
    return trades, summary


def run_paired_backtest_grid(
    signal_frame: pd.DataFrame,
    thresholds: list[float],
    cfg: PairedBacktestConfig = PairedBacktestConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_trades: list[pd.DataFrame] = []
    rows: list[dict[str, float | str]] = []
    for thr in thresholds:
        for hp in cfg.holding_periods:
            trades, summary = _simulate_paired_one_horizon(signal_frame, horizon=int(hp), threshold=float(thr), cfg=cfg)
            summary["threshold"] = float(thr)
            rows.append(summary)
            if not trades.empty:
                trades["threshold"] = float(thr)
                all_trades.append(trades)
    return pd.DataFrame(rows), (pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame())


def calibrate_pair_execution(trades: pd.DataFrame, *, n_bins: int = 10) -> pd.DataFrame:
    if trades.empty or "expected_pair_fill" not in trades.columns or "realized_pair_fill_pct" not in trades.columns:
        return pd.DataFrame()
    t = trades.copy()
    exp = pd.to_numeric(t["expected_pair_fill"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    act = pd.to_numeric(t["realized_pair_fill_pct"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    if exp.nunique() <= 1:
        bins = pd.Series(0, index=t.index)
    else:
        bins = pd.qcut(exp.rank(method="first"), q=min(n_bins, exp.nunique()), labels=False, duplicates="drop")
    t["fill_bin"] = bins
    rows: list[dict[str, float | str]] = []
    for b, g in t.groupby("fill_bin", observed=True):
        e = pd.to_numeric(g["expected_pair_fill"], errors="coerce").fillna(0.0)
        a = pd.to_numeric(g["realized_pair_fill_pct"], errors="coerce").fillna(0.0)
        rows.append(
            {
                "fill_bin": float(b),
                "n": float(len(g)),
                "avg_expected_fill_pct": float(e.mean()),
                "avg_actual_fill_proxy": float(a.mean()),
                "fill_gap": float((e - a).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("fill_bin")


def run_paired_execution_regime_sensitivity(
    signal_frame: pd.DataFrame,
    *,
    threshold: float,
    holding_period: int,
    cfg: PairedBacktestConfig,
    regimes: tuple[str, ...] = ("optimistic", "base", "pessimistic"),
) -> dict[str, pd.DataFrame]:
    sens_rows: list[dict[str, float | str]] = []
    cal_rows: list[pd.DataFrame] = []
    for regime in regimes:
        cur = PairedBacktestConfig(
            holding_periods=(int(holding_period),),
            fee_bps=cfg.fee_bps,
            impact_coeff=cfg.impact_coeff,
            max_impact=cfg.max_impact,
            base_trade_notional=cfg.base_trade_notional,
            max_participation=cfg.max_participation,
            vol_slippage_coeff=cfg.vol_slippage_coeff,
            fill_alpha_liquidity=cfg.fill_alpha_liquidity,
            fill_beta_spread=cfg.fill_beta_spread,
            fill_gamma_flow=cfg.fill_gamma_flow,
            fill_risk_buffer_bps=cfg.fill_risk_buffer_bps,
            staleness_tolerance_ms=cfg.staleness_tolerance_ms,
            execution_regime=str(regime),
            random_seed=cfg.random_seed,
        )
        summary, trades = run_paired_backtest_grid(signal_frame, thresholds=[float(threshold)], cfg=cur)
        row = summary[
            (pd.to_numeric(summary.get("threshold"), errors="coerce") == float(threshold))
            & (pd.to_numeric(summary.get("holding_period"), errors="coerce") == float(holding_period))
        ]
        if row.empty:
            sens_rows.append(
                {
                    "analysis_type": "execution_regime",
                    "execution_regime": str(regime),
                    "trade_count": 0.0,
                    "expectancy": 0.0,
                    "sharpe": 0.0,
                    "total_pnl": 0.0,
                    "avg_expected_ev": 0.0,
                    "observed_cost_bps": 0.0,
                    "breakeven_cost_bps": 0.0,
                    "cost_headroom_bps": 0.0,
                    "hedged_fill_rate": 0.0,
                }
            )
            continue
        d = row.iloc[0].to_dict()
        d["analysis_type"] = "execution_regime"
        d["execution_regime"] = str(regime)
        d["cost_headroom_bps"] = float(d.get("breakeven_cost_bps", 0.0) - d.get("observed_cost_bps", 0.0))
        sens_rows.append(d)
        cal = calibrate_pair_execution(trades)
        if not cal.empty:
            cal["execution_regime"] = str(regime)
            cal_rows.append(cal)
    return {
        "sensitivity": pd.DataFrame(sens_rows),
        "calibration": (pd.concat(cal_rows, ignore_index=True) if cal_rows else pd.DataFrame()),
    }


def run_paired_impact_coefficient_sensitivity(
    signal_frame: pd.DataFrame,
    *,
    threshold: float,
    holding_period: int,
    cfg: PairedBacktestConfig,
    impact_coeff_grid: tuple[float, ...] = (0.02, 0.05, 0.08, 0.10, 0.15, 0.20),
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for coeff in impact_coeff_grid:
        cur = PairedBacktestConfig(
            holding_periods=(int(holding_period),),
            fee_bps=cfg.fee_bps,
            impact_coeff=float(coeff),
            max_impact=cfg.max_impact,
            base_trade_notional=cfg.base_trade_notional,
            max_participation=cfg.max_participation,
            vol_slippage_coeff=cfg.vol_slippage_coeff,
            fill_alpha_liquidity=cfg.fill_alpha_liquidity,
            fill_beta_spread=cfg.fill_beta_spread,
            fill_gamma_flow=cfg.fill_gamma_flow,
            fill_risk_buffer_bps=cfg.fill_risk_buffer_bps,
            staleness_tolerance_ms=cfg.staleness_tolerance_ms,
            execution_regime=cfg.execution_regime,
            random_seed=cfg.random_seed,
        )
        summary, _ = run_paired_backtest_grid(signal_frame, thresholds=[float(threshold)], cfg=cur)
        row = summary[
            (pd.to_numeric(summary.get("threshold"), errors="coerce") == float(threshold))
            & (pd.to_numeric(summary.get("holding_period"), errors="coerce") == float(holding_period))
        ]
        if row.empty:
            rows.append(
                {
                    "analysis_type": "impact_coeff_grid",
                    "execution_regime": str(cfg.execution_regime).lower().strip(),
                    "impact_coeff_input": float(coeff),
                    "trade_count": 0.0,
                    "expectancy": 0.0,
                    "avg_expected_ev": 0.0,
                    "observed_cost_bps": 0.0,
                    "breakeven_cost_bps": 0.0,
                    "cost_headroom_bps": 0.0,
                }
            )
            continue
        d = row.iloc[0].to_dict()
        rows.append(
            {
                **d,
                "analysis_type": "impact_coeff_grid",
                "execution_regime": str(cfg.execution_regime).lower().strip(),
                "impact_coeff_input": float(coeff),
                "cost_headroom_bps": float(d.get("breakeven_cost_bps", 0.0) - d.get("observed_cost_bps", 0.0)),
            }
        )
    return pd.DataFrame(rows)
