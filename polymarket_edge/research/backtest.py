from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from polymarket_edge.backtest.metrics import max_drawdown, sharpe_ratio
from polymarket_edge.research.direction import (
    DIRECTION_FLAT,
    DIRECTION_LONG_NO,
    DIRECTION_LONG_YES,
    direction_label_series,
    direction_sign_series,
    traded_side_series,
)
from polymarket_edge.research.semantics import attach_prediction_contract, normalize_prediction_space_series


@dataclass
class StrategyBacktestConfig:
    holding_periods: tuple[int, ...] = (1, 3, 6, 12)
    fee_bps: float = 20.0
    impact_coeff: float = 0.10
    max_impact: float = 0.05
    base_trade_notional: float = 250.0
    max_participation: float = 0.15
    vol_slippage_coeff: float = 0.25
    fill_alpha_liquidity: float = 1.5
    fill_beta_spread: float = 2.0
    fill_gamma_volume: float = 0.8
    snapshot_hours: float = 1.0
    execution_regime: str = "base"  # optimistic | base | pessimistic


def _regime_adjusted_params(cfg: StrategyBacktestConfig) -> dict[str, float]:
    regime = str(cfg.execution_regime).lower().strip()
    if regime == "optimistic":
        return {
            "impact_coeff": cfg.impact_coeff * 0.70,
            "max_impact": cfg.max_impact * 0.70,
            "vol_slippage_coeff": cfg.vol_slippage_coeff * 0.60,
            "fill_alpha_liquidity": cfg.fill_alpha_liquidity + 0.60,
            "fill_beta_spread": cfg.fill_beta_spread * 0.70,
            "fill_gamma_volume": cfg.fill_gamma_volume * 1.20,
        }
    if regime == "pessimistic":
        return {
            "impact_coeff": cfg.impact_coeff * 1.40,
            "max_impact": cfg.max_impact * 1.40,
            "vol_slippage_coeff": cfg.vol_slippage_coeff * 1.50,
            "fill_alpha_liquidity": cfg.fill_alpha_liquidity - 0.60,
            "fill_beta_spread": cfg.fill_beta_spread * 1.40,
            "fill_gamma_volume": cfg.fill_gamma_volume * 0.70,
        }
    return {
        "impact_coeff": cfg.impact_coeff,
        "max_impact": cfg.max_impact,
        "vol_slippage_coeff": cfg.vol_slippage_coeff,
        "fill_alpha_liquidity": cfg.fill_alpha_liquidity,
        "fill_beta_spread": cfg.fill_beta_spread,
        "fill_gamma_volume": cfg.fill_gamma_volume,
    }


def _drawdown_duration(equity: pd.Series) -> int:
    if equity.empty:
        return 0
    peaks = equity.cummax()
    below = equity < peaks
    longest = 0
    current = 0
    for flag in below:
        if flag:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def _heuristic_model_prob_up(mid: pd.Series, signal: pd.Series, confidence: pd.Series | None = None) -> pd.Series:
    # Heuristic signals are scores; map to bounded local shifts around market probability.
    m = pd.to_numeric(mid, errors="coerce").fillna(0.5).clip(lower=1e-6, upper=1 - 1e-6)
    s = pd.to_numeric(signal, errors="coerce").fillna(0.0).clip(lower=-1.0, upper=1.0)
    if confidence is None:
        c = pd.Series(0.5, index=m.index)
    else:
        c = pd.to_numeric(confidence, errors="coerce").fillna(0.5).clip(lower=0.05, upper=1.0)
    headroom = np.minimum(m, 1.0 - m)
    max_shift = 0.10 * headroom * c
    return (m + s * max_shift).clip(lower=1e-6, upper=1 - 1e-6)


def _simulate_one_horizon(
    signal_frame: pd.DataFrame,
    *,
    horizon: int,
    threshold: float,
    cfg: StrategyBacktestConfig,
) -> tuple[pd.DataFrame, dict[str, float]]:
    regime_params = _regime_adjusted_params(cfg)
    df = signal_frame.copy()
    needed = {"signal", "confidence", "mid", "spread", "depth_total", f"future_mid_{horizon}"}
    if not needed.issubset(df.columns):
        empty = pd.DataFrame()
        return empty, {
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
            "execution_regime": str(cfg.execution_regime).lower().strip(),
            "return_per_day": 0.0,
            "return_per_event": 0.0,
            "n_days": 0.0,
            "n_events": 0.0,
        }

    df = df.copy()
    if "book_tradable" in df.columns:
        df = df[pd.to_numeric(df["book_tradable"], errors="coerce").fillna(0.0) > 0.0]
    if "best_bid" in df.columns:
        df = df[df["best_bid"].notna()]
    if "best_ask" in df.columns:
        df = df[df["best_ask"].notna()]
    if {"mid", "depth_total"}.issubset(df.columns):
        _mid = pd.to_numeric(df["mid"], errors="coerce").fillna(0.5)
        _depth = pd.to_numeric(df["depth_total"], errors="coerce").fillna(0.0)
        extreme = (_mid >= 0.99) | (_mid <= 0.01)
        df = df[(~extreme) | (_depth >= 50.0)]
    direction_label = direction_label_series(pd.to_numeric(df["signal"], errors="coerce").fillna(0.0), threshold=float(threshold))
    df = df[direction_label != DIRECTION_FLAT].copy()
    df = df[df[f"future_mid_{horizon}"].notna()]
    if df.empty:
        empty = pd.DataFrame()
        return empty, {
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
            "execution_regime": str(cfg.execution_regime).lower().strip(),
            "return_per_day": 0.0,
            "return_per_event": 0.0,
            "n_days": 0.0,
            "n_events": 0.0,
        }

    direction_label = direction_label_series(pd.to_numeric(df["signal"], errors="coerce").fillna(0.0), threshold=0.0)
    direction = direction_sign_series(pd.to_numeric(df["signal"], errors="coerce").fillna(0.0), threshold=0.0).astype(float)
    active_mask = direction != 0.0
    if not bool(active_mask.all()):
        df = df.loc[active_mask].copy()
        direction_label = direction_label.loc[df.index]
        direction = direction.loc[df.index].astype(float)
    if df.empty:
        empty = pd.DataFrame()
        return empty, {
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
            "execution_regime": str(cfg.execution_regime).lower().strip(),
            "return_per_day": 0.0,
            "return_per_event": 0.0,
            "n_days": 0.0,
            "n_events": 0.0,
        }
    confidence = df["confidence"].clip(0.05, 1.0)
    mid_yes = pd.to_numeric(df["mid"], errors="coerce").fillna(0.5).clip(lower=1e-6, upper=1 - 1e-6)
    future_mid_yes = pd.to_numeric(df[f"future_mid_{horizon}"], errors="coerce").fillna(mid_yes).clip(lower=1e-6, upper=1 - 1e-6)
    is_long_yes = direction_label == DIRECTION_LONG_YES
    mid = pd.Series(np.where(is_long_yes, mid_yes, 1.0 - mid_yes), index=df.index).clip(lower=1e-6, upper=1 - 1e-6)
    future_mid = pd.Series(np.where(is_long_yes, future_mid_yes, 1.0 - future_mid_yes), index=df.index).clip(lower=1e-6, upper=1 - 1e-6)
    depth_total = df["depth_total"].clip(lower=1.0)
    spread = df["spread"].fillna(0.0).clip(lower=0.0)
    spread_bps = df.get("spread_bps", pd.Series(0.0, index=df.index)).fillna(0.0).clip(lower=0.0)
    recent_volume = df.get("trade_notional_h", pd.Series(0.0, index=df.index)).fillna(0.0).clip(lower=0.0)
    prob_vol = df.get("prob_vol_6", pd.Series(0.0, index=df.index)).fillna(0.0).abs()

    depth_notional_cap = cfg.max_participation * depth_total * mid
    target_notional = np.minimum(cfg.base_trade_notional * confidence, depth_notional_cap).clip(lower=1.0)
    target_qty = target_notional / mid

    # Fill probability model: larger size ratio/spreads reduce fill odds, recent volume increases.
    size_ratio = target_notional / (depth_total * mid + 1e-9)
    fill_logit = (
        regime_params["fill_alpha_liquidity"]
        - 2.5 * size_ratio
        - regime_params["fill_beta_spread"] * (spread_bps / 150.0)
        + regime_params["fill_gamma_volume"] * np.log1p(recent_volume / (target_notional + 1.0))
    )
    expected_fill_pct = 1.0 / (1.0 + np.exp(-np.clip(fill_logit, -30, 30)))

    # Queue model: lower confidence implies deeper queue placement and slower fill.
    queue_position = (1.0 - confidence).clip(0.05, 0.95)
    fill_rate = np.log1p(recent_volume + 1.0) / (depth_total + 1.0)
    time_to_fill_h = cfg.snapshot_hours * queue_position / (fill_rate + 1e-6)
    horizon_h = max(cfg.snapshot_hours, horizon * cfg.snapshot_hours)
    queue_fill_pct = np.minimum(1.0, horizon_h / (time_to_fill_h + 1e-6))
    realized_fill_pct = np.clip(expected_fill_pct * queue_fill_pct, 0.0, 1.0)
    qty = target_qty * realized_fill_pct

    # Slippage model combines spread, impact, and volatility-linked adverse selection.
    impact = np.minimum(regime_params["max_impact"], regime_params["impact_coeff"] * (target_notional / (depth_total * mid + 1e-9)))
    vol_slippage = regime_params["vol_slippage_coeff"] * prob_vol * (1.0 + size_ratio)
    entry_slippage = 0.5 * spread + impact + vol_slippage
    entry_price = (mid + entry_slippage).clip(lower=0.001, upper=0.999)
    exit_mid = future_mid.clip(lower=0.001, upper=0.999)
    future_spread = df.get(f"future_spread_{horizon}", pd.Series(0.0, index=df.index)).fillna(0.0).clip(lower=0.0)
    exit_slippage = 0.5 * future_spread + impact + vol_slippage
    exit_price = (exit_mid - exit_slippage).clip(lower=0.001, upper=0.999)

    gross_pnl = (exit_price - entry_price) * qty
    fee_rate = cfg.fee_bps / 10_000.0
    fee = fee_rate * (entry_price * qty + exit_price * qty)
    spread_component = (0.5 * spread + 0.5 * future_spread) * qty
    impact_component = (impact + impact) * qty
    vol_slippage_component = (vol_slippage + vol_slippage) * qty
    net_pnl = gross_pnl - fee
    notional = (entry_price * qty).clip(lower=1.0)
    trade_ret = net_pnl / notional.clip(lower=1.0)
    gross_return = gross_pnl / notional.clip(lower=1.0)
    realized_slippage = (entry_price - mid)
    liquidity_fill_proxy = np.minimum(1.0, np.log1p(recent_volume + 1.0) / np.log1p(target_notional + 1.0))
    spread_fill_proxy = np.exp(-spread_bps / 250.0)
    actual_fill_proxy = np.clip(liquidity_fill_proxy * spread_fill_proxy, 0.0, 1.0)

    target_type = df["model_target_type"] if "model_target_type" in df.columns else pd.Series("", index=df.index)
    target_type = target_type.astype(str).str.upper().str.strip()
    proba_target_type = df["model_proba_target_type"] if "model_proba_target_type" in df.columns else pd.Series("", index=df.index)
    proba_target_type = proba_target_type.astype(str).str.upper().str.strip()
    if "model_pred_future_mid" in df.columns:
        default_target = "FUTURE_MID"
    elif "model_pred_outcome_prob_yes" in df.columns:
        default_target = "OUTCOME_PROB"
    elif "model_proba" in df.columns:
        default_target = "UP_MOVE_PROB"
    else:
        default_target = "FUTURE_MID"
    target_type = target_type.replace("", default_target)
    model_horizon = pd.to_numeric(df["model_horizon"], errors="coerce").fillna(float(horizon)) if "model_horizon" in df.columns else pd.Series(float(horizon), index=df.index)

    if "model_pred_outcome_prob_yes" in df.columns and pd.to_numeric(df["model_pred_outcome_prob_yes"], errors="coerce").notna().sum() > 0:
        model_prob_outcome_yes = pd.to_numeric(df["model_pred_outcome_prob_yes"], errors="coerce").fillna(mid_yes).clip(lower=1e-6, upper=1 - 1e-6)
    elif (
        "model_proba" in df.columns
        and pd.to_numeric(df["model_proba"], errors="coerce").notna().sum() > 0
        and (((target_type == "OUTCOME_PROB") | (proba_target_type == "OUTCOME_PROB")).any())
    ):
        model_prob_outcome_yes = pd.to_numeric(df["model_proba"], errors="coerce").fillna(mid_yes).clip(lower=1e-6, upper=1 - 1e-6)
    elif "model_prob_up" in df.columns and pd.to_numeric(df["model_prob_up"], errors="coerce").notna().sum() > 0:
        model_prob_outcome_yes = pd.to_numeric(df["model_prob_up"], errors="coerce").fillna(mid_yes).clip(lower=1e-6, upper=1 - 1e-6)
    else:
        model_prob_outcome_yes = _heuristic_model_prob_up(
            mid_yes,
            pd.to_numeric(df["signal"], errors="coerce").fillna(0.0),
            pd.to_numeric(df.get("confidence", 0.5), errors="coerce").fillna(0.5),
        )
    market_prob_up = mid_yes
    market_prob_dir = pd.Series(np.where(is_long_yes, market_prob_up, 1.0 - market_prob_up), index=df.index).clip(lower=1e-6, upper=1 - 1e-6)
    model_prob_dir = pd.Series(np.where(is_long_yes, model_prob_outcome_yes, 1.0 - model_prob_outcome_yes), index=df.index).clip(lower=1e-6, upper=1 - 1e-6)

    if "model_pred_up_move_proba" in df.columns and pd.to_numeric(df["model_pred_up_move_proba"], errors="coerce").notna().sum() > 0:
        model_p_up = pd.to_numeric(df["model_pred_up_move_proba"], errors="coerce").fillna(np.nan)
    elif (
        "model_proba" in df.columns
        and pd.to_numeric(df["model_proba"], errors="coerce").notna().sum() > 0
        and (((target_type == "UP_MOVE_PROB") | (proba_target_type == "UP_MOVE_PROB")).any())
    ):
        model_p_up = pd.to_numeric(df["model_proba"], errors="coerce").fillna(np.nan)
    else:
        model_p_up = pd.Series(np.nan, index=df.index)
    if model_p_up.notna().sum() == 0:
        model_p_up = ((pd.to_numeric(df["signal"], errors="coerce").fillna(0.0).clip(-1.0, 1.0) + 1.0) / 2.0)
    model_p_up = model_p_up.fillna(0.5).clip(lower=1e-6, upper=1 - 1e-6)

    if "model_pred_future_mid" in df.columns and pd.to_numeric(df["model_pred_future_mid"], errors="coerce").notna().sum() > 0:
        model_future_mid_yes = pd.to_numeric(df["model_pred_future_mid"], errors="coerce").fillna(mid_yes).clip(lower=1e-6, upper=1 - 1e-6)
    else:
        fallback_move = pd.to_numeric(df.get(f"expected_move_size_{horizon}", df.get("expected_move_size", np.nan)), errors="coerce")
        if pd.to_numeric(fallback_move, errors="coerce").notna().sum() == 0:
            global_move = float((future_mid_yes - mid_yes).abs().mean()) if len(df) else 0.01
            fallback_move = pd.Series(global_move, index=df.index)
        fallback_move = pd.to_numeric(fallback_move, errors="coerce").fillna(0.01).clip(lower=1e-4, upper=0.40)
        model_future_mid_yes = (mid_yes + pd.to_numeric(df["signal"], errors="coerce").fillna(0.0).clip(-1.0, 1.0) * fallback_move).clip(lower=1e-6, upper=1 - 1e-6)

    move_size_est = pd.to_numeric(df.get(f"expected_move_size_{horizon}", df.get("expected_move_size", np.nan)), errors="coerce")
    if move_size_est.notna().sum() == 0:
        move_size_est = pd.Series(float((future_mid_yes - mid_yes).abs().mean()) if len(df) else 0.01, index=df.index)
    move_size_est = move_size_est.fillna(0.01).clip(lower=1e-4, upper=0.40)

    fee_return = (fee / notional.clip(lower=1.0)).astype(float)
    spread_cost_return = ((0.5 * spread + 0.5 * future_spread) / entry_price).astype(float)
    impact_return = ((impact + impact) / entry_price).astype(float)
    vol_slippage_return = ((vol_slippage + vol_slippage) / entry_price).astype(float)
    expected_cost = (fee_return + spread_cost_return + impact_return + vol_slippage_return).astype(float)

    expected_gross_outcome = ((model_prob_dir - market_prob_dir) / entry_price).astype(float)
    expected_gross_future_mid = (direction * (model_future_mid_yes - mid_yes) / entry_price).astype(float)
    expected_gross_up_move = (direction * (2.0 * model_p_up - 1.0) * (move_size_est / entry_price)).astype(float)
    expected_gross = pd.Series(expected_gross_up_move, index=df.index, dtype=float)
    expected_gross = expected_gross.where(target_type != "OUTCOME_PROB", expected_gross_outcome)
    expected_gross = expected_gross.where(target_type != "FUTURE_MID", expected_gross_future_mid)
    expected_ev = (expected_gross - expected_cost).astype(float)
    ev_gap = trade_ret - expected_ev

    keep_cols = ["ts", "market_id", "token_id", "category", "signal", "confidence", "mid", "spread", "depth_total"]
    future_cols = {f"future_mid_{horizon}", f"future_spread_{horizon}"}
    extra_cols = [c for c in df.columns if c not in set(keep_cols).union(future_cols) and not c.startswith("future_")]
    trades = df[keep_cols + extra_cols].copy()
    trades["holding_period"] = horizon
    trades["direction"] = direction_label.astype(str)
    trades["direction_sign"] = direction.astype(float)
    trades["traded_side"] = traded_side_series(direction_label)
    trades["position_type"] = np.where(direction_label == DIRECTION_LONG_YES, "long_yes", np.where(direction_label == DIRECTION_LONG_NO, "long_no", "flat"))
    trades["instrument_mid_entry"] = mid
    trades["instrument_mid_exit"] = future_mid
    trades["exec_mid_used"] = mid
    trades["exec_spread_used"] = spread
    trades["target_notional"] = target_notional
    trades["notional"] = notional
    trades["expected_fill_pct"] = expected_fill_pct
    trades["realized_fill_pct"] = realized_fill_pct
    trades["actual_fill_proxy"] = actual_fill_proxy
    trades["time_to_fill_h"] = time_to_fill_h
    trades["entry_price"] = entry_price
    trades["exit_price"] = exit_price
    trades["execution_vs_mid"] = entry_price - mid
    trades["realized_slippage"] = realized_slippage
    trades["execution_regime"] = str(cfg.execution_regime).lower().strip()
    trades["model_target_type"] = target_type.astype(str)
    trades["model_horizon"] = model_horizon.astype(float)
    trades["prediction_space"] = normalize_prediction_space_series(trades["model_target_type"])
    trades["horizon"] = trades["model_horizon"].astype(float)
    trades["model_prob_up"] = model_prob_outcome_yes.astype(float)
    trades["market_prob_up"] = market_prob_up.astype(float)
    trades["model_pred_outcome_prob_yes"] = model_prob_outcome_yes.astype(float)
    trades["model_pred_up_move_proba"] = model_p_up.astype(float)
    trades["model_pred_future_mid"] = model_future_mid_yes.astype(float)
    trades["expected_move_size"] = move_size_est.astype(float)
    trades["gross_pnl"] = gross_pnl
    trades["gross_return"] = gross_return
    trades["fees"] = fee
    trades["spread_component"] = spread_component
    trades["impact_component"] = impact_component
    trades["vol_slippage_component"] = vol_slippage_component
    trades["net_pnl"] = net_pnl
    trades["trade_return"] = trade_ret
    trades["expected_gross_outcome_return"] = expected_gross_outcome
    trades["expected_gross_future_mid_return"] = expected_gross_future_mid
    trades["expected_gross_up_move_return"] = expected_gross_up_move
    trades["expected_gross"] = expected_gross
    trades["expected_cost"] = expected_cost
    trades["expected_cost_return"] = expected_cost
    trades["expected_fee_return"] = fee_return
    trades["expected_spread_cost_return"] = spread_cost_return
    trades["expected_impact_return"] = impact_return
    trades["expected_vol_slippage_return"] = vol_slippage_return
    trades["expected_ev"] = expected_ev
    trades["expected_net_return"] = expected_ev.astype(float)
    trades["expected_return"] = expected_ev.astype(float)
    trades["ev_gap"] = ev_gap
    trades = attach_prediction_contract(trades, expected_return_col="expected_return")

    pnl_ts = trades.groupby("ts", observed=True)["net_pnl"].sum().sort_index().cumsum()
    equity = 100_000.0 + pnl_ts
    returns = pnl_ts.diff().fillna(0.0) / 100_000.0

    summary = {
        "holding_period": float(horizon),
        "trade_count": float(len(trades)),
        "win_rate": float((trades["trade_return"] > 0).mean()) if len(trades) else 0.0,
        "expectancy": float(trades["trade_return"].mean()) if len(trades) else 0.0,
        "expectancy_per_trade": float(trades["trade_return"].mean()) if len(trades) else 0.0,
        "total_pnl": float(trades["net_pnl"].sum()) if len(trades) else 0.0,
        "sharpe": sharpe_ratio(returns, periods_per_year=24 * 365),
        "max_drawdown": max_drawdown(equity),
        "drawdown_duration": float(_drawdown_duration(equity)),
        "ret_p10": float(trades["trade_return"].quantile(0.10)) if len(trades) else 0.0,
        "ret_p50": float(trades["trade_return"].quantile(0.50)) if len(trades) else 0.0,
        "ret_p90": float(trades["trade_return"].quantile(0.90)) if len(trades) else 0.0,
        "avg_expected_fill_pct": float(trades["expected_fill_pct"].mean()) if len(trades) else 0.0,
        "avg_realized_fill_pct": float(trades["realized_fill_pct"].mean()) if len(trades) else 0.0,
        "avg_actual_fill_proxy": float(trades["actual_fill_proxy"].mean()) if len(trades) else 0.0,
        "avg_fill_calibration_error": float((trades["expected_fill_pct"] - trades["actual_fill_proxy"]).mean()) if len(trades) else 0.0,
        "avg_time_to_fill_h": float(trades["time_to_fill_h"].mean()) if len(trades) else 0.0,
        "avg_realized_slippage_bps": float((trades["realized_slippage"].abs() * 10000.0).mean()) if len(trades) else 0.0,
        "avg_gross_return": float(trades["gross_return"].mean()) if len(trades) else 0.0,
        "avg_expected_ev": float(trades["expected_ev"].mean()) if len(trades) else 0.0,
        "avg_ev_gap": float(trades["ev_gap"].mean()) if len(trades) else 0.0,
        "ev_hit_rate": float((trades.loc[trades["expected_ev"] > 0.0, "trade_return"] > 0.0).mean()) if float((trades["expected_ev"] > 0.0).sum()) > 0.0 else 0.0,
        "gross_pnl_total": float(trades["gross_pnl"].sum()) if len(trades) else 0.0,
        "fees_total": float(trades["fees"].sum()) if len(trades) else 0.0,
        "spread_cost_total": float(trades["spread_component"].sum()) if len(trades) else 0.0,
        "impact_cost_total": float(trades["impact_component"].sum()) if len(trades) else 0.0,
        "vol_slippage_cost_total": float(trades["vol_slippage_component"].sum()) if len(trades) else 0.0,
        "observed_cost_bps": float(((trades["gross_return"] - trades["trade_return"]).mean()) * 10_000.0) if len(trades) else 0.0,
        "breakeven_cost_bps": float(max(0.0, trades["gross_return"].mean()) * 10_000.0) if len(trades) else 0.0,
        "execution_regime": str(cfg.execution_regime).lower().strip(),
    }
    if len(trades):
        day_ret = (
            trades.assign(day=pd.to_datetime(trades["ts"]).dt.floor("D"))
            .groupby("day", observed=True)["net_pnl"]
            .sum()
            / 100_000.0
        )
        event_ret = (
            trades.groupby("market_id", observed=True)["net_pnl"].sum() / 100_000.0
            if "market_id" in trades.columns
            else pd.Series(dtype=float)
        )
        summary["return_per_day"] = float(day_ret.mean()) if len(day_ret) else 0.0
        summary["return_per_event"] = float(event_ret.mean()) if len(event_ret) else 0.0
        summary["n_days"] = float(len(day_ret))
        summary["n_events"] = float(len(event_ret))
    else:
        summary["return_per_day"] = 0.0
        summary["return_per_event"] = 0.0
        summary["n_days"] = 0.0
        summary["n_events"] = 0.0
    return trades, summary


def run_backtest_grid(
    signal_frame: pd.DataFrame,
    thresholds: list[float],
    cfg: StrategyBacktestConfig = StrategyBacktestConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_trades: list[pd.DataFrame] = []
    summary_rows: list[dict[str, float]] = []
    for threshold in thresholds:
        for hp in cfg.holding_periods:
            trades, summary = _simulate_one_horizon(signal_frame, horizon=hp, threshold=float(threshold), cfg=cfg)
            summary["threshold"] = float(threshold)
            summary_rows.append(summary)
            if not trades.empty:
                trades["threshold"] = float(threshold)
                all_trades.append(trades)
    summary_df = pd.DataFrame(summary_rows)
    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    return summary_df, trades_df


def calibrate_execution_model(trades: pd.DataFrame, *, n_bins: int = 10) -> pd.DataFrame:
    needed = {"expected_fill_pct", "actual_fill_proxy"}
    if trades.empty or not needed.issubset(trades.columns):
        return pd.DataFrame()
    t = trades.copy()
    expected = pd.to_numeric(t["expected_fill_pct"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    actual = pd.to_numeric(t["actual_fill_proxy"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    if expected.nunique() <= 1:
        bins = pd.Series(0, index=t.index)
    else:
        bins = pd.qcut(expected.rank(method="first"), q=min(n_bins, expected.nunique()), labels=False, duplicates="drop")
    t["fill_bin"] = bins
    rows: list[dict[str, float]] = []
    for b, g in t.groupby("fill_bin", observed=True):
        exp = pd.to_numeric(g["expected_fill_pct"], errors="coerce").fillna(0.0)
        act = pd.to_numeric(g["actual_fill_proxy"], errors="coerce").fillna(0.0)
        rows.append(
            {
                "fill_bin": float(b),
                "n": float(len(g)),
                "avg_expected_fill_pct": float(exp.mean()),
                "avg_actual_fill_proxy": float(act.mean()),
                "fill_gap": float((exp - act).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("fill_bin")


def run_execution_regime_sensitivity(
    signal_frame: pd.DataFrame,
    *,
    threshold: float,
    holding_period: int,
    cfg: StrategyBacktestConfig,
    regimes: tuple[str, ...] = ("optimistic", "base", "pessimistic"),
) -> dict[str, pd.DataFrame]:
    summary_rows: list[dict[str, float | str]] = []
    calibration_rows: list[pd.DataFrame] = []
    for regime in regimes:
        cfg_reg = StrategyBacktestConfig(
            holding_periods=(holding_period,),
            fee_bps=cfg.fee_bps,
            impact_coeff=cfg.impact_coeff,
            max_impact=cfg.max_impact,
            base_trade_notional=cfg.base_trade_notional,
            max_participation=cfg.max_participation,
            vol_slippage_coeff=cfg.vol_slippage_coeff,
            fill_alpha_liquidity=cfg.fill_alpha_liquidity,
            fill_beta_spread=cfg.fill_beta_spread,
            fill_gamma_volume=cfg.fill_gamma_volume,
            snapshot_hours=cfg.snapshot_hours,
            execution_regime=regime,
        )
        summary, trades = run_backtest_grid(signal_frame, thresholds=[float(threshold)], cfg=cfg_reg)
        row = summary[
            (summary["threshold"] == float(threshold))
            & (summary["holding_period"] == float(holding_period))
        ]
        if row.empty:
            summary_rows.append(
                {
                    "analysis_type": "execution_regime",
                    "execution_regime": regime,
                    "impact_coeff_input": float(cfg_reg.impact_coeff),
                    "trade_count": 0.0,
                    "expectancy": 0.0,
                    "sharpe": 0.0,
                    "total_pnl": 0.0,
                    "avg_expected_ev": 0.0,
                    "observed_cost_bps": 0.0,
                    "breakeven_cost_bps": 0.0,
                    "cost_headroom_bps": 0.0,
                }
            )
            continue
        d = row.iloc[0].to_dict()
        d["analysis_type"] = "execution_regime"
        d["execution_regime"] = regime
        d["impact_coeff_input"] = float(cfg_reg.impact_coeff)
        d["cost_headroom_bps"] = float(d.get("breakeven_cost_bps", 0.0) - d.get("observed_cost_bps", 0.0))
        summary_rows.append(d)
        if not trades.empty:
            cal = calibrate_execution_model(trades, n_bins=10)
            if not cal.empty:
                cal = cal.copy()
                cal["execution_regime"] = regime
                calibration_rows.append(cal)
    sensitivity = pd.DataFrame(summary_rows)
    calibration = pd.concat(calibration_rows, ignore_index=True) if calibration_rows else pd.DataFrame()
    return {"sensitivity": sensitivity, "calibration": calibration}


def run_impact_coefficient_sensitivity(
    signal_frame: pd.DataFrame,
    *,
    threshold: float,
    holding_period: int,
    cfg: StrategyBacktestConfig,
    impact_coeff_grid: tuple[float, ...] = (0.03, 0.05, 0.08, 0.10, 0.15, 0.20),
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for coeff in impact_coeff_grid:
        cfg_cur = StrategyBacktestConfig(
            holding_periods=(holding_period,),
            fee_bps=cfg.fee_bps,
            impact_coeff=float(coeff),
            max_impact=cfg.max_impact,
            base_trade_notional=cfg.base_trade_notional,
            max_participation=cfg.max_participation,
            vol_slippage_coeff=cfg.vol_slippage_coeff,
            fill_alpha_liquidity=cfg.fill_alpha_liquidity,
            fill_beta_spread=cfg.fill_beta_spread,
            fill_gamma_volume=cfg.fill_gamma_volume,
            snapshot_hours=cfg.snapshot_hours,
            execution_regime=cfg.execution_regime,
        )
        summary, _trades = run_backtest_grid(signal_frame, thresholds=[float(threshold)], cfg=cfg_cur)
        row = summary[
            (summary["threshold"] == float(threshold))
            & (summary["holding_period"] == float(holding_period))
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
        observed = float(d.get("observed_cost_bps", 0.0))
        breakeven = float(d.get("breakeven_cost_bps", 0.0))
        rows.append(
            {
                **d,
                "analysis_type": "impact_coeff_grid",
                "execution_regime": str(cfg.execution_regime).lower().strip(),
                "impact_coeff_input": float(coeff),
                "cost_headroom_bps": float(breakeven - observed),
            }
        )
    return pd.DataFrame(rows)
