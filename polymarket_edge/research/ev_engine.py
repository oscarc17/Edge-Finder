from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from polymarket_edge.research.direction import (
    DIRECTION_LONG_YES,
    direction_label_series,
    direction_sign_series,
)
from polymarket_edge.research.semantics import PredictionSpace, attach_prediction_contract, normalize_prediction_space_series


@dataclass(frozen=True)
class EVEngineConfig:
    fee_bps: float = 20.0
    impact_coeff: float = 0.10
    max_impact: float = 0.05
    base_trade_notional: float = 250.0
    max_participation: float = 0.15
    vol_slippage_coeff: float = 0.25
    default_horizon: int = 1


def _safe_series(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in frame.columns:
        return pd.to_numeric(frame[col], errors="coerce").fillna(default)
    return pd.Series(default, index=frame.index, dtype=float)


def compute_expected_cost_components(
    frame: pd.DataFrame,
    *,
    cfg: EVEngineConfig,
    entry_price: pd.Series | None = None,
) -> dict[str, pd.Series]:
    idx = frame.index
    spread = _safe_series(frame, "spread", 0.0).clip(lower=0.0)
    depth_total = _safe_series(frame, "depth_total", 1.0).clip(lower=1.0)
    confidence = _safe_series(frame, "confidence", 0.5).clip(lower=0.05, upper=1.0)
    prob_vol = _safe_series(frame, "prob_vol_6", 0.0).abs()
    future_spread = _safe_series(frame, f"future_spread_{int(cfg.default_horizon)}", np.nan)
    if future_spread.isna().all():
        future_spread = _safe_series(frame, "future_spread_1", np.nan)
    future_spread = future_spread.fillna(spread).clip(lower=0.0)

    if entry_price is None:
        signal = _safe_series(frame, "signal", 0.0)
        mid_yes = _safe_series(frame, "mid", 0.5).clip(lower=1e-6, upper=1 - 1e-6)
        dlab = direction_label_series(signal, threshold=0.0)
        is_long_yes = dlab == DIRECTION_LONG_YES
        entry_price = pd.Series(np.where(is_long_yes, mid_yes, 1.0 - mid_yes), index=idx).clip(lower=1e-6, upper=1 - 1e-6)
    entry_price = pd.to_numeric(entry_price, errors="coerce").fillna(0.5).clip(lower=1e-6, upper=1.0)

    target_notional = np.minimum(
        float(cfg.base_trade_notional) * confidence,
        float(cfg.max_participation) * depth_total * entry_price,
    ).clip(lower=1.0)
    size_ratio = target_notional / (depth_total * entry_price + 1e-9)
    impact = np.minimum(float(cfg.max_impact), float(cfg.impact_coeff) * (target_notional / (depth_total * entry_price + 1e-9)))
    vol_slippage = float(cfg.vol_slippage_coeff) * prob_vol * (1.0 + size_ratio)
    fee_rate = float(cfg.fee_bps) / 10_000.0
    fee_return = pd.Series(2.0 * fee_rate, index=idx, dtype=float)
    spread_cost_return = ((0.5 * spread) + (0.5 * future_spread)) / entry_price
    impact_return = ((impact + impact) / entry_price).astype(float)
    vol_slippage_return = ((vol_slippage + vol_slippage) / entry_price).astype(float)
    expected_cost_return = (fee_return + spread_cost_return + impact_return + vol_slippage_return).astype(float)
    return {
        "entry_price": entry_price.astype(float),
        "target_notional": target_notional.astype(float),
        "impact": pd.Series(impact, index=idx, dtype=float),
        "vol_slippage": pd.Series(vol_slippage, index=idx, dtype=float),
        "expected_fee_return": fee_return.astype(float),
        "expected_spread_cost_return": spread_cost_return.astype(float),
        "expected_impact_return": impact_return.astype(float),
        "expected_vol_slippage_return": vol_slippage_return.astype(float),
        "expected_cost_return": expected_cost_return.astype(float),
    }


def apply_ev_engine(
    frame: pd.DataFrame,
    *,
    cfg: EVEngineConfig,
    prediction_space_col: str = "prediction_space",
    model_target_type_col: str = "model_target_type",
    signal_col: str = "signal",
    mid_col: str = "mid",
    horizon_col: str = "model_horizon",
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    signal = _safe_series(out, signal_col, 0.0)
    direction_label = direction_label_series(signal, threshold=0.0)
    direction_sign = direction_sign_series(signal, threshold=0.0).astype(float)
    mid_yes = _safe_series(out, mid_col, 0.5).clip(lower=1e-6, upper=1 - 1e-6)
    is_long_yes = direction_label == DIRECTION_LONG_YES
    default_space = PredictionSpace.FUTURE_MID_YES if "model_pred_future_mid" in out.columns else PredictionSpace.UP_MOVE_PROB
    if prediction_space_col in out.columns:
        pred_space = normalize_prediction_space_series(out[prediction_space_col], default=default_space)
    elif model_target_type_col in out.columns:
        pred_space = normalize_prediction_space_series(out[model_target_type_col], default=default_space)
    else:
        pred_space = pd.Series(default_space.value, index=out.index, dtype=str)
    out[prediction_space_col] = pred_space.astype(str)

    horizon = _safe_series(out, horizon_col, float(cfg.default_horizon)).fillna(float(cfg.default_horizon))
    out["horizon"] = horizon.astype(float)
    entry_price = pd.Series(np.where(is_long_yes, mid_yes, 1.0 - mid_yes), index=out.index).clip(lower=1e-6, upper=1 - 1e-6)
    costs = compute_expected_cost_components(out, cfg=cfg, entry_price=entry_price)
    for k, v in costs.items():
        out[k] = v

    model_outcome_yes = _safe_series(out, "model_pred_outcome_prob_yes", np.nan)
    if model_outcome_yes.isna().all() and "model_proba" in out.columns:
        model_outcome_yes = _safe_series(out, "model_proba", np.nan)
    if model_outcome_yes.isna().all():
        conf = _safe_series(out, "confidence", 0.5).clip(0.05, 1.0)
        headroom = np.minimum(mid_yes, 1.0 - mid_yes)
        model_outcome_yes = (mid_yes + signal.clip(-1.0, 1.0) * (0.10 * headroom * conf)).clip(lower=1e-6, upper=1 - 1e-6)
    else:
        model_outcome_yes = model_outcome_yes.fillna(mid_yes).clip(lower=1e-6, upper=1 - 1e-6)
    model_prob_dir = pd.Series(np.where(is_long_yes, model_outcome_yes, 1.0 - model_outcome_yes), index=out.index).clip(lower=1e-6, upper=1 - 1e-6)
    market_prob_dir = pd.Series(np.where(is_long_yes, mid_yes, 1.0 - mid_yes), index=out.index).clip(lower=1e-6, upper=1 - 1e-6)
    expected_gross_outcome = ((model_prob_dir - market_prob_dir) / entry_price).astype(float)

    model_future_mid = _safe_series(out, "model_pred_future_mid", np.nan)
    if model_future_mid.isna().all():
        move_size = _safe_series(out, "expected_move_size", 0.01).clip(lower=1e-4, upper=0.4)
        model_future_mid = (mid_yes + signal.clip(-1.0, 1.0) * move_size).clip(lower=1e-6, upper=1 - 1e-6)
    else:
        model_future_mid = model_future_mid.fillna(mid_yes).clip(lower=1e-6, upper=1 - 1e-6)
    expected_gross_future_mid = (direction_sign * (model_future_mid - mid_yes) / entry_price).astype(float)

    model_p_up = _safe_series(out, "model_pred_up_move_proba", np.nan)
    if model_p_up.isna().all() and "model_proba" in out.columns:
        model_p_up = _safe_series(out, "model_proba", np.nan)
    if model_p_up.isna().all():
        model_p_up = ((signal.clip(-1.0, 1.0) + 1.0) / 2.0)
    model_p_up = model_p_up.fillna(0.5).clip(lower=1e-6, upper=1 - 1e-6)
    move_size = _safe_series(out, f"expected_move_size_{int(cfg.default_horizon)}", np.nan)
    if move_size.isna().all():
        move_size = _safe_series(out, "expected_move_size", 0.01)
    move_size = move_size.fillna(0.01).clip(lower=1e-4, upper=0.4)
    expected_gross_up_move = (direction_sign * (2.0 * model_p_up - 1.0) * (move_size / entry_price)).astype(float)

    model_return_next = _safe_series(out, "model_pred_return_next", np.nan)
    if model_return_next.isna().all():
        model_return_next = _safe_series(out, "model_pred_return", np.nan)
    if model_return_next.isna().all():
        expected_gross_return_next = expected_gross_future_mid.copy()
    else:
        expected_gross_return_next = (direction_sign * model_return_next.fillna(0.0)).astype(float)

    expected_gross = expected_gross_future_mid.copy()
    expected_gross = expected_gross.where(pred_space != PredictionSpace.OUTCOME_PROB_YES.value, expected_gross_outcome)
    expected_gross = expected_gross.where(pred_space != PredictionSpace.UP_MOVE_PROB.value, expected_gross_up_move)
    expected_gross = expected_gross.where(pred_space != PredictionSpace.RETURN_NEXT_H.value, expected_gross_return_next)
    expected_net = (expected_gross - out["expected_cost_return"]).astype(float)

    out["direction"] = direction_label.astype(str)
    out["direction_sign"] = direction_sign.astype(float)
    out["traded_side"] = np.where(direction_label == DIRECTION_LONG_YES, "YES", np.where(direction_sign < 0.0, "NO", "NONE"))
    out["expected_gross_return"] = expected_gross.astype(float)
    out["expected_net_return"] = expected_net.astype(float)
    out["expected_return"] = out["expected_net_return"].astype(float)
    out["expected_gross_outcome_return"] = expected_gross_outcome.astype(float)
    out["expected_gross_future_mid_return"] = expected_gross_future_mid.astype(float)
    out["expected_gross_up_move_return"] = expected_gross_up_move.astype(float)
    out["expected_gross_return_next_h"] = expected_gross_return_next.astype(float)
    out["model_pred_outcome_prob_yes"] = model_outcome_yes.astype(float)
    out["model_pred_future_mid"] = model_future_mid.astype(float)
    out["model_pred_up_move_proba"] = model_p_up.astype(float)
    out = attach_prediction_contract(
        out,
        prediction_space=None,
        horizon=float(cfg.default_horizon),
        expected_return_col="expected_return",
    )
    return out
