from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd

from polymarket_edge.research.direction import (
    DIRECTION_FLAT,
    DIRECTION_LONG_NO,
    DIRECTION_LONG_YES,
    direction_label_series,
)


class PredictionSpace(str, Enum):
    OUTCOME_PROB_YES = "OUTCOME_PROB_YES"
    FUTURE_MID_YES = "FUTURE_MID_YES"
    RETURN_NEXT_H = "RETURN_NEXT_H"
    UP_MOVE_PROB = "UP_MOVE_PROB"


_SPACE_ALIASES: dict[str, PredictionSpace] = {
    "OUTCOME_PROB": PredictionSpace.OUTCOME_PROB_YES,
    "OUTCOME_PROB_YES": PredictionSpace.OUTCOME_PROB_YES,
    "FUTURE_MID": PredictionSpace.FUTURE_MID_YES,
    "FUTURE_MID_YES": PredictionSpace.FUTURE_MID_YES,
    "RETURN_NEXT_H": PredictionSpace.RETURN_NEXT_H,
    "RETURN": PredictionSpace.RETURN_NEXT_H,
    "UP_MOVE_PROB": PredictionSpace.UP_MOVE_PROB,
}


def normalize_prediction_space(value: object, default: PredictionSpace = PredictionSpace.FUTURE_MID_YES) -> PredictionSpace:
    if value is None:
        return default
    key = str(value).strip().upper()
    return _SPACE_ALIASES.get(key, default)


def normalize_prediction_space_series(
    values: pd.Series | object,
    *,
    default: PredictionSpace = PredictionSpace.FUTURE_MID_YES,
) -> pd.Series:
    if isinstance(values, pd.Series):
        return values.map(lambda v: normalize_prediction_space(v, default=default).value).astype(str)
    return pd.Series([normalize_prediction_space(values, default=default).value], dtype=str)


def attach_prediction_contract(
    frame: pd.DataFrame,
    *,
    signal_col: str = "signal",
    threshold: float = 0.0,
    prediction_space: PredictionSpace | str | None = None,
    horizon: int | float | None = None,
    expected_return_col: str | None = None,
    horizon_steps: int | float | None = None,
    horizon_seconds: int | float | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    if "direction" not in out.columns:
        sig = pd.to_numeric(out.get(signal_col, 0.0), errors="coerce").fillna(0.0)
        out["direction"] = direction_label_series(sig, threshold=float(threshold)).astype(str)
    if "traded_side" not in out.columns:
        d = out["direction"].astype(str)
        out["traded_side"] = np.where(
            d == DIRECTION_LONG_YES,
            "YES",
            np.where(d == DIRECTION_LONG_NO, "NO", "NONE"),
        )
    if "prediction_space" not in out.columns:
        if prediction_space is None and "model_target_type" in out.columns:
            space = normalize_prediction_space_series(out["model_target_type"])
        else:
            space = normalize_prediction_space_series(prediction_space or PredictionSpace.FUTURE_MID_YES)
            if len(space) == 1:
                space = pd.Series(space.iloc[0], index=out.index, dtype=str)
        out["prediction_space"] = space.astype(str)
    else:
        out["prediction_space"] = normalize_prediction_space_series(out["prediction_space"])
    if "horizon" not in out.columns:
        if "model_horizon" in out.columns:
            out["horizon"] = pd.to_numeric(out["model_horizon"], errors="coerce").fillna(horizon if horizon is not None else 1.0)
        else:
            out["horizon"] = float(horizon if horizon is not None else 1.0)
    if "horizon_steps" not in out.columns:
        if horizon_steps is not None:
            out["horizon_steps"] = float(horizon_steps)
        elif "model_horizon" in out.columns:
            out["horizon_steps"] = pd.to_numeric(out["model_horizon"], errors="coerce").fillna(1.0)
        else:
            out["horizon_steps"] = pd.to_numeric(out["horizon"], errors="coerce").fillna(1.0)
    if "horizon_seconds" not in out.columns:
        if horizon_seconds is not None:
            out["horizon_seconds"] = float(horizon_seconds)
        else:
            out["horizon_seconds"] = pd.to_numeric(out["horizon_steps"], errors="coerce").fillna(1.0) * 3600.0
    if expected_return_col is not None and expected_return_col in out.columns:
        if "expected_return" not in out.columns:
            out["expected_return"] = pd.to_numeric(out[expected_return_col], errors="coerce").fillna(0.0)
        if "expected_net_return" not in out.columns:
            out["expected_net_return"] = pd.to_numeric(out[expected_return_col], errors="coerce").fillna(0.0)
    if "expected_return" in out.columns and "expected_net_return" not in out.columns:
        out["expected_net_return"] = pd.to_numeric(out["expected_return"], errors="coerce").fillna(0.0)
    if "expected_net_return" in out.columns and "expected_return" not in out.columns:
        out["expected_return"] = pd.to_numeric(out["expected_net_return"], errors="coerce").fillna(0.0)
    return out


def validate_prediction_contract(frame: pd.DataFrame, *, require_expected_return: bool = False) -> tuple[bool, list[str]]:
    required = ["prediction_space", "horizon", "direction", "horizon_steps", "horizon_seconds"]
    if require_expected_return:
        required.append("expected_net_return")
    missing = [c for c in required if c not in frame.columns]
    if missing:
        return False, [f"missing:{c}" for c in missing]
    issues: list[str] = []
    d = frame["direction"].astype(str)
    bad_dirs = ~d.isin([DIRECTION_LONG_YES, DIRECTION_LONG_NO, DIRECTION_FLAT])
    if bool(bad_dirs.any()):
        issues.append("invalid_direction_values")
    spaces = normalize_prediction_space_series(frame["prediction_space"])
    if spaces.isna().any():
        issues.append("invalid_prediction_space")
    if require_expected_return:
        er = pd.to_numeric(frame["expected_net_return"], errors="coerce")
        if er.isna().any():
            issues.append("expected_return_non_numeric")
    return len(issues) == 0, issues


def expected_return_comparable(
    frame: pd.DataFrame,
    *,
    expected_col: str = "expected_return",
    realized_col: str = "trade_return",
) -> bool:
    if frame.empty or expected_col not in frame.columns or realized_col not in frame.columns:
        return False
    exp = pd.to_numeric(frame[expected_col], errors="coerce").dropna()
    real = pd.to_numeric(frame[realized_col], errors="coerce").dropna()
    if exp.empty or real.empty:
        return False
    # Same units sanity: both should look like return fractions, not raw prices.
    exp_q = float(exp.abs().quantile(0.95))
    real_q = float(real.abs().quantile(0.95))
    if not np.isfinite(exp_q) or not np.isfinite(real_q):
        return False
    ratio = exp_q / max(real_q, 1e-9)
    return bool(1e-3 <= ratio <= 1e3)
