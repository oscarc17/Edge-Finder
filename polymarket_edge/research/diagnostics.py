from __future__ import annotations

import numpy as np
import pandas as pd

from polymarket_edge.backtest.metrics import sharpe_ratio
from polymarket_edge.research.direction import (
    DIRECTION_LONG_NO,
    DIRECTION_LONG_YES,
    direction_label_series,
    traded_side_series,
)


def _safe_log_loss(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    prob = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    tgt = np.asarray(y, dtype=float)
    loss = -(tgt * np.log(prob) + (1.0 - tgt) * np.log(1.0 - prob))
    return float(np.mean(loss))


def _attach_probability_columns(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    out = trades.copy()
    signal = pd.to_numeric(out.get("signal", 0.0), errors="coerce").fillna(0.0)
    if "direction" in out.columns:
        raw_dir = out["direction"]
        if pd.api.types.is_numeric_dtype(raw_dir):
            dir_num = pd.to_numeric(raw_dir, errors="coerce").fillna(0.0)
            direction_label = pd.Series("FLAT", index=out.index, dtype=object)
            direction_label[dir_num > 0.0] = DIRECTION_LONG_YES
            direction_label[dir_num < 0.0] = DIRECTION_LONG_NO
        else:
            up = raw_dir.astype(str).str.upper().str.strip()
            direction_label = pd.Series("FLAT", index=out.index, dtype=object)
            direction_label[up.isin([DIRECTION_LONG_YES, "LONG_YES", "LONGYES"])] = DIRECTION_LONG_YES
            direction_label[up.isin([DIRECTION_LONG_NO, "LONG_NO", "LONGNO"])] = DIRECTION_LONG_NO
            # legacy numeric string support
            num = pd.to_numeric(raw_dir, errors="coerce")
            direction_label[num > 0.0] = DIRECTION_LONG_YES
            direction_label[num < 0.0] = DIRECTION_LONG_NO
    else:
        direction_label = direction_label_series(signal, threshold=0.0)
    direction = pd.Series(
        np.where(direction_label == DIRECTION_LONG_YES, 1.0, np.where(direction_label == DIRECTION_LONG_NO, -1.0, 0.0)),
        index=out.index,
    )
    mid = pd.to_numeric(out.get("mid", 0.5), errors="coerce").fillna(0.5).clip(lower=1e-6, upper=1 - 1e-6)
    target_type = out["model_target_type"].astype(str).str.upper().str.strip() if "model_target_type" in out.columns else pd.Series("", index=out.index)
    proba_target_type = out["model_proba_target_type"].astype(str).str.upper().str.strip() if "model_proba_target_type" in out.columns else pd.Series("", index=out.index)

    if "model_pred_outcome_prob_yes" in out.columns and pd.to_numeric(out["model_pred_outcome_prob_yes"], errors="coerce").notna().sum() > 0:
        model_up = pd.to_numeric(out["model_pred_outcome_prob_yes"], errors="coerce").fillna(mid)
    elif "model_prob_up" in out.columns:
        model_up = pd.to_numeric(out["model_prob_up"], errors="coerce").fillna(mid)
    elif "model_proba" in out.columns and (((target_type == "OUTCOME_PROB") | (proba_target_type == "OUTCOME_PROB")).any()):
        model_up = pd.to_numeric(out["model_proba"], errors="coerce").fillna(mid)
    elif "model_pred_future_mid" in out.columns and pd.to_numeric(out["model_pred_future_mid"], errors="coerce").notna().sum() > 0:
        model_up = pd.to_numeric(out["model_pred_future_mid"], errors="coerce").fillna(mid)
    else:
        conf_raw = out["confidence"] if "confidence" in out.columns else pd.Series(0.5, index=out.index)
        confidence = pd.to_numeric(conf_raw, errors="coerce").fillna(0.5).clip(lower=0.05, upper=1.0)
        headroom = np.minimum(mid, 1.0 - mid)
        max_shift = 0.10 * headroom * confidence
        model_up = (mid + signal * max_shift).clip(lower=1e-6, upper=1 - 1e-6)
    model_up = model_up.clip(lower=1e-6, upper=1 - 1e-6)

    out["direction"] = direction_label.astype(str)
    out["direction_sign"] = direction.astype(float)
    out["traded_side"] = traded_side_series(direction_label)
    out["market_prob_dir"] = np.where(direction_label == DIRECTION_LONG_YES, mid, 1.0 - mid)
    out["model_prob_dir"] = np.where(direction_label == DIRECTION_LONG_YES, model_up, 1.0 - model_up)
    out["realized_win"] = (pd.to_numeric(out.get("trade_return", 0.0), errors="coerce").fillna(0.0) > 0).astype(float)
    out["signal_strength"] = np.abs(signal)
    out["model_edge_prob"] = out["model_prob_dir"] - out["market_prob_dir"]
    return out


def _metric_row(name: str, frame: pd.DataFrame) -> dict[str, float | str]:
    if frame.empty:
        return {
            "segment": name,
            "n": 0.0,
            "avg_return": 0.0,
            "win_rate": 0.0,
            "model_brier": 0.0,
            "market_brier": 0.0,
            "brier_improvement": 0.0,
            "model_logloss": 0.0,
            "market_logloss": 0.0,
            "avg_model_edge_prob": 0.0,
        }
    y = frame["realized_win"].to_numpy(dtype=float)
    model_prob = frame["model_prob_dir"].to_numpy(dtype=float)
    market_prob = frame["market_prob_dir"].to_numpy(dtype=float)
    model_brier = float(np.mean(np.square(model_prob - y)))
    market_brier = float(np.mean(np.square(market_prob - y)))
    return {
        "segment": name,
        "n": float(len(frame)),
        "avg_return": float(pd.to_numeric(frame["trade_return"], errors="coerce").fillna(0.0).mean()),
        "win_rate": float(np.mean(y)),
        "model_brier": model_brier,
        "market_brier": market_brier,
        "brier_improvement": float(market_brier - model_brier),
        "model_logloss": _safe_log_loss(y, model_prob),
        "market_logloss": _safe_log_loss(y, market_prob),
        "avg_model_edge_prob": float(pd.to_numeric(frame["model_edge_prob"], errors="coerce").fillna(0.0).mean()),
    }


def _edge_attribution(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    t = _attach_probability_columns(trades)
    rows: list[dict[str, float | str]] = [_metric_row("overall", t)]

    segment_cols = [c for c in ["category", "vol_regime", "is_low_liq_regime", "is_high_vol_regime"] if c in t.columns]
    for col in segment_cols:
        grouped = t.groupby(col, observed=True)
        for key, g in grouped:
            rows.append(_metric_row(f"{col}={key}", g))
    out = pd.DataFrame(rows)
    return out.sort_values(["segment"])


def _calibration_curve(trades: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    t = _attach_probability_columns(trades)
    rows: list[dict[str, float | str]] = []
    for source, col in [("model", "model_prob_dir"), ("market", "market_prob_dir")]:
        vals = pd.to_numeric(t[col], errors="coerce").fillna(0.5).clip(lower=1e-6, upper=1 - 1e-6)
        if vals.nunique() <= 1:
            bins = pd.Series(0, index=vals.index)
        else:
            bins = pd.qcut(vals.rank(method="first"), q=min(n_bins, vals.nunique()), labels=False, duplicates="drop")
        tmp = t.copy()
        tmp["bin"] = bins
        grouped = tmp.groupby("bin", observed=True)
        for b, g in grouped:
            avg_pred = float(pd.to_numeric(g[col], errors="coerce").mean())
            win = float(pd.to_numeric(g["realized_win"], errors="coerce").mean())
            rows.append(
                {
                    "source": source,
                    "bin": int(b),
                    "n": float(len(g)),
                    "avg_pred": avg_pred,
                    "observed_win_rate": win,
                    "calibration_gap": float(avg_pred - win),
                    "avg_return": float(pd.to_numeric(g["trade_return"], errors="coerce").mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["source", "bin"])


def _performance_by_signal_decile(trades: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    t = _attach_probability_columns(trades)
    strength = pd.to_numeric(t["signal_strength"], errors="coerce").fillna(0.0)
    if strength.nunique() <= 1:
        dec = pd.Series(0, index=t.index)
    else:
        dec = pd.qcut(strength.rank(method="first"), q=min(n_bins, strength.nunique()), labels=False, duplicates="drop")
    t["signal_decile"] = dec

    rows: list[dict[str, float]] = []
    for d, g in t.groupby("signal_decile", observed=True):
        r = pd.to_numeric(g["trade_return"], errors="coerce").fillna(0.0)
        if "expected_net_return" in g.columns:
            ev = pd.to_numeric(g["expected_net_return"], errors="coerce").fillna(0.0)
        elif "expected_return" in g.columns:
            ev = pd.to_numeric(g["expected_return"], errors="coerce").fillna(0.0)
        elif "expected_ev" in g.columns:
            ev = pd.to_numeric(g["expected_ev"], errors="coerce").fillna(0.0)
        else:
            ev = pd.Series(0.0, index=g.index)
        rows.append(
            {
                "signal_decile": float(d),
                "n": float(len(g)),
                "avg_return": float(r.mean()),
                "win_rate": float(np.mean(r > 0)),
                "sharpe": sharpe_ratio(r, periods_per_year=24 * 365),
                "ret_p10": float(r.quantile(0.10)),
                "ret_p50": float(r.quantile(0.50)),
                "ret_p90": float(r.quantile(0.90)),
                "avg_expected_ev": float(ev.mean()),
                "avg_model_edge_prob": float(pd.to_numeric(g["model_edge_prob"], errors="coerce").fillna(0.0).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("signal_decile")


def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    e = np.asarray(expected, dtype=float)
    a = np.asarray(actual, dtype=float)
    if len(e) < 20 or len(a) < 20:
        return 0.0
    qs = np.quantile(e, np.linspace(0.0, 1.0, bins + 1))
    qs[0] = -np.inf
    qs[-1] = np.inf
    e_hist, _ = np.histogram(e, bins=qs)
    a_hist, _ = np.histogram(a, bins=qs)
    e_pct = np.clip(e_hist / max(1.0, np.sum(e_hist)), 1e-6, None)
    a_pct = np.clip(a_hist / max(1.0, np.sum(a_hist)), 1e-6, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def _feature_drift(signal_frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    if signal_frame.empty:
        return pd.DataFrame()
    df = signal_frame.copy()
    if "ts" not in df.columns:
        return pd.DataFrame()
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts")
    n = len(df)
    if n < 40:
        return pd.DataFrame()

    split = max(10, n // 3)
    early = df.iloc[:split]
    late = df.iloc[-split:]
    rows: list[dict[str, float | str]] = []
    for col in feature_cols:
        if col not in df.columns:
            continue
        e = pd.to_numeric(early[col], errors="coerce").dropna().to_numpy(dtype=float)
        l = pd.to_numeric(late[col], errors="coerce").dropna().to_numpy(dtype=float)
        if len(e) < 20 or len(l) < 20:
            continue
        mean_shift = float(np.mean(l) - np.mean(e))
        std0 = float(np.std(e, ddof=0))
        drift_z = mean_shift / max(std0, 1e-6)
        psi = _psi(e, l, bins=10)
        rows.append(
            {
                "feature": col,
                "mean_early": float(np.mean(e)),
                "mean_late": float(np.mean(l)),
                "mean_shift": mean_shift,
                "drift_z": float(drift_z),
                "psi": psi,
                "drift_flag": float((abs(drift_z) >= 0.5) or (psi >= 0.2)),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["drift_flag", "psi", "drift_z"], ascending=[False, False, False]) if not out.empty else out


def _model_vs_market(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    t = _attach_probability_columns(trades)
    rows: list[dict[str, float | str]] = []

    # Where model adds value by confidence regime.
    strength = pd.to_numeric(t["signal_strength"], errors="coerce").fillna(0.0)
    if strength.nunique() <= 1:
        t["strength_bin"] = 0
    else:
        t["strength_bin"] = pd.qcut(strength.rank(method="first"), q=min(5, strength.nunique()), labels=False, duplicates="drop")
    for b, g in t.groupby("strength_bin", observed=True):
        row = _metric_row(f"strength_bin={b}", g)
        rows.append(row)

    if "category" in t.columns:
        for cat, g in t.groupby("category", observed=True):
            row = _metric_row(f"category={cat}", g)
            rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values("brier_improvement", ascending=False) if not out.empty else out


def _ev_diagnostics(trades: pd.DataFrame) -> pd.DataFrame:
    exp_col = None
    for c in ["expected_net_return", "expected_return", "expected_ev"]:
        if c in trades.columns:
            exp_col = c
            break
    if trades.empty or exp_col is None:
        return pd.DataFrame()
    t = trades.copy()
    ev = pd.to_numeric(t[exp_col], errors="coerce").fillna(0.0)
    real = pd.to_numeric(t["trade_return"], errors="coerce").fillna(0.0)
    if ev.nunique() <= 1:
        bins = pd.Series(0, index=t.index)
    else:
        bins = pd.qcut(ev.rank(method="first"), q=min(10, ev.nunique()), labels=False, duplicates="drop")
    t["ev_bin"] = bins

    rows: list[dict[str, float]] = []
    for b, g in t.groupby("ev_bin", observed=True):
        e = pd.to_numeric(g[exp_col], errors="coerce").fillna(0.0)
        r = pd.to_numeric(g["trade_return"], errors="coerce").fillna(0.0)
        rows.append(
            {
                "ev_bin": float(b),
                "n": float(len(g)),
                "avg_expected_ev": float(e.mean()),
                "avg_realized_return": float(r.mean()),
                "ev_gap": float((r - e).mean()),
                "ev_hit_rate": float(np.mean(r > 0.0)),
            }
        )

    bins_only = pd.DataFrame(rows).sort_values("ev_bin")
    n_bins_populated = int(len(bins_only))
    min_bin_count = int(pd.to_numeric(bins_only["n"], errors="coerce").min()) if not bins_only.empty else 0
    monotonic_pass = 0.0
    if not bins_only.empty and len(bins_only) >= 2:
        ordered = bins_only.sort_values("avg_expected_ev")
        y = pd.to_numeric(ordered["avg_realized_return"], errors="coerce").fillna(0.0)
        dy = np.diff(y.to_numpy(dtype=float))
        monotonic_pass = float(np.all(dy >= -1e-12))

    if len(t) >= 5 and float(ev.std(ddof=0)) > 1e-12 and float(real.std(ddof=0)) > 1e-12:
        ev_spearman = float(pd.Series(ev).corr(pd.Series(real), method="spearman"))
        beta = float(np.cov(ev, real, ddof=0)[0, 1] / np.var(ev))
        corr = float(np.corrcoef(ev, real)[0, 1])
    else:
        ev_spearman = 0.0
        beta = 0.0
        corr = 0.0
    pos_ev_mask = ev > 0.0
    hit_rate_pos_ev = float(np.mean(real[pos_ev_mask] > 0.0)) if float(pos_ev_mask.sum()) > 0.0 else 0.0
    min_trades_for_sign = 100
    well_populated_bins = bool(n_bins_populated >= min(5, max(1, ev.nunique()))) and (min_bin_count > 0)
    if len(t) < min_trades_for_sign:
        ev_model_valid = 1.0 if well_populated_bins else 0.0
    else:
        ev_model_valid = 1.0 if (ev_spearman >= 0.0 and well_populated_bins) else 0.0

    out = pd.DataFrame(rows).sort_values("ev_bin")
    out["ev_calibration_beta"] = beta
    out["ev_calibration_slope"] = beta
    out["ev_realized_corr"] = corr
    out["ev_spearman"] = ev_spearman
    out["ev_monotonic_pass"] = monotonic_pass
    out["hit_rate_pos_ev"] = hit_rate_pos_ev
    out["ev_model_valid"] = ev_model_valid
    out["n_trades"] = float(len(t))
    out["ev_expected_col"] = str(exp_col)
    out["ev_bins_populated"] = float(n_bins_populated)
    out["ev_min_bin_count"] = float(min_bin_count)
    out["ev_well_populated_bins"] = float(well_populated_bins)
    return out


def run_signal_diagnostics(
    *,
    signal_frame: pd.DataFrame,
    trades: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, pd.DataFrame]:
    if trades.empty:
        empty = pd.DataFrame()
        return {
            "edge_attribution": empty,
            "calibration_curve": empty,
            "signal_deciles": empty,
            "feature_drift": _feature_drift(signal_frame, feature_cols),
            "model_vs_market": empty,
            "ev_diagnostics": empty,
        }

    return {
        "edge_attribution": _edge_attribution(trades),
        "calibration_curve": _calibration_curve(trades, n_bins=10),
        "signal_deciles": _performance_by_signal_decile(trades, n_bins=10),
        "feature_drift": _feature_drift(signal_frame, feature_cols),
        "model_vs_market": _model_vs_market(trades),
        "ev_diagnostics": _ev_diagnostics(trades),
    }
