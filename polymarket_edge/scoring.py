from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def score_edges(
    edge_summary: pd.DataFrame,
    *,
    capacity_ref_usd: float = 50_000.0,
    w_profitability: float = 0.45,
    w_significance: float = 0.15,
    w_capacity: float = 0.10,
    w_stability: float = 0.10,
    w_reliability: float = 0.20,
) -> pd.DataFrame:
    """
    Expected columns:
    edge_name, capacity_usd, stability
    mean_return (canonical should be return_per_day)
    Optional: p_value, t_stat, n_obs, ci_low, test_trade_count, walkforward_valid_folds, label_degenerate
    """
    if edge_summary.empty:
        return edge_summary

    frame = edge_summary.copy()
    frame["mean_return"] = pd.to_numeric(frame.get("mean_return", 0.0), errors="coerce").fillna(0.0)
    frame["p_value"] = pd.to_numeric(frame.get("p_value", 1.0), errors="coerce").fillna(1.0)
    frame["t_stat"] = pd.to_numeric(frame.get("t_stat", np.nan), errors="coerce")
    frame["n_obs"] = pd.to_numeric(frame.get("n_obs", np.nan), errors="coerce")
    frame["ci_low"] = pd.to_numeric(frame.get("ci_low", frame.get("mean_return_ci_low", np.nan)), errors="coerce")
    frame["capacity_usd"] = frame["capacity_usd"].fillna(0.0)
    frame["stability"] = frame["stability"].fillna(0.0)
    trade_raw = frame["test_trade_count"] if "test_trade_count" in frame.columns else pd.Series(0.0, index=frame.index)
    folds_raw = frame["walkforward_valid_folds"] if "walkforward_valid_folds" in frame.columns else pd.Series(0.0, index=frame.index)
    label_raw = frame["label_degenerate"] if "label_degenerate" in frame.columns else pd.Series(0.0, index=frame.index)
    frame["test_trade_count"] = pd.to_numeric(trade_raw, errors="coerce").fillna(0.0)
    frame["walkforward_valid_folds"] = pd.to_numeric(folds_raw, errors="coerce").fillna(0.0)
    frame["label_degenerate"] = pd.to_numeric(label_raw, errors="coerce").fillna(0.0)
    hedged_raw = frame["hedged_fill_rate"] if "hedged_fill_rate" in frame.columns else pd.Series(1.0, index=frame.index)
    cost_dom_raw = frame["cost_dominance_ratio"] if "cost_dominance_ratio" in frame.columns else pd.Series(0.0, index=frame.index)
    frame["hedged_fill_rate"] = pd.to_numeric(hedged_raw, errors="coerce").fillna(1.0).clip(lower=0.0, upper=1.0)
    frame["cost_dominance_ratio"] = pd.to_numeric(cost_dom_raw, errors="coerce").fillna(0.0).clip(lower=0.0)
    frame["p_value"] = frame["p_value"].clip(lower=1e-12)

    def _p_value_pos(row: pd.Series) -> float:
        mu = float(row.get("mean_return", 0.0))
        p_two = float(np.clip(row.get("p_value", 1.0), 1e-12, 1.0))
        t_stat = row.get("t_stat", np.nan)
        n_obs = row.get("n_obs", np.nan)
        if np.isfinite(t_stat):
            t_val = float(t_stat)
            if np.isfinite(n_obs) and float(n_obs) > 1:
                df = float(n_obs) - 1.0
                p = float(stats.t.sf(t_val, df=df))
            else:
                p = float(stats.norm.sf(t_val))
        else:
            if mu > 0.0:
                p = 0.5 * p_two
            elif mu < 0.0:
                p = 1.0 - 0.5 * p_two
            else:
                p = 1.0
        return float(np.clip(p, 1e-12, 1.0))

    frame["p_value_pos"] = frame.apply(_p_value_pos, axis=1)

    # Profitability score emphasizes conservative positive return and CI support.
    conservative = np.where(frame["ci_low"].notna(), np.minimum(frame["mean_return"], frame["ci_low"]), frame["mean_return"])
    conservative = np.maximum(conservative, 0.0)
    scale = float(np.nanpercentile(conservative, 75)) if np.any(conservative > 0.0) else 0.0
    scale = max(scale, 1e-6)
    frame["profitability_score"] = np.clip(conservative / scale, 0.0, 1.0)
    frame.loc[frame["ci_low"].fillna(-np.inf) <= 0.0, "profitability_score"] = 0.0

    # Significance only rewarded for positive-mean candidates.
    raw_sig = np.maximum(np.clip(-np.log10(frame["p_value_pos"]) / 6.0, 0.0, 1.0), 0.0)
    frame["significance_score"] = np.where((frame["mean_return"] > 0.0) & (frame["ci_low"].fillna(-np.inf) > 0.0), raw_sig, 0.0)

    frame["capacity_score"] = np.clip(np.log1p(frame["capacity_usd"]) / np.log1p(capacity_ref_usd), 0.0, 1.0)
    frame["stability_score"] = frame["stability"].clip(lower=0.0, upper=1.0)
    trades_score = np.clip(frame["test_trade_count"] / 200.0, 0.0, 1.0)
    folds_score = np.clip(frame["walkforward_valid_folds"] / 4.0, 0.0, 1.0)
    label_score = np.where(frame["label_degenerate"] > 0.5, 0.0, 1.0)
    frame["reliability_score"] = 0.5 * trades_score + 0.3 * folds_score + 0.2 * label_score
    # Structural-arb quality hooks (harmless defaults for non-arb strategies).
    frame["reliability_score"] = (
        frame["reliability_score"]
        * (0.75 + 0.25 * frame["hedged_fill_rate"])
        * np.where(frame["cost_dominance_ratio"] <= 1.0, 1.0, 1.0 / (1.0 + (frame["cost_dominance_ratio"] - 1.0)))
    ).clip(lower=0.0, upper=1.0)

    frame["edge_score_pro"] = (
        w_profitability * frame["profitability_score"]
        + w_significance * frame["significance_score"]
        + w_capacity * frame["capacity_score"]
        + w_stability * frame["stability_score"]
        + w_reliability * frame["reliability_score"]
    )
    frame.loc[(frame["mean_return"] <= 0.0) | (frame["ci_low"].fillna(-np.inf) <= 0.0), "edge_score_pro"] = 0.0
    frame.loc[frame["test_trade_count"] < 50.0, "edge_score_pro"] = frame["edge_score_pro"] * 0.25
    frame.loc[frame["walkforward_valid_folds"] < 2.0, "edge_score_pro"] = frame["edge_score_pro"] * 0.25
    frame.loc[frame["label_degenerate"] > 0.5, "edge_score_pro"] = 0.0
    if "hedged_fill_rate" in frame.columns:
        frame.loc[frame["hedged_fill_rate"] < 0.25, "edge_score_pro"] = frame["edge_score_pro"] * 0.5
    if "cost_dominance_ratio" in frame.columns:
        frame.loc[frame["cost_dominance_ratio"] > 1.0, "edge_score_pro"] = frame["edge_score_pro"] * 0.5
    # Legacy compatibility column.
    frame["edge_score"] = frame["edge_score_pro"]
    return frame.sort_values("edge_score_pro", ascending=False)


def example_edge_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "edge_name": "cross_market_inefficiency",
                "p_value": 0.012,
                "capacity_usd": 18000,
                "stability": 0.69,
                "mean_return": 0.021,
            },
            {
                "edge_name": "liquidity_premium",
                "p_value": 0.004,
                "capacity_usd": 9500,
                "stability": 0.74,
                "mean_return": 0.028,
            },
            {
                "edge_name": "whale_behavior",
                "p_value": 0.039,
                "capacity_usd": 52000,
                "stability": 0.58,
                "mean_return": 0.013,
            },
        ]
    )
