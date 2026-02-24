from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def _feature_importance(frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    usable = [c for c in feature_cols if c in frame.columns]
    if not usable or frame.empty:
        return pd.DataFrame(columns=["feature", "importance"])
    x = frame[usable].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = pd.to_numeric(frame["trade_return"], errors="coerce").fillna(0.0)
    if len(x) < 30:
        corr_rows = []
        for col in usable:
            corr = float(np.corrcoef(x[col], y)[0, 1]) if float(x[col].std(ddof=0)) > 1e-12 else 0.0
            corr_rows.append({"feature": col, "importance": abs(corr), "signed_corr": corr})
        return pd.DataFrame(corr_rows).sort_values("importance", ascending=False).head(10)
    model = RandomForestRegressor(n_estimators=200, random_state=7, min_samples_leaf=10)
    model.fit(x, y)
    rows = [{"feature": col, "importance": float(imp)} for col, imp in zip(usable, model.feature_importances_)]
    return pd.DataFrame(rows).sort_values("importance", ascending=False).head(10)


def _condition_tables(frame: pd.DataFrame, top_features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty or not top_features:
        empty = pd.DataFrame(columns=["feature", "condition", "n", "avg_return", "win_rate"])
        return empty, empty

    strong_rows: list[dict[str, object]] = []
    weak_rows: list[dict[str, object]] = []
    for feat in top_features[:3]:
        if feat not in frame.columns:
            continue
        vals = pd.to_numeric(frame[feat], errors="coerce")
        if vals.notna().sum() < 20:
            continue
        q25 = float(vals.quantile(0.25))
        q75 = float(vals.quantile(0.75))
        high = frame[vals >= q75]
        low = frame[vals <= q25]
        for label, subset, collector in [
            (f"{feat}>=Q75", high, strong_rows),
            (f"{feat}<=Q25", low, weak_rows),
        ]:
            if subset.empty:
                continue
            collector.append(
                {
                    "feature": feat,
                    "condition": label,
                    "n": int(len(subset)),
                    "avg_return": float(subset["trade_return"].mean()),
                    "win_rate": float((subset["trade_return"] > 0).mean()),
                }
            )

    strong = pd.DataFrame(strong_rows).sort_values("avg_return", ascending=False) if strong_rows else pd.DataFrame()
    weak = pd.DataFrame(weak_rows).sort_values("avg_return", ascending=True) if weak_rows else pd.DataFrame()
    return strong, weak


def explain_strategy(trades: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if trades.empty:
        empty = pd.DataFrame()
        return empty, empty, empty

    imp = _feature_importance(trades, feature_cols)
    top = imp["feature"].head(5).tolist() if not imp.empty else []
    strong, weak = _condition_tables(trades, top)
    return imp, strong, weak
