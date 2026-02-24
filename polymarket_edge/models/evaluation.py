from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from polymarket_edge.models.base import AlphaModel


def _split_time_series(
    ts: pd.Series,
    *,
    n_splits: int,
    min_train_periods: int = 4,
) -> list[tuple[pd.Series, pd.Series]]:
    uniq = sorted(pd.to_datetime(ts).dropna().unique())
    if len(uniq) < min_train_periods + 2:
        return []
    fold_test = max(1, len(uniq) // (n_splits + 1))
    splits: list[tuple[pd.Series, pd.Series]] = []
    for i in range(n_splits):
        train_end = min_train_periods + i * fold_test
        test_end = train_end + fold_test
        if test_end > len(uniq):
            break
        train_ts = set(uniq[:train_end])
        test_ts = set(uniq[train_end:test_end])
        train_idx = ts.isin(train_ts)
        test_idx = ts.isin(test_ts)
        if train_idx.sum() == 0 or test_idx.sum() == 0:
            continue
        splits.append((train_idx, test_idx))
    return splits


def _metrics(y_true: np.ndarray, p: np.ndarray) -> dict[str, float]:
    if len(y_true) == 0:
        return {"auc": 0.5, "log_loss": 0.693, "brier": 0.25}
    y = y_true.astype(int)
    proba = np.clip(p.astype(float), 1e-6, 1 - 1e-6)
    if np.unique(y).size < 2:
        auc = 0.5
    else:
        auc = float(roc_auc_score(y, proba))
    return {
        "auc": auc,
        "log_loss": float(log_loss(y, proba, labels=[0, 1])),
        "brier": float(brier_score_loss(y, proba)),
    }


def compare_models(
    model_factories: dict[str, Callable[[], AlphaModel]],
    x: pd.DataFrame,
    y: pd.Series,
    ts: pd.Series,
    *,
    n_splits: int = 4,
) -> pd.DataFrame:
    frame = x.copy()
    target = pd.to_numeric(y, errors="coerce")
    mask = target.notna()
    if mask.sum() == 0:
        return pd.DataFrame()
    frame = frame.loc[mask].copy()
    target = target.loc[mask].astype(int)
    ts = pd.to_datetime(ts.loc[mask])
    splits = _split_time_series(ts, n_splits=n_splits)
    if not splits:
        rows = []
        for name in model_factories:
            rows.append({"model": name, "auc": 0.5, "log_loss": 0.693, "brier": 0.25, "folds": 0.0})
        return pd.DataFrame(rows).sort_values(["log_loss", "brier", "auc"], ascending=[True, True, False])

    rows: list[dict[str, float | str]] = []
    for name, make_model in model_factories.items():
        fold_metrics: list[dict[str, float]] = []
        for train_idx, test_idx in splits:
            x_train = frame.loc[train_idx]
            y_train = target.loc[train_idx]
            x_test = frame.loc[test_idx]
            y_test = target.loc[test_idx]
            if y_train.nunique() < 2 or len(x_test) == 0:
                continue
            model = make_model()
            model.fit(x_train, y_train)
            proba = model.predict_proba(x_test)
            fold_metrics.append(_metrics(y_test.to_numpy(), proba))
        if not fold_metrics:
            rows.append({"model": name, "auc": 0.5, "log_loss": 0.693, "brier": 0.25, "folds": 0.0})
            continue
        fm = pd.DataFrame(fold_metrics)
        rows.append(
            {
                "model": name,
                "auc": float(fm["auc"].mean()),
                "log_loss": float(fm["log_loss"].mean()),
                "brier": float(fm["brier"].mean()),
                "folds": float(len(fm)),
            }
        )
    return pd.DataFrame(rows).sort_values(["log_loss", "brier", "auc"], ascending=[True, True, False])
