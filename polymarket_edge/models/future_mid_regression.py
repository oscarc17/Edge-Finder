from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class RidgeFutureMidRegressor:
    def __init__(self, feature_cols: list[str], *, alpha: float = 1.0) -> None:
        self.feature_cols = list(feature_cols)
        self.model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("reg", Ridge(alpha=alpha)),
            ]
        )

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "RidgeFutureMidRegressor":
        data = x[self.feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        target = pd.to_numeric(y, errors="coerce").fillna(0.5).clip(lower=1e-6, upper=1 - 1e-6)
        self.model.fit(data, target)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        data = x[self.feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        pred = self.model.predict(data)
        return np.asarray(np.clip(pred, 1e-6, 1 - 1e-6), dtype=float)

    def feature_importance(self) -> pd.DataFrame:
        coef = np.abs(self.model.named_steps["reg"].coef_)
        rows = [{"feature": f, "importance": float(v)} for f, v in zip(self.feature_cols, coef)]
        return pd.DataFrame(rows).sort_values("importance", ascending=False)


class GradientBoostingFutureMidRegressor:
    def __init__(
        self,
        feature_cols: list[str],
        *,
        n_estimators: int = 150,
        learning_rate: float = 0.05,
        max_depth: int = 3,
    ) -> None:
        self.feature_cols = list(feature_cols)
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=7,
        )

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "GradientBoostingFutureMidRegressor":
        data = x[self.feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        target = pd.to_numeric(y, errors="coerce").fillna(0.5).clip(lower=1e-6, upper=1 - 1e-6)
        self.model.fit(data, target)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        data = x[self.feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        pred = self.model.predict(data)
        return np.asarray(np.clip(pred, 1e-6, 1 - 1e-6), dtype=float)

    def feature_importance(self) -> pd.DataFrame:
        imp = np.abs(self.model.feature_importances_)
        rows = [{"feature": f, "importance": float(v)} for f, v in zip(self.feature_cols, imp)]
        return pd.DataFrame(rows).sort_values("importance", ascending=False)


def fit_predict_future_mid_ensemble(
    *,
    feature_cols: list[str],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_pred_train: pd.DataFrame,
    x_pred_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    models = [
        ("future_mid_ridge", RidgeFutureMidRegressor(feature_cols, alpha=1.0)),
        ("future_mid_gb", GradientBoostingFutureMidRegressor(feature_cols, n_estimators=120, learning_rate=0.05, max_depth=2)),
    ]
    preds_train: list[np.ndarray] = []
    preds_test: list[np.ndarray] = []
    importances: list[pd.DataFrame] = []
    for name, model in models:
        model.fit(x_train, y_train)
        preds_train.append(model.predict(x_pred_train))
        preds_test.append(model.predict(x_pred_test))
        imp = model.feature_importance()
        if not imp.empty:
            imp = imp.copy()
            imp["model"] = name
            imp["task"] = "future_mid_regression"
            importances.append(imp)
    pred_train = np.mean(np.vstack(preds_train), axis=0) if preds_train else np.full(len(x_pred_train), 0.5, dtype=float)
    pred_test = np.mean(np.vstack(preds_test), axis=0) if preds_test else np.full(len(x_pred_test), 0.5, dtype=float)
    imp_df = pd.concat(importances, ignore_index=True) if importances else pd.DataFrame()
    return np.asarray(pred_train, dtype=float), np.asarray(pred_test, dtype=float), imp_df
