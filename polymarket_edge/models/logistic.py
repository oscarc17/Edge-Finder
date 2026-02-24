from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from polymarket_edge.models.base import AlphaModel


class LogisticAlphaModel(AlphaModel):
    def __init__(self, feature_cols: list[str], *, c: float = 1.0) -> None:
        super().__init__(feature_cols)
        self.model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=c, penalty="l2", solver="lbfgs", max_iter=2000)),
            ]
        )

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "LogisticAlphaModel":
        data = x[self.feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        target = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)
        self.model.fit(data, target)
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        data = x[self.feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        p = self.model.predict_proba(data)[:, 1]
        return np.asarray(p, dtype=float)

    def feature_importance(self) -> pd.DataFrame:
        clf = self.model.named_steps["clf"]
        coef = np.abs(clf.coef_[0])
        rows = [{"feature": f, "importance": float(v)} for f, v in zip(self.feature_cols, coef)]
        return pd.DataFrame(rows).sort_values("importance", ascending=False)
