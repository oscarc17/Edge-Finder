from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from polymarket_edge.models.base import AlphaModel


class GradientBoostingAlphaModel(AlphaModel):
    def __init__(
        self,
        feature_cols: list[str],
        *,
        n_estimators: int = 250,
        learning_rate: float = 0.05,
        max_depth: int = 3,
    ) -> None:
        super().__init__(feature_cols)
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=7,
        )

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "GradientBoostingAlphaModel":
        data = x[self.feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        target = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)
        self.model.fit(data, target)
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        data = x[self.feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        p = self.model.predict_proba(data)[:, 1]
        return np.asarray(p, dtype=float)

    def feature_importance(self) -> pd.DataFrame:
        imp = np.abs(self.model.feature_importances_)
        rows = [{"feature": f, "importance": float(v)} for f, v in zip(self.feature_cols, imp)]
        return pd.DataFrame(rows).sort_values("importance", ascending=False)
