from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class AlphaModel(ABC):
    def __init__(self, feature_cols: list[str], *, prior_col: str = "mid") -> None:
        self.feature_cols = list(feature_cols)
        self.prior_col = prior_col

    @abstractmethod
    def fit(self, x: pd.DataFrame, y: pd.Series) -> "AlphaModel":
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def generate_signals(self, x: pd.DataFrame) -> pd.DataFrame:
        p = np.clip(self.predict_proba(x), 1e-6, 1 - 1e-6)
        signal = 2.0 * (p - 0.5)
        confidence = np.clip(np.abs(signal), 0.05, 1.0)
        return pd.DataFrame({"proba": p, "signal": signal, "confidence": confidence}, index=x.index)

    def feature_importance(self) -> pd.DataFrame:
        return pd.DataFrame(columns=["feature", "importance"])
