from __future__ import annotations

import numpy as np
import pandas as pd

from polymarket_edge.models.base import AlphaModel


class BayesianUpdatingAlphaModel(AlphaModel):
    def __init__(self, feature_cols: list[str], *, prior_col: str = "mid") -> None:
        super().__init__(feature_cols, prior_col=prior_col)
        self._mu0: dict[str, float] = {}
        self._mu1: dict[str, float] = {}
        self._sd0: dict[str, float] = {}
        self._sd1: dict[str, float] = {}
        self._base_prior: float = 0.5

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "BayesianUpdatingAlphaModel":
        data = x[self.feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        target = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)
        self._base_prior = float(target.mean()) if len(target) else 0.5
        self._base_prior = float(np.clip(self._base_prior, 1e-3, 1 - 1e-3))

        y0 = data[target == 0]
        y1 = data[target == 1]
        for col in self.feature_cols:
            mu0 = float(y0[col].mean()) if len(y0) else 0.0
            mu1 = float(y1[col].mean()) if len(y1) else 0.0
            sd0 = float(y0[col].std(ddof=0)) if len(y0) else 1.0
            sd1 = float(y1[col].std(ddof=0)) if len(y1) else 1.0
            self._mu0[col] = mu0
            self._mu1[col] = mu1
            self._sd0[col] = max(sd0, 1e-4)
            self._sd1[col] = max(sd1, 1e-4)
        return self

    @staticmethod
    def _gaussian_logpdf(x: np.ndarray, mu: float, sd: float) -> np.ndarray:
        z = (x - mu) / sd
        return -0.5 * np.log(2.0 * np.pi * sd * sd) - 0.5 * z * z

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        data = x[self.feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if self.prior_col in x.columns:
            prior = pd.to_numeric(x[self.prior_col], errors="coerce").fillna(self._base_prior).clip(1e-3, 1 - 1e-3).to_numpy()
        else:
            prior = np.full(len(data), self._base_prior, dtype=float)

        ll0 = np.zeros(len(data), dtype=float)
        ll1 = np.zeros(len(data), dtype=float)
        for col in self.feature_cols:
            v = data[col].to_numpy(dtype=float)
            ll0 += self._gaussian_logpdf(v, self._mu0[col], self._sd0[col])
            ll1 += self._gaussian_logpdf(v, self._mu1[col], self._sd1[col])

        logit_prior = np.log(prior / (1.0 - prior))
        logit_post = logit_prior + (ll1 - ll0)
        proba = 1.0 / (1.0 + np.exp(-np.clip(logit_post, -50, 50)))
        return np.asarray(np.clip(proba, 1e-6, 1 - 1e-6), dtype=float)

    def feature_importance(self) -> pd.DataFrame:
        rows = []
        for col in self.feature_cols:
            pooled = np.sqrt(0.5 * (self._sd0[col] ** 2 + self._sd1[col] ** 2))
            score = abs(self._mu1[col] - self._mu0[col]) / max(pooled, 1e-6)
            rows.append({"feature": col, "importance": float(score)})
        return pd.DataFrame(rows).sort_values("importance", ascending=False)
