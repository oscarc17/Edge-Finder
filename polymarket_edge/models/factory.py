from __future__ import annotations

from polymarket_edge.models.base import AlphaModel
from polymarket_edge.models.bayesian import BayesianUpdatingAlphaModel
from polymarket_edge.models.gradient_boosting import GradientBoostingAlphaModel
from polymarket_edge.models.logistic import LogisticAlphaModel


def create_alpha_model(name: str, feature_cols: list[str], *, prior_col: str = "mid") -> AlphaModel:
    key = name.lower().strip()
    if key in {"logistic", "logreg", "logistic_regression"}:
        return LogisticAlphaModel(feature_cols)
    if key in {"gb", "gboost", "gradient_boosting", "gradientboosting"}:
        return GradientBoostingAlphaModel(feature_cols)
    if key in {"bayesian", "bayes", "bayesian_update"}:
        return BayesianUpdatingAlphaModel(feature_cols, prior_col=prior_col)
    raise ValueError(f"Unsupported model name: {name}")
