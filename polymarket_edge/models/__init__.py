"""Model-based alpha interfaces and implementations."""

from polymarket_edge.models.base import AlphaModel
from polymarket_edge.models.bayesian import BayesianUpdatingAlphaModel
from polymarket_edge.models.evaluation import compare_models
from polymarket_edge.models.factory import create_alpha_model
from polymarket_edge.models.future_mid_regression import (
    GradientBoostingFutureMidRegressor,
    RidgeFutureMidRegressor,
    fit_predict_future_mid_ensemble,
)
from polymarket_edge.models.gradient_boosting import GradientBoostingAlphaModel
from polymarket_edge.models.logistic import LogisticAlphaModel

__all__ = [
    "AlphaModel",
    "LogisticAlphaModel",
    "GradientBoostingAlphaModel",
    "BayesianUpdatingAlphaModel",
    "RidgeFutureMidRegressor",
    "GradientBoostingFutureMidRegressor",
    "fit_predict_future_mid_ensemble",
    "create_alpha_model",
    "compare_models",
]
