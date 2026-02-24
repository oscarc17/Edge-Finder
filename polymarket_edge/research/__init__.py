"""Shared research infrastructure for production-grade strategy modules."""

from polymarket_edge.research.backtest import (
    StrategyBacktestConfig,
    calibrate_execution_model,
    run_backtest_grid,
    run_execution_regime_sensitivity,
    run_impact_coefficient_sensitivity,
)
from polymarket_edge.research.advanced_validation import run_advanced_validation
from polymarket_edge.research.capacity import estimate_capacity
from polymarket_edge.research.data import build_feature_panel, build_yes_feature_panel
from polymarket_edge.research.deployment import evaluate_deployment_readiness
from polymarket_edge.research.diagnostics import run_signal_diagnostics
from polymarket_edge.research.explain import explain_strategy
from polymarket_edge.research.portfolio import PortfolioConfig, build_portfolio, build_portfolio_report
from polymarket_edge.research.types import StrategyExecutionResult
from polymarket_edge.research.validation import bootstrap_metrics, run_walkforward, validate_strategy
from polymarket_edge.research.direction import (
    DIRECTION_FLAT,
    DIRECTION_LONG_NO,
    DIRECTION_LONG_YES,
    direction_from_signal,
    direction_sign,
    direction_sign_from_signal,
)

__all__ = [
    "StrategyBacktestConfig",
    "StrategyExecutionResult",
    "build_feature_panel",
    "build_yes_feature_panel",
    "run_backtest_grid",
    "run_execution_regime_sensitivity",
    "run_impact_coefficient_sensitivity",
    "calibrate_execution_model",
    "run_walkforward",
    "bootstrap_metrics",
    "validate_strategy",
    "run_signal_diagnostics",
    "run_advanced_validation",
    "evaluate_deployment_readiness",
    "explain_strategy",
    "estimate_capacity",
    "PortfolioConfig",
    "build_portfolio",
    "build_portfolio_report",
    "DIRECTION_LONG_YES",
    "DIRECTION_LONG_NO",
    "DIRECTION_FLAT",
    "direction_from_signal",
    "direction_sign_from_signal",
    "direction_sign",
]
