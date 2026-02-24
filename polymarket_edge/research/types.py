from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class StrategyExecutionResult:
    strategy_name: str
    tuned_params: dict[str, object]
    signal_frame: pd.DataFrame
    train_summary: pd.DataFrame
    test_summary: pd.DataFrame
    train_trades: pd.DataFrame
    test_trades: pd.DataFrame
    validation_summary: pd.DataFrame
    explain_summary: pd.DataFrame
    strong_conditions: pd.DataFrame
    weak_conditions: pd.DataFrame
    capacity_summary: pd.DataFrame
    model_comparison: pd.DataFrame
    model_feature_importance: pd.DataFrame
    label_diagnostics: pd.DataFrame
    walkforward_folds: pd.DataFrame
    walkforward_summary: pd.DataFrame
    stability_time: pd.DataFrame
    stability_category: pd.DataFrame
    parameter_sensitivity: pd.DataFrame
    filter_impact: pd.DataFrame
    edge_attribution: pd.DataFrame
    calibration_curve: pd.DataFrame
    signal_deciles: pd.DataFrame
    feature_drift: pd.DataFrame
    model_vs_market: pd.DataFrame
    ev_diagnostics: pd.DataFrame
    advanced_validation: pd.DataFrame
    deployment_readiness: pd.DataFrame
    execution_sensitivity: pd.DataFrame
    execution_calibration: pd.DataFrame
    cost_decomposition: pd.DataFrame
    ev_threshold_grid: pd.DataFrame
