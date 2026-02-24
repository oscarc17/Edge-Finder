from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from polymarket_edge.strategies.base import BaseStrategy, StrategyConfig


@dataclass
class MomentumStrategyConfig(StrategyConfig):
    signal_thresholds: tuple[float, ...] = (0.005, 0.01, 0.02, 0.04, 0.08)
    momentum_hours: float = 72.0


class MomentumReversionStrategy(BaseStrategy):
    name = "momentum_vs_mean_reversion_v2"

    def __init__(self, config: MomentumStrategyConfig | None = None) -> None:
        super().__init__(config or MomentumStrategyConfig())
        self.config: MomentumStrategyConfig
        self.feature_cols = self.feature_cols + ["regime_signal"]

    def build_signal_frame(
        self,
        conn: duckdb.DuckDBPyConnection,
        panel: pd.DataFrame,
        context: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        if panel.empty:
            return pd.DataFrame()
        df = panel.copy()
        vel = df["velocity_1"].astype(float)
        acc = df["acceleration"].astype(float)

        vel_z = (vel - vel.mean()) / (vel.std(ddof=0) + 1e-9)
        acc_z = (acc - acc.mean()) / (acc.std(ddof=0) + 1e-9)
        ttr = df["time_to_resolution_h"].fillna(9_999.0)
        regime = np.where(ttr <= self.config.momentum_hours, 1.0, -1.0)
        df["regime_signal"] = regime

        base = 0.7 * vel_z + 0.3 * acc_z
        if float(base.std(ddof=0)) < 1e-12:
            # Sparse panels can have flat short-horizon returns; use level deviation fallback.
            base = df["mid"].astype(float) - 0.5
        df["raw_signal"] = regime * base
        df["signal"] = np.tanh(df["raw_signal"] / 2.0)

        vol_control = 1.0 / (1.0 + (df["prob_vol_24"].abs() * 30.0))
        trend_strength = np.minimum(1.0, np.abs(vel_z) / 3.0)
        df["confidence"] = (0.5 * trend_strength + 0.5 * vol_control).clip(lower=0.05, upper=1.0)
        return df.sort_values(["ts", "token_id"]).reset_index(drop=True)
