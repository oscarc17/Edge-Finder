from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from polymarket_edge.strategies.base import BaseStrategy, StrategyConfig


@dataclass
class LiquidityPremiumStrategyConfig(StrategyConfig):
    signal_thresholds: tuple[float, ...] = (0.005, 0.01, 0.02, 0.04, 0.08)
    low_liq_quantile: float = 0.35


class LiquidityPremiumStrategy(BaseStrategy):
    name = "liquidity_premium_v2"

    def __init__(self, config: LiquidityPremiumStrategyConfig | None = None) -> None:
        super().__init__(config or LiquidityPremiumStrategyConfig())
        self.config: LiquidityPremiumStrategyConfig
        self.feature_cols = self.feature_cols + ["low_liq_score"]

    def build_signal_frame(
        self,
        conn: duckdb.DuckDBPyConnection,
        panel: pd.DataFrame,
        context: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        if panel.empty:
            return pd.DataFrame()
        df = panel.copy()
        liq_cut = float(df["liquidity_log"].quantile(self.config.low_liq_quantile))
        low_liq = (liq_cut - df["liquidity_log"]).clip(lower=0.0)
        liq_scale = float(low_liq.std(ddof=0)) + 1e-9
        df["low_liq_score"] = low_liq / liq_scale

        vol_boost = 1.0 + (df["prob_vol_6"].abs() * 40.0).clip(0.0, 2.0)
        spread_penalty = 1.0 / (1.0 + df["spread_bps"].clip(lower=0.0) / 300.0)
        contrarian = -(df["mid"] - 0.5)
        df["raw_signal"] = contrarian * df["low_liq_score"] * vol_boost * spread_penalty
        df["signal"] = np.tanh(df["raw_signal"])
        df["confidence"] = (
            (df["low_liq_score"] / (1.0 + df["low_liq_score"]))
            * (0.5 + 0.5 * spread_penalty)
            * (0.5 + 0.5 * np.minimum(1.0, vol_boost / 2.0))
        ).clip(lower=0.05, upper=1.0)
        return df.sort_values(["ts", "token_id"]).reset_index(drop=True)
