from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from polymarket_edge.strategies.base import BaseStrategy, StrategyConfig


@dataclass
class CrossMarketStrategyConfig(StrategyConfig):
    signal_thresholds: tuple[float, ...] = (0.01, 0.02, 0.04, 0.08)
    min_event_size: int = 2


class CrossMarketStrategy(BaseStrategy):
    name = "cross_market_inefficiency_v2"

    def __init__(self, config: CrossMarketStrategyConfig | None = None) -> None:
        super().__init__(config or CrossMarketStrategyConfig())
        self.config: CrossMarketStrategyConfig

    def build_signal_frame(
        self,
        conn: duckdb.DuckDBPyConnection,
        panel: pd.DataFrame,
        context: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        if panel.empty:
            return pd.DataFrame()
        df = panel.copy()
        g = df.groupby(["event_id", "ts"], observed=True)
        df["event_n"] = g["market_id"].transform("nunique")
        df = df[df["event_n"] >= self.config.min_event_size].copy()
        if df.empty:
            return df
        df["event_mean_prob"] = g["mid"].transform("mean")
        df["event_std_prob"] = g["mid"].transform("std").replace(0.0, np.nan).fillna(0.01)
        df["dispersion_z"] = (df["mid"] - df["event_mean_prob"]) / df["event_std_prob"]
        df["raw_signal"] = -df["dispersion_z"] * np.log1p(df["event_n"])
        df["confidence"] = (
            (df["dispersion_z"].abs() / 3.0)
            * (1.0 / (1.0 + df["spread_bps"].clip(lower=0.0) / 250.0))
            * (1.0 + df["depth_log"] / 6.0)
        ).clip(lower=0.05, upper=1.0)
        df["signal"] = np.tanh(df["raw_signal"] / 2.0)
        return df.sort_values(["ts", "token_id"]).reset_index(drop=True)
