from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import duckdb
import numpy as np
import pandas as pd

from polymarket_edge.strategies.base import BaseStrategy, StrategyConfig


AMBIGUITY_TERMS: Iterable[str] = (
    "official source",
    "if and only if",
    "unless",
    "dispute",
    "clarification",
    "at discretion",
    "invalidated",
    "or equivalent",
    "as determined by",
)


def ambiguity_score(text: str | None) -> float:
    if not text:
        return 0.0
    low = text.lower()
    token_hits = sum(low.count(token) for token in AMBIGUITY_TERMS)
    punctuation = low.count(";") + low.count(":") + low.count("(")
    return (len(low) / 400.0) + 0.8 * token_hits + 0.2 * punctuation


@dataclass
class ResolutionRuleStrategyConfig(StrategyConfig):
    signal_thresholds: tuple[float, ...] = (0.005, 0.01, 0.02, 0.04, 0.08)
    ambiguity_floor: float = 0.15


class ResolutionRuleStrategy(BaseStrategy):
    name = "resolution_rule_mispricing_v2"

    def __init__(self, config: ResolutionRuleStrategyConfig | None = None) -> None:
        super().__init__(config or ResolutionRuleStrategyConfig())
        self.config: ResolutionRuleStrategyConfig
        self.feature_cols = self.feature_cols + ["ambiguity_score"]

    def build_signal_frame(
        self,
        conn: duckdb.DuckDBPyConnection,
        panel: pd.DataFrame,
        context: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        if panel.empty:
            return pd.DataFrame()
        df = panel.copy()
        text = df["question"].fillna("") + " " + df["description"].fillna("")
        df["ambiguity_score"] = text.map(ambiguity_score)
        df["ambiguity_z"] = (df["ambiguity_score"] - df["ambiguity_score"].median()) / (
            df["ambiguity_score"].std(ddof=0) + 1e-9
        )
        df = df[df["ambiguity_score"] >= self.config.ambiguity_floor].copy()
        if df.empty:
            return df

        extreme = (df["mid"] - 0.5)
        uncertainty = np.maximum(0.0, df["ambiguity_z"])
        spread_penalty = 1.0 / (1.0 + df["spread_bps"].clip(lower=0.0) / 250.0)
        liquidity_penalty = 1.0 / (1.0 + np.exp(df["liquidity_log"] - df["liquidity_log"].median()))
        df["raw_signal"] = -extreme * (1.0 + uncertainty) * spread_penalty
        df["signal"] = np.tanh(df["raw_signal"] * (1.0 + liquidity_penalty))
        df["confidence"] = (uncertainty * spread_penalty * (0.5 + liquidity_penalty)).clip(lower=0.05, upper=1.0)
        return df.sort_values(["ts", "token_id"]).reset_index(drop=True)
