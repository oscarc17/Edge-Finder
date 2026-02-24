from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from polymarket_edge.strategies.base import BaseStrategy, StrategyConfig


@dataclass
class WhaleStrategyConfig(StrategyConfig):
    signal_thresholds: tuple[float, ...] = (0.005, 0.01, 0.02, 0.04, 0.08)
    whale_quantile: float = 0.90
    alpha_lookback_hours: int = 72


def _load_trade_frame(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return conn.execute(
        """
        SELECT
            t.trade_id,
            t.market_id,
            t.token_id,
            t.trader,
            t.side,
            t.price,
            t.size,
            t.notional,
            t.trade_ts,
            r.winner_token_id
        FROM trades t
        LEFT JOIN resolutions r
            ON t.market_id = r.market_id
        WHERE t.trade_ts IS NOT NULL
          AND t.token_id IS NOT NULL
          AND t.trader IS NOT NULL
          AND t.price IS NOT NULL
          AND t.size IS NOT NULL
        """
    ).df()


def _trade_alpha(row: pd.Series) -> float:
    if pd.isna(row["winner_token_id"]):
        return np.nan
    y = 1.0 if str(row["token_id"]) == str(row["winner_token_id"]) else 0.0
    side = str(row["side"]).upper()
    if side == "BUY":
        return y - float(row["price"])
    if side == "SELL":
        return float(row["price"]) - y
    return np.nan


class WhaleBehaviorStrategy(BaseStrategy):
    name = "whale_behavior_v2"

    def __init__(self, config: WhaleStrategyConfig | None = None) -> None:
        super().__init__(config or WhaleStrategyConfig())
        self.config: WhaleStrategyConfig
        self.feature_cols = self.feature_cols + ["whale_flow_z", "whale_alpha_z"]

    def _build_whale_flow(self, conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
        trades = _load_trade_frame(conn)
        if trades.empty:
            return pd.DataFrame(columns=["hr", "token_id", "whale_flow", "whale_alpha"])

        trades["trade_ts"] = pd.to_datetime(trades["trade_ts"])
        trades["alpha"] = trades.apply(_trade_alpha, axis=1)
        trades["signed_notional"] = np.where(trades["side"].str.upper() == "BUY", 1.0, -1.0) * trades["notional"].astype(float)

        wallet_notional = trades.groupby("trader", observed=True)["notional"].sum()
        cutoff = float(wallet_notional.quantile(self.config.whale_quantile))
        whales = set(wallet_notional[wallet_notional >= cutoff].index)
        trades = trades[trades["trader"].isin(whales)].copy()
        if trades.empty:
            return pd.DataFrame(columns=["hr", "token_id", "whale_flow", "whale_alpha"])

        alpha_by_wallet = trades.groupby("trader", observed=True)["alpha"].mean().rename("wallet_alpha")
        trades = trades.join(alpha_by_wallet, on="trader")
        trades["flow_raw"] = trades["signed_notional"]
        trades["weighted_flow"] = trades["signed_notional"] * (0.5 + trades["wallet_alpha"].fillna(0.0))
        trades["hr"] = trades["trade_ts"].dt.floor("h")

        hourly = (
            trades.groupby(["hr", "token_id"], observed=True)
            .agg(
                whale_flow_raw=("flow_raw", "sum"),
                whale_flow=("weighted_flow", "sum"),
                whale_alpha=("wallet_alpha", "mean"),
            )
            .reset_index()
        )
        return hourly

    def build_signal_frame(
        self,
        conn: duckdb.DuckDBPyConnection,
        panel: pd.DataFrame,
        context: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        if panel.empty:
            return pd.DataFrame()
        df = panel.copy()
        flow = self._build_whale_flow(conn)
        if flow.empty:
            df["whale_flow"] = 0.0
            df["whale_flow_raw"] = 0.0
            df["whale_alpha"] = 0.0
        else:
            df["hr"] = pd.to_datetime(df["ts"]).dt.floor("h")
            df = df.merge(flow, on=["hr", "token_id"], how="left")
            df["whale_flow"] = df["whale_flow"].fillna(0.0)
            df["whale_flow_raw"] = df["whale_flow_raw"].fillna(0.0)
            df["whale_alpha"] = df["whale_alpha"].fillna(0.0)
            df = df.drop(columns=["hr"])

        df["whale_flow_z"] = (df["whale_flow"] - df["whale_flow"].mean()) / (df["whale_flow"].std(ddof=0) + 1e-9)
        df["whale_alpha_z"] = (df["whale_alpha"] - df["whale_alpha"].mean()) / (df["whale_alpha"].std(ddof=0) + 1e-9)
        raw_flow_z = (df["whale_flow_raw"] - df["whale_flow_raw"].mean()) / (df["whale_flow_raw"].std(ddof=0) + 1e-9)
        df["raw_signal"] = 0.6 * raw_flow_z + 0.4 * (df["whale_flow_z"] * (1.0 + df["whale_alpha_z"].clip(lower=-1.0, upper=2.0)))
        df["signal"] = np.tanh(df["raw_signal"] / 2.0)
        df["confidence"] = (
            np.minimum(1.0, np.abs(df["whale_flow_z"]) / 3.0)
            * np.minimum(1.0, 0.5 + np.maximum(0.0, df["whale_alpha_z"]) / 2.0)
        ).clip(lower=0.05, upper=1.0)
        return df.sort_values(["ts", "token_id"]).reset_index(drop=True)
