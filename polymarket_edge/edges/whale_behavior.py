from __future__ import annotations

from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings


@dataclass
class WhaleConfig:
    whale_quantile: float = 0.95
    min_trades: int = 30
    early_fraction: float = 0.2
    n_clusters: int = 4


def _safe_ttest_ind(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    if float(a.std(ddof=0)) < 1e-12 and float(b.std(ddof=0)) < 1e-12:
        return 0.0, 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
    return float(t_stat), float(p_val)


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
            m.close_ts,
            r.winner_token_id
        FROM trades t
        LEFT JOIN markets m
            ON t.market_id = m.market_id
        LEFT JOIN resolutions r
            ON t.market_id = r.market_id
        WHERE t.trade_ts IS NOT NULL
          AND t.price IS NOT NULL
          AND t.size IS NOT NULL
          AND t.trader IS NOT NULL
        """
    ).df()


def _trade_alpha(row: pd.Series) -> float:
    if pd.isna(row["winner_token_id"]) or pd.isna(row["token_id"]):
        return np.nan
    outcome = 1.0 if str(row["token_id"]) == str(row["winner_token_id"]) else 0.0
    side = str(row["side"]).upper()
    if side == "BUY":
        return outcome - float(row["price"])
    if side == "SELL":
        return float(row["price"]) - outcome
    return np.nan


def run(conn: duckdb.DuckDBPyConnection, config: WhaleConfig = WhaleConfig()) -> dict[str, pd.DataFrame]:
    trades = _load_trade_frame(conn)
    if trades.empty:
        return {"summary": pd.DataFrame(), "wallet_clusters": pd.DataFrame(), "top_wallets": pd.DataFrame()}

    trades["trade_ts"] = pd.to_datetime(trades["trade_ts"])
    trades["close_ts"] = pd.to_datetime(trades["close_ts"])
    trades = trades.dropna(subset=["trade_ts", "notional", "price", "size"])
    if trades.empty:
        return {"summary": pd.DataFrame(), "wallet_clusters": pd.DataFrame(), "top_wallets": pd.DataFrame()}

    wallet_stats = (
        trades.groupby("trader", observed=True)
        .agg(
            trades=("trade_id", "nunique"),
            total_notional=("notional", "sum"),
            avg_trade_notional=("notional", "mean"),
            markets=("market_id", "nunique"),
            buy_ratio=("side", lambda s: float(np.mean(s.str.upper() == "BUY"))),
        )
        .reset_index()
    )
    wallet_stats = wallet_stats[wallet_stats["trades"] >= config.min_trades].copy()
    if wallet_stats.empty:
        return {"summary": pd.DataFrame(), "wallet_clusters": pd.DataFrame(), "top_wallets": pd.DataFrame()}

    cutoff = wallet_stats["total_notional"].quantile(config.whale_quantile)
    whales = wallet_stats[wallet_stats["total_notional"] >= cutoff]["trader"]
    trades["is_whale"] = trades["trader"].isin(set(whales))

    market_open = trades.groupby("market_id")["trade_ts"].min().rename("market_open")
    trades = trades.join(market_open, on="market_id")
    trades["open_to_close_s"] = (trades["close_ts"] - trades["market_open"]).dt.total_seconds()
    trades["since_open_s"] = (trades["trade_ts"] - trades["market_open"]).dt.total_seconds()
    trades["is_early"] = (trades["since_open_s"] >= 0) & (
        trades["since_open_s"] <= config.early_fraction * trades["open_to_close_s"]
    )

    trades["alpha_per_share"] = trades.apply(_trade_alpha, axis=1)
    trades["alpha_dollars"] = trades["alpha_per_share"] * trades["size"]
    eval_frame = trades.dropna(subset=["alpha_per_share"]).copy()
    early_eval = eval_frame[eval_frame["is_early"]].copy()

    whale_alpha = early_eval[early_eval["is_whale"]]["alpha_per_share"]
    non_alpha = early_eval[~early_eval["is_whale"]]["alpha_per_share"]
    if len(whale_alpha) > 20 and len(non_alpha) > 20:
        t_stat, p_val = _safe_ttest_ind(whale_alpha, non_alpha)
    else:
        t_stat, p_val = np.nan, np.nan

    cluster_cols = ["total_notional", "avg_trade_notional", "trades", "markets", "buy_ratio"]
    scaled = StandardScaler().fit_transform(wallet_stats[cluster_cols].astype(float))
    n_clusters = int(min(config.n_clusters, max(1, len(wallet_stats))))
    model = KMeans(n_clusters=n_clusters, random_state=7, n_init=10)
    wallet_stats["cluster"] = model.fit_predict(scaled)

    whale_details = wallet_stats[wallet_stats["trader"].isin(set(whales))].sort_values("total_notional", ascending=False)
    summary = pd.DataFrame(
        [
            {
                "edge_name": "whale_behavior",
                "n_trades_eval": len(early_eval),
                "n_whale_wallets": int(len(whales)),
                "mean_alpha_whale": float(whale_alpha.mean()) if len(whale_alpha) else np.nan,
                "mean_alpha_non_whale": float(non_alpha.mean()) if len(non_alpha) else np.nan,
                "whale_minus_non_tstat": t_stat,
                "whale_minus_non_pvalue": p_val,
            }
        ]
    )
    return {"summary": summary, "wallet_clusters": wallet_stats, "top_wallets": whale_details}
