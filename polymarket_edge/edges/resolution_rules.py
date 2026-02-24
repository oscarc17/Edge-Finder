from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import duckdb
import numpy as np
import pandas as pd
from scipy import stats
import warnings

from polymarket_edge.features import build_pre_resolution_frame


@dataclass
class ResolutionRuleConfig:
    horizon_hours: int = 24
    min_samples: int = 20
    quantiles: int = 5


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


def ambiguity_score(text: str | None) -> float:
    if not text:
        return 0.0
    low = text.lower()
    token_hits = sum(low.count(token) for token in AMBIGUITY_TERMS)
    punctuation = low.count(";") + low.count(":") + low.count("(")
    return (len(low) / 400.0) + 0.8 * token_hits + 0.2 * punctuation


def run(conn: duckdb.DuckDBPyConnection, config: ResolutionRuleConfig = ResolutionRuleConfig()) -> dict[str, object]:
    frame = build_pre_resolution_frame(conn, horizon_hours=config.horizon_hours)
    if frame.empty or len(frame) < config.min_samples:
        return {"summary": pd.DataFrame(), "market_candidates": pd.DataFrame()}

    text = frame["question"].fillna("") + " " + frame["description"].fillna("")
    frame["ambiguity_score"] = text.map(ambiguity_score)
    frame["liq_log"] = np.log1p(frame["liquidity_score"].fillna(0.0))
    frame["abs_error"] = frame["abs_error"].astype(float)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["abs_error", "ambiguity_score", "liq_log"])
    if frame.empty:
        return {"summary": pd.DataFrame(), "market_candidates": pd.DataFrame()}

    frame["ambiguity_bucket"] = pd.qcut(frame["ambiguity_score"], q=config.quantiles, labels=False, duplicates="drop")
    bucket = (
        frame.groupby("ambiguity_bucket", observed=True)
        .agg(
            mean_abs_error=("abs_error", "mean"),
            median_abs_error=("abs_error", "median"),
            n=("market_id", "count"),
        )
        .reset_index()
    )

    top = frame[frame["ambiguity_bucket"] == frame["ambiguity_bucket"].max()]["abs_error"]
    bottom = frame[frame["ambiguity_bucket"] == frame["ambiguity_bucket"].min()]["abs_error"]
    if len(top) > 5 and len(bottom) > 5:
        t_stat, p_val = _safe_ttest_ind(top, bottom)
    else:
        t_stat, p_val = np.nan, np.nan

    corr, corr_p = stats.spearmanr(frame["ambiguity_score"], frame["abs_error"])
    summary = pd.DataFrame(
        [
            {
                "edge_name": "resolution_rule_mispricing",
                "n_samples": len(frame),
                "spearman_corr_abs_error": corr,
                "spearman_pvalue": corr_p,
                "top_vs_bottom_tstat": t_stat,
                "top_vs_bottom_pvalue": p_val,
                "horizon_hours": config.horizon_hours,
            }
        ]
    )

    unresolved = conn.execute(
        """
        WITH latest AS (
            SELECT
                market_id,
                token_id,
                mid,
                spread_bps,
                liquidity_score,
                ROW_NUMBER() OVER (PARTITION BY token_id ORDER BY snapshot_ts DESC) AS rn
            FROM orderbook_snapshots
        ),
        yes_token AS (
            SELECT
                market_id,
                token_id,
                ROW_NUMBER() OVER (
                    PARTITION BY market_id
                    ORDER BY
                        CASE
                            WHEN lower(coalesce(outcome_label, '')) = 'yes' THEN 0
                            ELSE 1
                        END,
                        outcome_index
                ) AS rn
            FROM market_outcomes
        )
        SELECT
            m.market_id,
            m.question,
            m.description,
            m.category,
            l.mid AS prob,
            l.spread_bps,
            l.liquidity_score
        FROM markets m
        JOIN yes_token yt
            ON m.market_id = yt.market_id
           AND yt.rn = 1
        JOIN latest l
            ON yt.token_id = l.token_id
           AND l.rn = 1
        LEFT JOIN resolutions r
            ON m.market_id = r.market_id
        WHERE r.market_id IS NULL
        """
    ).df()
    if unresolved.empty:
        candidates = pd.DataFrame()
    else:
        unresolved["ambiguity_score"] = (unresolved["question"].fillna("") + " " + unresolved["description"].fillna("")).map(
            ambiguity_score
        )
        unresolved["candidate_score"] = (
            unresolved["ambiguity_score"] * unresolved["spread_bps"].fillna(0).clip(lower=0)
        ) / (np.log1p(unresolved["liquidity_score"].fillna(0)) + 1.0)
        candidates = unresolved.sort_values("candidate_score", ascending=False).head(50)

    return {"summary": summary, "bucket_detail": bucket, "market_candidates": candidates}
