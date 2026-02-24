from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import duckdb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from statsmodels.tsa.stattools import adfuller

from polymarket_edge.features import load_latest_yes_probabilities, load_yes_probability_panel


@dataclass
class CrossMarketConfig:
    similarity_threshold: float = 0.72
    min_common_observations: int = 5


def _related_pairs(latest: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for event_id, group in latest.groupby("event_id", dropna=True):
        if len(group) < 2:
            continue
        tfidf = TfidfVectorizer(stop_words="english")
        matrix = tfidf.fit_transform(group["question"].fillna(""))
        sim = cosine_similarity(matrix)
        local = group.reset_index(drop=True)
        for i, j in combinations(range(len(local)), 2):
            s = float(sim[i, j])
            if s < threshold:
                continue
            rows.append(
                {
                    "event_id": event_id,
                    "market_a": local.loc[i, "market_id"],
                    "market_b": local.loc[j, "market_id"],
                    "question_a": local.loc[i, "question"],
                    "question_b": local.loc[j, "question"],
                    "prob_a": local.loc[i, "prob"],
                    "prob_b": local.loc[j, "prob"],
                    "similarity": s,
                }
            )
    return pd.DataFrame(rows)


def _inequality_violation(row: pd.Series) -> float:
    qa = str(row["question_a"]).lower()
    qb = str(row["question_b"]).lower()
    pa = float(row["prob_a"])
    pb = float(row["prob_b"])
    if qa in qb and pb > pa:
        return pb - pa
    if qb in qa and pa > pb:
        return pa - pb
    return 0.0


def run(conn: duckdb.DuckDBPyConnection, config: CrossMarketConfig = CrossMarketConfig()) -> pd.DataFrame:
    latest = load_latest_yes_probabilities(conn)
    if latest.empty:
        return pd.DataFrame()
    pairs = _related_pairs(latest, config.similarity_threshold)
    if pairs.empty:
        return pairs

    ids = set(pairs["market_a"]).union(set(pairs["market_b"]))
    panel = load_yes_probability_panel(conn, ids)
    if panel.empty:
        return pd.DataFrame()
    panel["snapshot_ts"] = pd.to_datetime(panel["snapshot_ts"])

    stats_rows: list[dict[str, object]] = []
    for pair in pairs.itertuples(index=False):
        subset = panel[panel["market_id"].isin([pair.market_a, pair.market_b])]
        if subset.empty:
            continue
        pivot = (
            subset.pivot_table(index="snapshot_ts", columns="market_id", values="prob", aggfunc="mean")
            .dropna()
            .sort_index()
        )
        if len(pivot) < config.min_common_observations:
            continue
        spread = pivot[pair.market_a] - pivot[pair.market_b]
        if spread.std(ddof=0) == 0:
            continue
        try:
            adf_p = float(adfuller(spread, autolag="AIC")[1])
        except ValueError:
            adf_p = np.nan
        z = float((spread.iloc[-1] - spread.mean()) / spread.std(ddof=0))
        stats_rows.append(
            {
                "edge_name": "cross_market_inefficiency",
                "entity_id": f"{pair.market_a}|{pair.market_b}",
                "event_id": pair.event_id,
                "similarity": pair.similarity,
                "adf_pvalue": adf_p,
                "spread_mean": float(spread.mean()),
                "spread_std": float(spread.std(ddof=0)),
                "spread_zscore": z,
                "inequality_violation": _inequality_violation(pd.Series(pair._asdict())),
                "n_obs": len(pivot),
            }
        )
    return pd.DataFrame(stats_rows)
